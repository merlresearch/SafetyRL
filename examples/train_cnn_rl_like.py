# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
import os

import numpy as np
import tensorflow as tf
from cpprb import ReplayBuffer
from tf2rl.experiments.trainer import Trainer
from tf2rl.misc.prepare_output_dir import prepare_output_dir

from safety_rl.algos.waypoints_generator import move_toward_kth_waypoint
from safety_rl.experiments.generate_dataset import decode_way_points_to_world_coord, to_goselo_format
from safety_rl.misc.logger import initialize_logger
from safety_rl.path_planner.a_star_safety_gym import AStarSafetyGym


class WayPointTrainer(Trainer):
    def __init__(self, way_points_generator, env, vis_env, args):
        self._set_from_args(args)
        self._way_points_generator = way_points_generator
        self._env = env
        self._vis_env = vis_env

        # prepare log directory
        self._output_dir = prepare_output_dir(
            args=args, user_specified_dir=self._logdir, suffix="{}".format(args.dir_suffix)
        )
        self.logger = initialize_logger(
            logging_level=logging.getLevelName(args.logging_level), output_dir=self._output_dir
        )

        if args.evaluate:
            assert args.model_dir is not None
        self._set_check_point(args.model_dir)

        # prepare TensorBoard output
        self.writer = tf.summary.create_file_writer(self._output_dir)
        self.writer.set_as_default()

    def _set_check_point(self, model_dir):
        # Save and restore model
        self._checkpoint = tf.train.Checkpoint(way_points_generator=self._way_points_generator.model)
        self.checkpoint_manager = tf.train.CheckpointManager(
            self._checkpoint, directory=self._output_dir, max_to_keep=5
        )

        if model_dir is not None:
            assert os.path.isdir(model_dir)
            self._latest_path_ckpt = tf.train.latest_checkpoint(model_dir)
            self._checkpoint.restore(self._latest_path_ckpt)
            self.logger.info("Restored {}".format(self._latest_path_ckpt))

    def __call__(self, samples_eval=None):
        total_steps = 0
        tf.summary.experimental.set_step(total_steps)

        replay_buffer_dict = {
            "size": self._n_eval_data,
            "default_dtype": np.float32,
            "env_dict": {
                "inputs": {"shape": self._env.observation_space.shape},
                "goal_dists": {"shape": (1,)},
                "way_points": {"shape": (self._way_points_generator.output_dim,)},
            },
        }

        self._exploration_replay_buffer = ReplayBuffer(**replay_buffer_dict)
        replay_buffer_dict["size"] = 50000
        self._all_transiton_replay_buffer = ReplayBuffer(**replay_buffer_dict)

        path_planner = AStarSafetyGym(self._env, resolution=self._resolution, show_process=False)

        if samples_eval is not None:
            assert "inputs" in samples_eval and "way_points" in samples_eval
            assert samples_eval["inputs"].shape[0] == samples_eval["way_points"].shape[0] == self._n_eval_data
            assert samples_eval["inputs"].shape[1:] == self._env.observation_space.shape
            assert samples_eval["way_points"].shape[1] == self._way_points_generator.output_dim
        else:
            self.logger.info("Generating evaluation {} data...".format(self._n_eval_data))
            samples_eval = self._collect_data(path_planner, self._n_eval_data)

        self.evaluate_supervised_learning(
            x_test=(samples_eval["inputs"], samples_eval["goal_dists"]),
            y_test=samples_eval["way_points"],
            n_trained=total_steps,
        )

        n_request_data = 1000
        while total_steps < self._max_steps:

            samples = self._collect_data(path_planner, n_request_data)
            assert samples["inputs"].shape[0] == n_request_data

            self._all_transiton_replay_buffer.add(
                inputs=samples["inputs"], way_points=samples["way_points"], goal_dists=samples["goal_dists"]
            )

            total_steps += n_request_data
            tf.summary.experimental.set_step(total_steps)

            if total_steps < self._n_warm_up:
                continue

            for _ in range(n_request_data):
                samples = self._all_transiton_replay_buffer.sample(self._way_points_generator.batch_size)
                self._way_points_generator.train(samples["inputs"], samples["way_points"])
            self.logger.info(
                "n_trained: {}, Train Cost: {:.5f}".format(total_steps, self._way_points_generator.train_loss)
            )
            tf.summary.scalar("GOSELO/training_loss", data=self._way_points_generator.train_loss)
            self._way_points_generator.reset_stats()

            if total_steps % self._test_interval == 0:
                self.evaluate_supervised_learning(
                    x_test=(samples_eval["inputs"], samples_eval["goal_dists"]),
                    y_test=samples_eval["way_points"],
                    n_trained=total_steps,
                )

            if self._test_env_interval is not None and total_steps % self._test_env_interval == 0:
                self.evaluate_in_env(10)

            if total_steps % self._save_model_interval == 0:
                self.checkpoint_manager.save()

        self.checkpoint_manager.save()
        tf.summary.flush()

    def _collect_data(self, path_planner, n_request_data):
        collected_steps = 0

        self._exploration_replay_buffer.clear()
        while collected_steps < n_request_data:
            # Generate optimal path using A*
            way_points_world_coord = path_planner.planning()

            # Convert to GOSELO expression
            inputs, way_points, goal_dists = to_goselo_format(
                self._env, way_points_world_coord, self._n_way_points, self._show_progress
            )

            if len(inputs) == 0:
                continue

            collected_steps += inputs.shape[0]
            self._exploration_replay_buffer.add(inputs=inputs, way_points=way_points, goal_dists=goal_dists)
        return self._exploration_replay_buffer._encode_sample(np.arange(n_request_data))

    def evaluate_supervised_learning(self, x_test, y_test, n_trained):
        n_batches_test = x_test[0].shape[0] // self._way_points_generator.batch_size
        for test_batch in range(n_batches_test):
            start_test = test_batch * self._way_points_generator.batch_size
            end_test = start_test + self._way_points_generator.batch_size
            self._way_points_generator.evaluate(x_test[0][start_test:end_test], y_test[start_test:end_test])
        self.logger.info("n_trained: {}, Test Cost: {:.5f}".format(n_trained, self._way_points_generator.test_loss))
        tf.summary.scalar(name="GOSELO/test_loss", data=self._way_points_generator.test_loss)
        self._way_points_generator.reset_stats()

    def evaluate_policy(self, total_steps, n_episodes=100):
        avg_test_return = 0.0

        for i in range(n_episodes):
            episode_return = 0.0
            obs = self._env.reset()
            self._vis_env.reset()
            self._vis_env.set_goal_pos(self._env.goal_pos[:2])
            self._vis_env.set_robot_pos(self._env.robot_pos[:2])
            if len(self._env.hazards_pos) > 0:
                self._vis_env.set_hazards_pos(np.stack(self._env.hazards_pos)[:, :2])
            if len(self._env.pillars_pos) > 0:
                self._vis_env.set_pillars_pos(np.stack(self._env.pillars_pos)[:, :2])
            if len(self._env.gremlins_obj_pos) > 0:
                self._vis_env.set_gremlins_obj_pos(np.stack(self._env.gremlins_obj_pos)[:, :2])

            for _ in range(self._episode_max_steps):
                abs_way_points_goselo_coord = self._way_points_generator.get_action(obs, test=True, policy="all")
                action = move_toward_kth_waypoint(
                    abs_way_points=abs_way_points_goselo_coord,
                    current_pos=self._env.robot_pos[:2],
                    goal_pos=self._env.goal_pos[:2],
                    k=3,
                )

                abs_way_points_world_coord = decode_way_points_to_world_coord(
                    current_pos=self._env.robot_pos[:2],
                    goal_pos=self._env.goal_pos[:2],
                    abs_vertices=abs_way_points_goselo_coord.reshape(-1, 2),
                )
                abs_way_points_world_coord = abs_way_points_world_coord.reshape(-1, 2)
                abs_way_points_world_coord += self._env.robot_pos[:2]

                self._env.set_reference_path_world_coord(abs_way_points_world_coord)
                self._vis_env.set_reference_path_world_coord(abs_way_points_world_coord)
                self._vis_env.set_robot_pos(self._env.robot_pos[:2])

                if self._show_test_progress:
                    self._env.render()
                    self._vis_env.render()

                next_obs, reward, done, _ = self._env.step(action)

                episode_return += reward
                obs = next_obs
                if done:
                    break

            avg_test_return += episode_return
        avg_test_return /= n_episodes

        return avg_test_return

    def _set_from_args(self, args):
        super()._set_from_args(args)
        self._resolution = args.resolution
        self._n_way_points = args.n_way_points
        self._n_warm_up = args.n_warm_up
        self._n_eval_data = args.n_eval_data
        self._test_env_interval = args.test_env_interval

    @staticmethod
    def get_argument(parser=None):
        parser = Trainer.get_argument(parser)
        parser.add_argument(
            "--dataset-prefix", type=str, default="results/dataset/20190620T183624", help="Prefix to GOSELO dataset"
        )
        parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
        parser.add_argument("--n-way-points", type=int, default=10)
        parser.add_argument("--resolution", type=float, default=0.1)
        parser.add_argument("--n-warm-up", type=int, default=10000)
        parser.add_argument("--n-eval-data", type=int, default=10000)
        parser.add_argument("--test-env-interval", type=int, default=None)
        return parser


def main():
    from examples.config import get_config, get_config_argument
    from examples.train_cnn import ForceRobotMoveEnv
    from safety_rl.algos.waypoints_generator import WayPointsGenerator
    from safety_rl.envs.engine_wrapper import EngineWrapper
    from safety_rl.experiments.dataset import load_dataset

    parser = WayPointTrainer.get_argument()
    parser = get_config_argument(parser)
    parser.add_argument("--field-size", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-data-dir", type=str, default=None, help="Path to directory that contains test data")
    parser.add_argument("--plot-goal-pos", action="store_true")
    parser.add_argument("--rollout-only", action="store_true")
    parser.add_argument("--logdir", type=str, default="results")
    parser.set_defaults(dataset_prefix=None)
    parser.set_defaults(episode_max_steps=300)
    args = parser.parse_args()

    logger = initialize_logger(logging_level=logging.INFO, save_log=False)

    # Load dataset
    if args.dataset_prefix is None:
        input_shape = (64, 64, 6)
        output_dim = 2 * args.n_way_points
        samples_eval = None
    else:
        _, _, x_test, y_test, output_dim = load_dataset(
            max_size=args.n_eval_data, dataset_name_prefix=args.dataset_prefix, split_ratio=1
        )
        input_shape = x_test[0].shape[1:]
        assert x_test[0].shape[0] == args.n_eval_data
        samples_eval = {"inputs": x_test[0], "goal_dists": x_test[1], "way_points": y_test}

    # Prepare Env.
    config, _ = get_config(robot_type="point", field_size=args.field_size)
    env = ForceRobotMoveEnv(out_img_size=input_shape[:2], logging_level=args.logging_level, config=config, img_reso=0.5)

    vis_env = EngineWrapper(config=config, visualize_waypoints=True)

    logger.info("NN output is {} way points".format(args.n_way_points))
    way_points_generator = WayPointsGenerator(
        input_shape=env.observation_space.shape, batch_size=args.batch_size, output_dim=output_dim, lr=args.lr
    )

    trainer = WayPointTrainer(env=env, vis_env=vis_env, way_points_generator=way_points_generator, args=args)
    if args.rollout_only:
        trainer._show_test_progress = True
        trainer.evaluate_policy_continuously()
    else:
        trainer(samples_eval=samples_eval)


if __name__ == "__main__":
    main()
