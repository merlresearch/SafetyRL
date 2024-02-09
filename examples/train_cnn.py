# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
import os
from time import sleep

import numpy as np
import tensorflow as tf
from tf2rl.experiments.utils import frames_to_gif

from examples.config import get_common_argument, get_config, get_config_argument
from safety_rl.algos.waypoints_generator import WayPointsGenerator, move_toward_kth_waypoint
from safety_rl.envs.engine_wrapper import EngineWrapper
from safety_rl.envs.goselo_env import GoseloEnv
from safety_rl.experiments.dataset import load_dataset
from safety_rl.experiments.generate_dataset import decode_way_points_to_world_coord
from safety_rl.experiments.trainer import Trainer
from safety_rl.misc.logger import initialize_logger

for gpu in tf.config.experimental.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)


class ForceRobotMoveEnv(GoseloEnv):
    def step(self, action):
        next_robot_pos = self.robot_pos[:2].copy()
        next_robot_pos += action
        self.set_robot_pos(next_robot_pos)
        self.sim.forward()

        info = {}
        info["cost_hazards"] = self.hazard_cost()
        info["cost_pillars"] = self.pillar_cost()
        info["goal_met"] = self.goal_met()
        collided, collided_idx = self._is_collided_with_hazards()
        info["collided"] = collided
        info["collided_idx"] = collided_idx

        discrete_robot_pos = self.to_img_pos(self.robot_pos)
        discrete_goal_pos = self.to_img_pos(self.goal_pos)
        if np.array_equal(discrete_robot_pos, discrete_goal_pos) or self.goal_met():
            return None, self._compute_reward(), True, info

        # Record path history
        discrete_robot_pos = self.to_img_pos(self.robot_pos)
        if (
            abs(discrete_robot_pos[0]) >= self._field_img.shape[0]
            or abs(discrete_robot_pos[1]) >= self._field_img.shape[1]
        ):
            self._logger.debug(
                "Discretized value {} violates input state space {}".format(
                    discrete_robot_pos, self._field_img.shape[:2]
                )
            )
            return self._get_obs(), self._compute_reward(), True, {}

        self._pathlog_map[discrete_robot_pos[1]][discrete_robot_pos[0]] += 1
        return self._get_obs(), self._compute_reward(), False, info


class CustomTrainer(Trainer):
    def __init__(self, *args, vis_env=None, sleep_for_vis=1 / 30.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._vis_env = vis_env
        self._sleep_for_vis = sleep_for_vis

    def _validate_env(self, env):
        assert isinstance(env, ForceRobotMoveEnv)

    def evaluate_in_env(self, epoch, n_episodes):
        avg_test_return = 0.0
        avg_n_goal = 0.0
        avg_n_collision = 0.0
        avg_cost_hazards = 0.0
        avg_cost_pillars = 0.0

        for i in range(n_episodes):
            episode_return = 0.0
            frames = []
            obs = self._env.reset()
            self._vis_env.reset()
            self._vis_env.set_goal_pos(self._env.goal_pos[:2])
            if len(self._env.hazards_pos) > 0:
                self._vis_env.set_hazards_pos(np.stack(self._env.hazards_pos)[:, :2])
            if len(self._env.pillars_pos) > 0:
                self._vis_env.set_pillars_pos(np.stack(self._env.pillars_pos)[:, :2])
            if len(self._env.gremlins_obj_pos) > 0:
                self._vis_env.set_gremlins_obj_pos(np.stack(self._env.gremlins_obj_pos)[:, :2])

            for _ in range(self._episode_max_steps):
                self._vis_env.set_robot_pos(self._env.robot_pos[:2])

                abs_way_points_goselo_coord = self._way_points_generator.get_action(obs, test=True, policy="all")
                action = move_toward_kth_waypoint(
                    abs_way_points=abs_way_points_goselo_coord,
                    current_pos=self._env.robot_pos[:2],
                    goal_pos=self._env.goal_pos[:2],
                    k=1,
                )

                abs_way_points_world_coord = decode_way_points_to_world_coord(
                    current_pos=self._env.robot_pos[:2],
                    goal_pos=self._env.goal_pos[:2],
                    abs_vertices=abs_way_points_goselo_coord.reshape(-1, 2),
                )
                abs_way_points_world_coord += self._env.robot_pos[:2]

                self._env.set_reference_path_world_coord(abs_way_points_world_coord)
                self._vis_env.set_reference_path_world_coord(abs_way_points_world_coord)

                next_obs, reward, done, info = self._env.step(action)
                avg_n_goal += info["goal_met"]
                avg_cost_hazards += info["cost_hazards"]
                avg_cost_pillars += info["cost_pillars"]
                avg_n_collision += info["collided"]

                if self._save_test_movie:
                    frame = self._env.render(mode="rgb_array")
                    frames.append(frame)
                elif self._show_test_progress:
                    self._env.render()
                    self._vis_env.render()
                    sleep(self._sleep_for_vis)

                episode_return += reward
                obs = next_obs
                if done:
                    break
            prefix = "epoch_{0:08d}_epi_{1:02d}_return_{2:010.4f}".format(epoch, i, episode_return)
            if self._save_test_movie:
                frames_to_gif(frames, prefix, self._output_dir)
            avg_test_return += episode_return
        avg_test_return /= n_episodes
        avg_n_goal /= n_episodes
        avg_cost_hazards /= n_episodes
        avg_cost_pillars /= n_episodes
        avg_n_collision /= n_episodes

        return avg_test_return, avg_n_goal, avg_cost_hazards, avg_cost_pillars, avg_n_collision


def main():
    parser = Trainer.get_argument()
    parser = get_config_argument(parser)
    parser = get_common_argument(parser)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--field-size", type=int, default=2)
    parser.add_argument("--sleep-for-vis", type=float, default=1 / 30.0)
    parser.add_argument("--eval-dataset-dir", type=str, default=None)
    parser.set_defaults(dataset_prefix=None)
    parser.set_defaults(episode_steps=300)
    args = parser.parse_args()

    # Get configurations
    config, exp_name = get_config(
        robot_type=args.robot_type,
        field_size=args.field_size,
        place_room=args.place_room,
        no_obs=args.no_obs,
        room_type=args.room_type,
        hazards_num=args.hazards_num,
        pillars_num=args.pillars_num,
        gremlins_num=args.gremlins_num,
    )

    if args.dataset_dir is not None:
        assert args.dataset_dir is not None and os.path.isdir(args.dataset_dir)
        assert args.eval_dataset_dir is not None and os.path.isdir(args.eval_dataset_dir)
        dataset_dir = args.dataset_dir
        eval_dataset_dir = args.eval_dataset_dir
    else:
        dataset_dir = os.path.join(args.root_dir, "dataset", exp_name, "waypoints_generator")
        eval_dataset_dir = dataset_dir + "_eval"

    args.model_dir = os.path.join(dataset_dir, "model")

    logdir = os.path.join(dataset_dir, "model")
    logger = initialize_logger(logging_level=logging.INFO, output_dir=logdir, save_log=False)

    # Load dataset
    max_size = 1 if args.rollout_only else None
    x_train, y_train, _, _, output_dim = load_dataset(dataset_dir=dataset_dir, split_ratio=0.0, max_size=max_size)
    if not args.rollout_only:
        x_test, y_test, _, _, _ = load_dataset(dataset_dir=eval_dataset_dir, split_ratio=0.0)
    input_shape = x_train.shape[1:]

    env = ForceRobotMoveEnv(out_img_size=input_shape[:2], logging_level=args.logging_level, config=config, img_reso=0.5)
    vis_env = EngineWrapper(config=config, visualize_waypoints=True, visualize_only_raw_waypoints=True)

    logger.info("NN output is {} way points".format(output_dim / 2))
    way_points_generator = WayPointsGenerator(input_shape=input_shape, output_dim=output_dim, lr=args.lr)

    trainer = CustomTrainer(
        args, way_points_generator, env, logdir=logdir, vis_env=vis_env, sleep_for_vis=args.sleep_for_vis
    )
    if args.rollout_only:
        trainer._show_test_progress = True
        trainer.evaluate_policy_continuously()
    else:
        trainer.supervised_learning(x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    main()
