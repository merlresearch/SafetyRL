# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import os
import random

import numpy as np
import tensorflow as tf
from tf2rl.experiments.utils import frames_to_gif
from tf2rl.misc.initialize_logger import initialize_logger

from safety_rl.algos.waypoints_generator import move_toward_kth_waypoint
from safety_rl.envs.goselo_env import GoseloEnv
from safety_rl.misc.prepare_output_dir import prepare_output_dir

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def get_suffix(args):
    suffix = "_{}_{}".format(args.dataset, args.nn_class)
    if args.rollout_only:
        suffix += "_rollout_only"
    if args.dir_suffix != "":
        suffix += "_{}".format(args.dir_suffix)
    return suffix


class Trainer:
    def __init__(self, args, way_points_generator, env, logdir="results"):
        self._way_points_generator = way_points_generator
        self._set_from_args(args)

        # prepare log directory
        self._output_dir = prepare_output_dir(args=args, user_specified_dir=logdir, without_timestamp=True)

        self.logger = initialize_logger(output_dir=self._output_dir + "/")

        self._validate_env(env)
        self._env = env
        self._set_from_args(args)

        # Save and restore model
        self._checkpoint = tf.train.Checkpoint(way_points_generator=self._way_points_generator.model)
        self.checkpoint_manager = tf.train.CheckpointManager(
            self._checkpoint, directory=self._output_dir, max_to_keep=5
        )

        if args.model_dir is not None:
            assert os.path.isdir(args.model_dir)
            self._latest_path_ckpt = tf.train.latest_checkpoint(args.model_dir)
            assert self._latest_path_ckpt is not None, "Failed to find latest checkpoint from {}".format(args.model_dir)
            self._checkpoint.restore(self._latest_path_ckpt)
            self.logger.info("Restored {}".format(self._latest_path_ckpt))

        # prepare TensorBoard output
        self.writer = tf.summary.create_file_writer(self._output_dir)
        self.writer.set_as_default()

    def _validate_env(self, env):
        assert isinstance(env, GoseloEnv)

    def supervised_learning(self, x_train, y_train, x_test, y_test):
        epoch = 0
        indices = list(range(x_train.shape[0]))
        n_batches = x_train.shape[0] // self._way_points_generator.batch_size

        while epoch < self._epochs:
            random.shuffle(indices)
            epoch += 1
            tf.summary.experimental.set_step(epoch)
            self._way_points_generator.reset_stats()
            for batch in range(n_batches):
                start = batch * self._way_points_generator.batch_size
                end = start + self._way_points_generator.batch_size
                self._way_points_generator.train(x_train[indices[start:end]], y_train[indices[start:end]])

            # Write results to TensorBoard
            tf.summary.scalar(name="GOSELO/train_loss", data=self._way_points_generator.train_loss)
            self.logger.info("Epoch: {}, Train Loss: {:.8f}".format(epoch, self._way_points_generator.train_loss))

            if epoch % self._test_interval == 0:
                self._way_points_generator.reset_stats()
                self.evaluate_supervised_learning(x_test, y_test, epoch)

            if epoch % self._save_model_interval == 0:
                self.checkpoint_manager.save()

            if self._test_env_interval is not None and epoch % self._test_env_interval == 0:
                avg_test_return, avg_n_goal, avg_hazards_cost, avg_pillars_cost, avg_n_collision = self.evaluate_in_env(
                    int(epoch), int(self._test_episodes)
                )
                tf.summary.scalar(name="Common/average_test_return", data=avg_test_return)
                tf.summary.scalar(name="Common/average_reach_goal", data=avg_n_goal)
                tf.summary.scalar(name="Common/hazards_cost", data=avg_hazards_cost)
                tf.summary.scalar(name="Common/pillars_cost", data=avg_pillars_cost)
                tf.summary.scalar(name="Common/average_n_collision", data=avg_n_collision)
                self.logger.info(
                    "Epoch: {}, Test Return: {:.3f} Goal Rate: {:.3f} Hazards Cost: {:.3f} Pillars Cost: {:.3f} N_collision: {:.3f} over {} run".format(
                        epoch,
                        avg_test_return,
                        avg_n_goal,
                        avg_hazards_cost,
                        avg_pillars_cost,
                        avg_n_collision,
                        self._test_episodes,
                    )
                )

    def evaluate_supervised_learning(self, x_test, y_test, epoch):
        n_batches_test = x_test.shape[0] // self._way_points_generator.batch_size
        for test_batch in range(n_batches_test):
            start_test = test_batch * self._way_points_generator.batch_size
            end_test = start_test + self._way_points_generator.batch_size
            self._way_points_generator.evaluate(x_test[start_test:end_test], y_test[start_test:end_test])
        self.logger.info("Epoch: {}, Test Cost: {:.5f}".format(epoch, self._way_points_generator.test_loss))
        tf.summary.scalar(name="GOSELO/test_loss", data=self._way_points_generator.test_loss)

    def evaluate_policy_continuously(self):
        if self._model_dir is None:
            self.logger.error("Please specify model directory by passing command line argument `--model-dir`")
            exit(-1)

        self.evaluate_in_env(epoch=0, n_episodes=10)
        while True:
            latest_path_ckpt = tf.train.latest_checkpoint(self._model_dir)
            if self._latest_path_ckpt != latest_path_ckpt:
                self._latest_path_ckpt = latest_path_ckpt
                self._checkpoint.restore(self._latest_path_ckpt)
                self.logger.info("Restored {}".format(self._latest_path_ckpt))
            self.evaluate_in_env(epoch=0, n_episodes=10)

    def evaluate_in_env(self, epoch, n_episodes):
        avg_test_return = 0.0
        avg_n_goal = 0.0
        avg_cost_hazards = 0.0
        avg_cost_pillars = 0.0

        for i in range(n_episodes):
            episode_return = 0.0
            frames = []
            obs = self._env.reset()
            for _ in range(self._episode_max_steps):
                way_points = self._way_points_generator.get_action(obs, test=True, policy="all")
                action = move_toward_kth_waypoint(
                    abs_way_points=way_points, current_pos=self._env.robot_pos[:2], goal_pos=self._env.goal_pos[:2]
                )
                self._env.set_reference_path_world_coord(np.reshape(way_points, (-1, 2)))

                next_obs, reward, done, info = self._env.step(action)
                avg_n_goal += info["goal_met"]
                avg_cost_hazards += info["cost_hazards"]
                avg_cost_pillars += info["cost_pillars"]

                if self._save_test_movie:
                    frame = self._env.render(mode="rgb_array")
                    frames.append(frame)
                elif self._show_test_progress:
                    self._env.render()
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

        return avg_test_return, avg_n_goal, avg_cost_hazards, avg_cost_pillars

    def _set_from_args(self, args):
        # Supervised learning settings
        self._epochs = args.epochs
        self._max_size = args.max_size
        # Experiments settings
        self._save_model_interval = args.save_model_interval
        self._model_dir = args.model_dir
        # Evaluation for supervised learning settings
        self._test_interval = args.test_interval
        # Evaluation with Env settings
        self._test_env_interval = args.test_env_interval
        self._show_test_progress = args.show_test_progress
        self._save_test_path = args.save_test_path
        self._save_test_movie = args.save_test_movie
        self._episode_max_steps = args.episode_max_steps
        self._test_episodes = args.test_episodes

    @staticmethod
    def get_argument(parser=None):
        if parser is None:
            parser = argparse.ArgumentParser(conflict_handler="resolve")
        # Supervised learning settings
        parser.add_argument("--dataset-dir", type=str, default=None, help="Path to GOSELO dataset directory")
        parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
        parser.add_argument("--test-split-ratio", type=float, default=0.1, help="Ratio for test")
        parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
        parser.add_argument(
            "--max-size", type=int, default=20000, help="Maximum number of data to load from GOSELO dataset"
        )
        # Experiments settings
        parser.add_argument("--gpu", type=int, default=0, help="GPU id")
        parser.add_argument("--save-model-interval", type=int, default=int(10), help="Interval to save model by epochs")
        parser.add_argument("--model-dir", type=str, default=None, help="Directory to restore model")
        parser.add_argument("--dir-suffix", type=str, default="", help="Suffix for directory that contains results")
        parser.add_argument("--rollout-only", action="store_true")
        # Evaluation for supervised learning settings
        parser.add_argument("--test-interval", type=int, default=int(5), help="Interval to evaluate trained model")
        # Evaluation with Env settings
        parser.add_argument("--test-env-interval", type=int, default=None, help="Interval to evaluate trained model")
        parser.add_argument("--evaluate-in-env", action="store_true", help="Evaluate learned policy with GoseloEnv")
        parser.add_argument("--episode-max-steps", type=int, default=int(1e3), help="Maximum steps in an episode")
        parser.add_argument("--show-test-progress", action="store_true", help="Call `render` in evaluation process")
        parser.add_argument("--test-episodes", type=int, default=5, help="Number of episodes to evaluate at once")
        parser.add_argument("--save-test-path", action="store_true", help="Save trajectories of evaluation")
        parser.add_argument("--save-test-movie", action="store_true", help="Save rendering results")
        # others
        parser.add_argument(
            "--logging-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO", help="Logging level"
        )
        return parser
