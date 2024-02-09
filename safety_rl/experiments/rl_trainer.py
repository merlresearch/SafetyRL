# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import tensorflow as tf
from tf2rl.experiments.trainer import Trainer
from tf2rl.experiments.utils import frames_to_gif


class RLTrainer(Trainer):
    """
    ゴール率とステップ数を可視化できるようにこの部分だけevaluationだけ変更
    """

    def evaluate_policy(self, total_steps):
        tf.summary.experimental.set_step(total_steps)

        avg_test_return = 0.0
        n_reach_goal = 0
        n_collision = 0
        goal_steps = 0
        hazards_cost = 0
        pillars_cost = 0
        for i in range(self._test_episodes):
            episode_return = 0.0
            obs = self._test_env.reset()
            frames = []
            for n_transition in range(self._episode_max_steps):
                action = self._policy.get_action(obs, test=True)
                next_obs, reward, done, info = self._test_env.step(action)

                if self._show_test_progress:
                    self._test_env.render()

                if self._save_test_movie:
                    frames.append(self._test_env.render(mode="rgb_array"))

                goal_steps += 1
                episode_return += reward
                n_collision += int(info["collided"])
                hazards_cost += float(info["cost_hazards"])
                pillars_cost += float(info["cost_pillars"])
                obs = next_obs
                if done:
                    if info["goal_met"]:
                        n_reach_goal += 1
                    break

            if self._save_test_movie:
                prefix = "step_{0:08d}_epi_{1:02d}_return_{2:010.4f}".format(total_steps, i, episode_return)
                frames_to_gif(frames, prefix, self._output_dir)

            avg_test_return += episode_return
        tf.summary.scalar(name="Common/average_reach_goal", data=n_reach_goal / self._test_episodes)
        tf.summary.scalar(name="Common/average_steps_to_reach_goal", data=goal_steps / self._test_episodes)
        tf.summary.scalar(name="Common/average_n_collision", data=n_collision / self._test_episodes)
        tf.summary.scalar(name="Common/hazards_cost", data=hazards_cost / self._test_episodes)
        tf.summary.scalar(name="Common/pillars_cost", data=pillars_cost / self._test_episodes)
        return avg_test_return / self._test_episodes
