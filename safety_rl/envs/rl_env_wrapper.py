# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
from gym import spaces


class RLEnvWrapper:
    def __init__(
        self,
        env,
        coef_goal_distance_penalty=-1.0,
        coef_goal_achieved_reward=1.0,
        coef_collision_penalty=-1.0,
    ):
        self._env = env
        self._env.continue_goal = False

        # Reward weights
        self._coef_goal_distance_penalty = coef_goal_distance_penalty
        self._coef_goal_achieved_reward = coef_goal_achieved_reward
        self._coef_collision_penalty = coef_collision_penalty
        self.last_robot_pos = np.zeros(shape=(3,), dtype=np.float32)

        # Update observation space
        goal_pos = np.ones(shape=(2,))
        observation_space = np.concatenate((self._env.observation_space.high, goal_pos))
        self.observation_space = spaces.Box(low=-observation_space, high=observation_space, dtype="float32")

    def _get_obs(self):
        env_obs = self._env._get_obs()
        return np.concatenate((env_obs, self.goal_pos[:2] - self.robot_pos[:2]))

    def reset(self):
        self._env.reset()
        return self._get_obs()

    def step(self, action):
        self.last_robot_pos = self.robot_pos.copy()
        next_obs, reward, done, info = self._env.step(action)
        info["cost_hazards"] = self.hazard_cost()
        info["cost_pillars"] = self.pillar_cost()
        return self._get_obs(), self._compute_reward(), done, info

    def _baseline_reward(self):
        # Ignore reward comes from safety-gym for improving readability and clarity
        reward = self._coef_goal_achieved_reward * float(self.goal_met())
        reward += self._coef_collision_penalty * float(self._is_collided())
        reward += self._coef_goal_distance_penalty * np.linalg.norm(self.goal_pos[:2] - self.robot_pos[:2])
        return reward

    def _compute_reward(self):
        # Ignore reward comes from safety-gym for improving readability and clarity
        return self._baseline_reward()

    def __getattr__(self, attr):
        # Enable to call functions defined in `self._env`
        return self._env.__getattribute__(attr)
