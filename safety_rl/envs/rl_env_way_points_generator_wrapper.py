# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
from gym import spaces

from safety_rl.envs.rl_env_wrapper import RLEnvWrapper
from safety_rl.envs.rl_interface_env import RLInterfaceEnv
from safety_rl.experiments.generate_dataset import decode_way_points_to_world_coord
from safety_rl.path_planner.utils import divide_path, find_nearest_point


class RLEnvWayPointsGeneratorWrapper(RLEnvWrapper):
    def __init__(
        self,
        way_points_generator,
        *args,
        input_raw_way_points=False,
        threshold_min_dist_to_reference_path=0.3,
        n_own_path_division=3,
        coef_progress_index_reward=0.5,
        coef_reference_distance_penalty=-1.0,
        input_cnn_feature=False,
        **kwargs
    ):
        self._way_points_generator = way_points_generator

        self._threshold_min_dist_to_reference_path = threshold_min_dist_to_reference_path
        self._n_own_path_division = n_own_path_division

        self._coef_progress_index_reward = coef_progress_index_reward
        self._coef_reference_distance_penalty = coef_reference_distance_penalty

        super().__init__(*args, **kwargs)
        assert isinstance(self._env, RLInterfaceEnv)

        self._input_raw_way_points = input_raw_way_points
        self._input_cnn_feature = input_cnn_feature

        # Update observation space
        way_points = np.ones(shape=(self._n_way_points * 2))
        observation_space = np.concatenate((self._env.observation_space.high, way_points))
        if self._input_cnn_feature:
            assert len(observation_space.shape) == 1
            observation_space = np.ones(
                shape=(observation_space.shape[0] + way_points_generator.get_feature_size()), dtype=np.float32
            )
        self.observation_space = spaces.Box(low=-observation_space, high=observation_space, dtype=np.float32)

    def _get_obs(self):
        env_obs = self._env._get_obs()

        if self._input_raw_way_points:
            relative_way_points = self._raw_reference_path_world_coord.copy()
        else:
            nearest_idx, _ = find_nearest_point(self.robot_pos[:2], self._reference_path_world_coord)
            relative_way_points = self._reference_path_world_coord - self.robot_pos[:2]

            # Expand array so as not to exceed the index
            if nearest_idx + self._n_way_points > self._reference_path_world_coord.shape[0]:
                relative_way_points = np.concatenate(
                    (
                        relative_way_points,
                        np.repeat(np.expand_dims(relative_way_points[-1], axis=0), self._n_way_points, axis=0),
                    ),
                    axis=0,
                )

            relative_way_points = relative_way_points[nearest_idx : nearest_idx + self._n_way_points]
        obs = np.concatenate((env_obs, relative_way_points.flatten()))

        if self._input_cnn_feature:
            goselo_img = self._env.get_goselo_img()
            cnn_feature = self._way_points_generator.extract_feature(
                (goselo_img, np.array([self.goal_dist], dtype=np.float32))
            )
            return np.concatenate((obs, cnn_feature))
        else:
            return obs

    def reset(self):
        """
        Update reference path when reset is called
        """
        obs = super().reset()
        self._update_reference_path()
        return obs

    def step(self, action):
        next_obs, rew, done, info = super().step(action)
        info["update_reference"] = False
        if self._is_update_reference():
            info["update_reference"] = True
            self._update_reference_path()
        done = done or info["goal_met"]
        return next_obs, rew, done, info

    def _compute_reward(self):
        # Ignore reward comes from safety-gym for improving readability and clarity
        reward = self._coef_goal_achieved_reward * float(self.goal_met())
        reward += self._coef_collision_penalty * float(self._is_collided())

        # Reward from way points
        reward += self._coef_progress_index_reward * self._compute_progress_index()
        reward += self._coef_reference_distance_penalty * self._compute_reference_distance_error()
        return reward

    def _compute_progress_index(self):
        """
        Compute how many indices does the agent progress in one step.
        If positive, it will get reward, and vice versa.

        :return (int): Number of indices the agent progresses in one step.
        """
        assert self._reference_path_world_coord is not None and self._reference_path_world_coord.ndim != 0
        nearest_index_cur, _ = find_nearest_point(self.robot_pos[:2], self._reference_path_world_coord)
        nearest_index_last, _ = find_nearest_point(self.last_robot_pos[:2], self._reference_path_world_coord)
        # print("last_idx: {} {} pos: {}\ncur_idx : {} {} pos: {}\n".format(
        #     nearest_index_last, self._reference_path_world_coord[nearest_index_last], self.last_robot_pos,
        #     nearest_index_cur, self._reference_path_world_coord[nearest_index_cur], self.robot_pos))
        return nearest_index_cur - nearest_index_last

    def _compute_reference_distance_error(self):
        """
        Compute nearest distance between the agent and reference paths.
        The agent's path is divided into some points whose number is
        defined by `self._divide_num_own`

        :return (float): Maximum or mean distance to reference path
        """
        agent_poses = np.vstack((self.last_robot_pos[:2], self.robot_pos[:2]))
        agent_poses = divide_path(agent_poses, self._n_own_path_division)

        distances = []
        # trajectory_points is {# of points} x {# of dimensions} array
        for agent_pos in agent_poses:
            _, min_dist = find_nearest_point(agent_pos, self._reference_path_world_coord)
            distances.append(min_dist)
        return max(distances)

    def _get_way_points_world_coord(self):
        goselo_img = self._env.get_goselo_img()
        abs_way_points_goselo_coord = self._way_points_generator.get_action(goselo_img, test=True, policy="all")
        abs_way_points_world_coord = decode_way_points_to_world_coord(
            current_pos=self._env.robot_pos[:2],
            goal_pos=self._env.goal_pos[:2],
            abs_vertices=abs_way_points_goselo_coord.reshape(-1, 2),
        )
        abs_way_points_world_coord = abs_way_points_world_coord.reshape(-1, 2)
        abs_way_points_world_coord += self._env.robot_pos[:2]
        return abs_way_points_world_coord

    def _update_reference_path(self):
        abs_way_points_world_coord = self._get_way_points_world_coord()
        self.set_reference_path_world_coord(
            abs_way_points_world_coord, max_divide_distance=0.005, default_min_divide_distance=0.0025
        )

    def _is_update_reference(self):
        min_idx, min_dist = find_nearest_point(self.robot_pos[:2], self._reference_path_world_coord)
        if min_idx > int(self._reference_path_world_coord.shape[0] / 2.0):
            return True
        if min_dist > self._threshold_min_dist_to_reference_path:
            return True
        return False


class RLEnvWayPointsGeneratorEvalWrapper(RLEnvWayPointsGeneratorWrapper):
    def _compute_reward(self):
        return self._baseline_reward()


class VisOnlyInputEnvWrapper:
    def __init__(self, env):
        self._env = env

    def render(self):
        if self._env._input_raw_way_points:
            for i in range(self._env._n_way_points_for_vis):
                if i < self._env._n_way_points:
                    self._env._set_pos("way_point_{}".format(i), self._env._raw_reference_path_world_coord[i])
                else:
                    self._env._set_pos("way_point_{}".format(i), np.zeros(shape=(2,), dtype=np.float32))
        else:
            nearest_idx, _ = find_nearest_point(self._env.robot_pos[:2], self._env._reference_path_world_coord)
            n_input_way_points = min(
                self._env._n_way_points, self._env._reference_path_world_coord.shape[0] - (nearest_idx + 1)
            )
            for i in range(self._env._n_way_points_for_vis):
                if i < n_input_way_points:
                    self._env._set_pos("way_point_{}".format(i), self._env._reference_path_world_coord[nearest_idx + i])
                else:
                    self._env._set_pos("way_point_{}".format(i), np.zeros(shape=(2,), dtype=np.float32))
        self._env.update_layout()
        self._env.render()

    def __getattr__(self, attr):
        # Enable to call functions defined in `self._env`
        return self._env.__getattribute__(attr)
