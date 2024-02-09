# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import cv2
import gym
import numpy as np

from safety_rl.envs.engine_wrapper import EngineWrapper


class GridEnv(EngineWrapper):
    def __init__(self, *args, out_img_size=(64, 64), astar_reso=0.1, img_reso=0.5, **kwargs):
        super().__init__(*args, **kwargs)

        minx, miny, maxx, maxy = self.placements_extents
        self._minx = minx
        self._miny = miny
        # field image size
        self._field_img_size = np.array(
            [maxx * 2 / astar_reso / img_reso + 1, maxy * 2 / astar_reso / img_reso + 1], dtype=np.int32
        )
        # members to get grid image
        self._expansion_rate = int(1.0 / astar_reso / img_reso)
        self._gremlin_radius = int(self.gremlins_size * self._expansion_rate * np.sqrt(2))  # Rectangle to circle
        self._hazard_radius = int(self.hazards_size * self._expansion_rate)
        self._pillar_radius = int(self.pillars_size * self._expansion_rate)
        self._robot_radius = int(self.robot_keepout * self._expansion_rate)
        self._goal_radius = int(self.goal_size * self._expansion_rate)
        self._way_point_radius = int(self._goal_radius / 2)
        self._output_img_size = out_img_size
        self._goal_plot_size = int(0.5 * self._expansion_rate)
        self._img_reso = img_reso

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=out_img_size, dtype=np.uint8)

    def _initialize_map(self):
        np_img = np.zeros(shape=self._field_img_size, dtype=np.float32)
        field_img = cv2.cvtColor(np_img, cv2.COLOR_GRAY2BGR)
        # Draw double circle to goal location
        goal_pos = self.to_img_pos(self.goal_pos)
        cv2.circle(
            img=field_img,
            center=(goal_pos[0], goal_pos[1]),
            radius=self._goal_radius,
            color=(255, 255, 255),
            thickness=-1,
        )
        cv2.circle(
            img=field_img,
            center=(goal_pos[0], goal_pos[1]),
            radius=int(self._goal_radius / 3.0),
            color=(0, 0, 0),
            thickness=-1,
        )
        # Draw circle to hazards location
        for hazard_pos in self.hazards_pos:
            img_pos = self.to_img_pos(hazard_pos)
            cv2.circle(
                img=field_img,
                center=(img_pos[0], img_pos[1]),
                radius=self._hazard_radius,
                color=(255, 255, 255),
                thickness=-1,
            )
        # Draw circle to pillars location
        for pillar_pos in self.pillars_pos:
            img_pos = self.to_img_pos(pillar_pos)
            cv2.circle(
                img=field_img,
                center=(img_pos[0], img_pos[1]),
                radius=self._pillar_radius,
                color=(255, 255, 255),
                thickness=-1,
            )
        # Draw circle to gremlins location
        for gremlin_obj_pos in self.gremlins_obj_pos:
            img_pos = self.to_img_pos(gremlin_obj_pos)
            cv2.circle(
                img=field_img,
                center=(img_pos[0], img_pos[1]),
                radius=self._gremlin_radius,
                color=(255, 255, 255),
                thickness=-1,
            )
        if self.place_room:
            room_wall_thickness = int(
                np_img.shape[0] * self.room_wall_thickness / max(self.placements_extents) / self._img_reso
            )
            if self.room_type == 0:
                center_wall_img_size = int(np_img.shape[0] * self.center_wall_size / 2.0 / max(self.placements_extents))
                start_point = (0, int(np_img.shape[1] / 2.0 - room_wall_thickness / 2.0))
                end_point = (center_wall_img_size, int(np_img.shape[1] / 2.0 + room_wall_thickness / 2.0))
                field_img = cv2.rectangle(field_img, start_point, end_point, (255, 255, 255), -1)
            elif self.room_type == 1:
                center_wall_img_size = int(np_img.shape[0] * self.center_wall_size / max(self.placements_extents))
                corner_wall_img_size = int(np_img.shape[0] * self.corner_wall_size / max(self.placements_extents))
                # Horizontal walls
                start_point = (
                    int(np_img.shape[0] / 2.0 - center_wall_img_size / 2.0),
                    int(np_img.shape[1] / 2.0 - room_wall_thickness / 2.0),
                )
                end_point = (
                    int(np_img.shape[0] / 2.0 + center_wall_img_size / 2.0),
                    int(np_img.shape[1] / 2.0 + room_wall_thickness / 2.0),
                )
                field_img = cv2.rectangle(field_img, start_point, end_point, (255, 255, 255), -1)

                start_point = (0, start_point[1])
                end_point = (corner_wall_img_size, end_point[1])
                field_img = cv2.rectangle(field_img, start_point, end_point, (255, 255, 255), -1)

                start_point = (np_img.shape[0] - corner_wall_img_size, start_point[1])
                end_point = (np_img.shape[0], end_point[1])
                field_img = cv2.rectangle(field_img, start_point, end_point, (255, 255, 255), -1)

                # Vertical walls
                start_point = (
                    int(np_img.shape[0] / 2.0 - room_wall_thickness / 2.0),
                    int(np_img.shape[1] / 2.0 - center_wall_img_size / 2.0),
                )
                end_point = (
                    int(np_img.shape[0] / 2.0 + room_wall_thickness / 2.0),
                    int(np_img.shape[1] / 2.0 + center_wall_img_size / 2.0),
                )
                field_img = cv2.rectangle(field_img, start_point, end_point, (255, 255, 255), -1)

                start_point = (start_point[0], 0)
                end_point = (end_point[0], corner_wall_img_size)
                field_img = cv2.rectangle(field_img, start_point, end_point, (255, 255, 255), -1)

                start_point = (start_point[0], np_img.shape[1] - corner_wall_img_size)
                end_point = (end_point[0], np_img.shape[1])
                field_img = cv2.rectangle(field_img, start_point, end_point, (255, 255, 255), -1)
            else:
                raise NotImplementedError

        self._field_img = field_img

    def _get_obs(self):
        out_img = self._field_img.copy()
        robot_pos = self.to_img_pos(self.robot_pos)
        cv2.circle(
            img=out_img, center=(robot_pos[0], robot_pos[1]), radius=self._robot_radius, color=(0, 255, 0), thickness=-1
        )
        out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2GRAY)
        return cv2.resize(src=out_img, dsize=self._output_img_size)

    def reset(self):
        # Repeat reset until the discretized robot and goal location is not same
        for trial in range(100):
            self.reset_sim()
            discrete_robot_pos = self.to_img_pos(self.robot_pos)
            discrete_goal_pos = self.to_img_pos(self.goal_pos)
            if not np.array_equal(discrete_robot_pos, discrete_goal_pos):
                break
        self._initialize_map()
        return self._get_obs()

    def step(self, action):
        _, rew, done, env_info = super().step(action)
        return self._get_obs(), rew, done, env_info

    def to_img_pos(self, pos):
        """
        Note that Y-axis will be flipped in image coordinate
        :param pos: (X,Y) location on MuJoCo env
        :return: (X,Y) location on image (pixel values)
        """
        return np.array(
            [
                self._expansion_rate * (pos[0] - self._minx),
                (self._field_img_size[1] - 1) - self._expansion_rate * (pos[1] - self._miny),
            ],
            dtype=np.int32,
        )

    def _prepare_field_img_for_render(self):
        field_img = self._field_img.copy()
        robot_pos = self.to_img_pos(self.robot_pos)
        cv2.circle(
            img=field_img,
            center=(robot_pos[0], robot_pos[1]),
            radius=self._robot_radius,
            color=(0, 255, 0),
            thickness=-1,
        )
        goal_pos = self.to_img_pos(self.goal_pos)
        cv2.circle(
            img=field_img, center=(goal_pos[0], goal_pos[1]), radius=self._goal_radius, color=(255, 0, 0), thickness=-1
        )
        for way_point in self._reference_path_world_coord:
            target_pos = self.to_img_pos(way_point)
            if abs(target_pos[0]) >= self._field_img.shape[0] or abs(target_pos[1]) >= self._field_img.shape[1]:
                continue
            target_pos += self._field_img.shape[:2]
            cv2.circle(
                img=field_img,
                center=(target_pos[0], target_pos[1]),
                radius=self._way_point_radius,
                color=(0, 0, 255),
                thickness=-1,
            )
        return field_img

    def save_current_pos_to_pathlog(self):
        xy = self.to_img_pos(self.robot_pos)
        self._pathlog_map[xy[1]][xy[0]] += 1

    def render(self, mode=None):
        field_img = self._prepare_field_img_for_render()
        cv2.imshow("GridEnv/field_img", field_img)
        cv2.waitKey(1)
        if mode == "rgb_array":
            return field_img
        super().render()

    @property
    def map_size(self):
        return self._field_img_size

    @property
    def out_img_size(self):
        return self._output_img_size


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hazards-num", type=int, default=10)
    parser.add_argument("--field-size", type=int, default=2)
    parser.add_argument("--resolution", type=float, default=0.1)
    args = parser.parse_args()

    config = {
        "robot_base": "xmls/point.xml",
        "task": "goal",
        "hazards_num": args.hazards_num,
        "placements_extents": [-1 * args.field_size, -1 * args.field_size, args.field_size, args.field_size],
    }
    env = GridEnv(config)
    while True:
        env.reset()
        for _ in range(300):
            _, _, done, _ = env.step(env.action_space.sample())
            env.render("debug")
            if done:
                break
