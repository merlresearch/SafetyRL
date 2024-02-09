# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


import math

import cv2
import gym
import numpy as np
import scipy
import scipy.ndimage

from safety_rl.envs.grid_env import GridEnv


class GoseloEnv(GridEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._output_img_size = self.out_img_size + (6,)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=self.out_img_size, dtype=np.uint8)

    def _initialize_map(self):
        super()._initialize_map()
        # Binarize field map and convert it to gray scale image
        binarized_field_map = cv2.threshold(self._field_img, 100, 1, cv2.THRESH_BINARY_INV)[1]
        if self._field_img.ndim == 3:
            binarized_field_map = binarized_field_map[:, :, 0]
        self._binarized_field_map = binarized_field_map
        self._pathlog_map = np.zeros_like(self._field_img[:, :, 0])
        discrete_robot_pos = self.to_img_pos(self.robot_pos)
        self._pathlog_map[discrete_robot_pos[1]][discrete_robot_pos[0]] += 1

    def on_episode_end(self):
        """
        Save explored image. This function should be called after episode termination and before `reset`
        """
        x = self._start[0]
        y = self._start[1]
        field_map_vis = self._field_map.copy()
        cv2.circle(field_map_vis, (x, y), 2, (0, 0, 255), -1)
        cv2.imwrite("result.png", field_map_vis)

    def step(self, action):
        """
        Apply one step forward
        """
        _, rew, done, info = super().step(action)

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
            info["exceed_limit"] = True
            done = True
        else:
            info["exceed_limit"] = False
            self._pathlog_map[discrete_robot_pos[1]][discrete_robot_pos[0]] += 1

        if len(self.gremlins_obj_pos) > 0 and not info["exceed_limit"]:
            self._initialize_map()

        return self._get_obs(), rew, done, info

    def _get_obs(self):
        rotated_img = self._rotate_img()
        self._clipped_img = self._clip_img(rotated_img).copy()
        return self._clipped_img

    def _rotate_img(self):
        """
        Generate rotated image

        return: Rotated image that consists of field and pathlog image
        """
        binarized_field_map = np.copy(self._binarized_field_map)
        # Prepare original image
        orig_map = np.zeros(
            shape=(2, self._field_img.shape[0], self._field_img.shape[1]), dtype=np.float
        )  # (2, height, width)
        orig_map[0] = np.array(binarized_field_map)
        orig_map[1] = np.array(self._pathlog_map)
        orig_map = orig_map.transpose((1, 2, 0))  # (height, width, 2)

        # Prepare a little bit larger image buffer so that it can be rotated
        discrete_robot_pos = self.to_img_pos(self.robot_pos)
        discrete_goal_pos = self.to_img_pos(self.goal_pos)
        center_x, center_y = (discrete_robot_pos + discrete_goal_pos) / 2.0
        new_height = max(center_y, orig_map.shape[0] - center_y)
        new_width = max(center_x, orig_map.shape[1] - center_x)
        try:
            rotated_img = np.zeros(
                shape=(int(new_height * 2), int(new_width * 2), orig_map.shape[2]), dtype=np.float
            )  # (>height, >width, 2)
        except MemoryError:
            print(center_y, orig_map.shape[0], center_x, center_x, self.robot_pos, self.goal_pos)

        # Insert original map to buffer so that the center position between
        # agent and goal should be the center position of newly prepared image
        insert_pos_lower_left = (int(new_height - center_y), int(new_width - center_x))
        rotated_img[
            insert_pos_lower_left[0] : insert_pos_lower_left[0] + orig_map.shape[0],
            insert_pos_lower_left[1] : insert_pos_lower_left[1] + orig_map.shape[1],
        ] = orig_map

        # Compute relative angle between agent and goal in polar coordinate
        goal_angle = self._compute_goal_angle_in_polar_coord()

        # Multiply by two for linear interpolation
        rotated_img[rotated_img == 1] = 2

        # Rotate image so that goal always exists upper position of agent
        # Why need `90`? It is because goal angle is computed on polar coordinate.
        # Suppose an agent's position is (0, 0), then theta can be computed as:
        #   theta = 0:   (x_g, y_g) = (1, 0)
        #   theta = 90:  (x_g, y_g) = (0, 1)
        #   theta = 180: (x_g, y_g) = (-1, 0)
        #   theta = 270: (x_g, y_g) = (0, -1)
        # You might think we should subtract 90 instead of add, but it is not true.
        # Be careful that Y-axis is upside down in image coordinate.
        rotated_img = scipy.ndimage.interpolation.rotate(rotated_img, goal_angle + 90)

        # Convert from (height, width, channel) to (channel, height, width)
        return rotated_img.transpose((2, 0, 1))

    def _compute_goal_angle_in_polar_coord(self):
        """
        Compute angle between agent position and goal in **polar coordinate**.
        Suppose an agent locates in (x, y) = (0, 0), then the rotation angle
        can be computed against goal position (x_g, y_g) as:
            theta = 0:   (x_g, y_g) = (1, 0)
            theta = 90:  (x_g, y_g) = (0, 1)
            theta = 180: (x_g, y_g) = (-1, 0)
            theta = 270: (x_g, y_g) = (0, -1)

        return (float): Rotation angle in degree
        """
        discrete_robot_pos = self.to_img_pos(self.robot_pos)
        discrete_goal_pos = self.to_img_pos(self.goal_pos)
        return compute_deg_in_polar_coord(origin=discrete_robot_pos, query=discrete_goal_pos)

    def _clip_img(self, rotated_img):
        """
        Clip rotated global map and trajectories agent moved with three scales

        :param rotated_img (np.ndarray):
            (channel, height, width) shape array that contains
            binalized image in first channel and path logs in second channel
        """
        clipped_img = np.zeros((6, self.out_img_size[0], self.out_img_size[1]), dtype=np.uint8)  # (6, height, width)
        L = int(np.linalg.norm(self.to_img_pos(self.robot_pos[:2]) - self.to_img_pos(self.goal_pos[:2])))

        # Iterate over two channel: rotation of original image, and path log
        for channel in range(2):
            # Iterate over three different scales
            # for i, scale in enumerate((L+4, L*4, L*8)):
            for i, scale in enumerate((L + 4, L * 2, L * 4)):
                # Prepare buffer for scaled image
                scaled_rotate_img = np.zeros((scale, scale), dtype=np.uint8)

                # Compute the position of image that will be inserted to scaled image
                y_src = int(max(0, (rotated_img[channel].shape[0] - scale) / 2))
                x_src = int(max(0, (rotated_img[channel].shape[1] - scale) / 2))

                # Compute the position of image that specifies the inserted position
                y_dst = int(max(0, -(rotated_img[channel].shape[0] - scale) / 2))
                x_dst = int(max(0, -(rotated_img[channel].shape[1] - scale) / 2))

                # Compute the size of image to be inserted to scaled image
                height = min(scale, rotated_img[channel].shape[0])
                width = min(scale, rotated_img[channel].shape[1])

                # Insert original image to scaled image
                scaled_rotate_img[y_dst : y_dst + height, x_dst : x_dst + width] = rotated_img[channel][
                    y_src : y_src + height, x_src : x_src + width
                ]

                # Resize scaled image to output size
                clipped_img[i + channel * 3] = cv2.resize(
                    scaled_rotate_img, dsize=clipped_img[i + channel * 3].shape, interpolation=cv2.INTER_AREA
                )

        clipped_img[clipped_img > 0] = 1

        # Transpose from (channel, height, width) to (height, width, channel)
        return clipped_img.transpose((1, 2, 0)) * 255

    def render(self, mode=None):
        field_img = super().render(mode)
        temp = np.ones(shape=(self._clipped_img.shape[1], 3)) * 255
        goselo_img = np.concatenate(
            (
                self._clipped_img[:, :, 0],
                temp,
                self._clipped_img[:, :, 1],
                temp,
                self._clipped_img[:, :, 2],
                temp,
                self._clipped_img[:, :, 3],
                self._clipped_img[:, :, 4],
                self._clipped_img[:, :, 5],
            ),
            axis=1,
        )
        cv2.imshow("GoseloEnv/GoseloImg", goselo_img)
        cv2.waitKey(1)
        if mode == "rgb_array":
            return field_img


def compute_deg_in_polar_coord(origin, query):
    assert isinstance(origin, np.ndarray) and isinstance(query, np.ndarray)
    assert origin.ndim == query.ndim == 1
    assert origin.shape[0] == query.shape[0] == 2
    # x position of origin and query is the same
    if query[0] == origin[0]:
        if query[1] > origin[1]:
            return 90
        else:
            return 270

    theta = math.atan((float(query[1]) - float(origin[1])) / (float(query[0]) - float(origin[0]))) * 180 / math.pi

    # If the origin is left side to the query, then plus 180 deg
    # because the output range of `math.atan` is [-90, 90]
    if float(query[0]) - float(origin[0]) < 0.0:
        theta += 180

    while theta > 360:
        theta -= 360

    return theta


if __name__ == "__main__":
    print("Please try GoseloEnv by `python examples/run_goselo_randomly.py`")
