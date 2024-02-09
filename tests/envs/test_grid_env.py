# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import unittest

import numpy as np

from safety_rl.envs.engine_wrapper import EngineWrapper
from safety_rl.envs.grid_env import GridEnv


class TestGridEnv(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = {
            "robot_base": "xmls/point.xml",
            "task": "goal",
            "hazards_num": 10,
            "placements_extents": [-1, -1, 1, 1],
        }
        cls.out_img_size = (32, 32)
        cls.env = GridEnv(config, out_img_size=cls.out_img_size)

    def test_init(self):
        np.testing.assert_array_equal(self.env.observation_space.shape, self.env.out_img_size)

    def test_reset(self):
        obs = self.env.reset()
        self.assertEqual(self.out_img_size, self.env.out_img_size)
        self.assertEqual(obs.shape, self.env.out_img_size)

    def test_step(self):
        self.env.reset()
        next_obs, *_ = self.env.step(self.env.action_space.sample())
        self.assertEqual(next_obs.shape, self.env.out_img_size)

    def test_to_img_pos(self):
        tuples = []
        # Top left of the image
        tuples.append([np.array([-1.0, 1.0]), np.array([0, 0], dtype=np.int32)])
        # Center of the image
        tuples.append([np.array([0.0, 0.0]), (self.env._field_img_size - 1) / 2])
        # Bottom right of the image
        tuples.append([np.array([1.0, -1.0]), self.env._field_img_size - 1])
        for data in tuples:
            np.testing.assert_array_equal(self.env.to_img_pos(data[0]), data[1])


if __name__ == "__main__":
    unittest.main()
