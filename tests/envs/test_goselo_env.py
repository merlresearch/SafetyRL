# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import unittest

import numpy as np

from safety_rl.envs.engine_wrapper import EngineWrapper
from safety_rl.envs.goselo_env import GoseloEnv


class TestGoseloEnv(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.hazards_num = 10
        config = {
            "robot_base": "xmls/point.xml",
            "task": "goal",
            "hazards_num": cls.hazards_num,
            "placements_extents": [-1, -1, 1, 1],
        }
        cls.out_img_size = (32, 32)
        cls.goselo_img_size = cls.out_img_size + (6,)
        cls.env = GoseloEnv(config, out_img_size=cls.out_img_size)

    def test_init(self):
        np.testing.assert_array_equal(self.env.observation_space.shape, self.env.out_img_size)

    def test_reset(self):
        obs = self.env.reset()
        self.assertEqual(self.goselo_img_size, self.env.out_img_size)
        self.assertEqual(obs.shape, self.env.out_img_size)

    def test_step(self):
        self.env.reset()
        next_obs, *_ = self.env.step(self.env.action_space.sample())
        self.assertEqual(next_obs.shape, self.env.out_img_size)

    def test_get_obs(self):
        """
        --------
        |      |
        xxxxxxxx
        |      |
        --------
        """
        hazards_pos = np.linspace([-1, 0], [1, 0], self.hazards_num)
        self.env.set_hazards_pos(hazards_pos)
        self.env.set_robot_pos(np.array([0.0, -1.0]))
        self.env.set_goal_pos(np.array([0.0, 1.0]))
        self.env._initialize_map()


if __name__ == "__main__":
    unittest.main()
