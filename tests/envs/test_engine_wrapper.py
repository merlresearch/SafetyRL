# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import unittest

import numpy as np

from safety_rl.envs.engine_wrapper import EngineWrapper


class TestEnvWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = {
            "robot_base": "xmls/car.xml",
            "task": "push",
            "hazards_num": 3,
        }
        cls.test_env = EngineWrapper(config)
        cls.test_envs = [cls.test_env]
        config["robot_base"] = "xmls/point.xml"
        cls.test_envs.append(EngineWrapper(config))
        config["robot_base"] = "xmls/doggo.xml"
        cls.test_envs.append(EngineWrapper(config))
        for env in cls.test_envs:
            env.reset()

    def test_set_hazards_pos(self):
        expected_hazards_pos = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        self.test_env.set_hazards_pos(expected_hazards_pos)
        np.testing.assert_equal(np.array(self.test_env.hazards_pos)[:, :2], expected_hazards_pos)

    def test_set_robot_pos(self):
        for test_env in self.test_envs:
            expected_pos = np.zeros(shape=(2,), dtype=np.float32)
            test_env.set_robot_pos(expected_pos)
            np.testing.assert_equal(np.array(test_env.robot_pos)[:2], expected_pos)

    def test_set_goal_pos(self):
        expected_pos = np.zeros(shape=(2,), dtype=np.float32)
        self.test_env.set_goal_pos(expected_pos)
        np.testing.assert_equal(np.array(self.test_env.goal_pos)[:2], expected_pos)


if __name__ == "__main__":
    unittest.main()
