# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import unittest

import numpy as np

from safety_rl.envs.engine_wrapper import EngineWrapper
from safety_rl.path_planner.a_star import AStarNode
from safety_rl.path_planner.a_star_safety_gym import AStarSafetyGym


class TestAStarSafetyGym(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = {"robot_base": "xmls/car.xml", "task": "goal", "hazards_num": 1}
        cls.planner = AStarSafetyGym(env=EngineWrapper(cls.config))

    def test_is_collided(self):
        n_test_point = 5
        keepout_radius = self.planner.env.hazards_size + self.planner.env.robot_keepout
        keepout_radiuses = np.linspace([0, 0, 0], [keepout_radius, 0, 0], n_test_point)
        expected_trues = keepout_radiuses + self.planner.env.hazards_pos[0]

        for expected_true in expected_trues:
            x, y = expected_true[:2]
            node = AStarNode(x, y, cost=0, parent=0)
            self.assertTrue(self.planner._is_collided(node))

        expected_falses = np.copy(expected_trues)
        expected_falses[:, 0] += self.planner.env.hazards_keepout + self.planner.env.robot_keepout + 1e-6

        for expected_false in expected_falses:
            x, y = expected_false[:2]
            node = AStarNode(x, y, cost=0, parent=0)
            self.assertFalse(self.planner._is_collided(node))

    def test_is_violate_area(self):
        """
        Specify allowed area to a unit square
        ------
        |    |
        |    |
        ------
        :return:
        """
        config = self.config
        config["placements_extents"] = [0, 0, 1, 1]
        config["hazards_num"] = 1

        planner = AStarSafetyGym(env=EngineWrapper(config), resolution=1.0)
        allowed_pos = np.random.uniform(0, 1, 20).reshape(-1, 2)
        not_allowed_pos = np.array([[-1, 0.0], [1.1, 0.0], [1.1, 1.1]], dtype=np.float32)
        for pos in allowed_pos:
            node = AStarNode(x=pos[0], y=pos[1], parent=0, cost=0.0)
            self.assertFalse(planner._is_violate_area(node))
        for pos in not_allowed_pos:
            node = AStarNode(x=pos[0], y=pos[1], parent=0, cost=0.0)
            self.assertTrue(planner._is_violate_area(node))

    def test_planning(self):
        """
        An agent starts at S(0., 0.) and goal G(2., 0.) while
        avoiding collision with obstacle(1.,0.)
        ---------
        |       |
        | S x G |
        ---------
        :return:
        """
        config = self.config
        config["placements_extents"] = [0, 0, 2, 1]
        config["hazards_num"] = 1

        def test_over_different_reso(reso, expected):
            planner = AStarSafetyGym(env=EngineWrapper(config), resolution=reso)
            # Change hazard location
            planner.env.set_hazards_pos(np.array([[1.0, 0.0]]))
            results = planner.planning(start=np.array([0.0, 0.0]), goal=np.array([2.0, 0.0]))
            np.testing.assert_array_equal(results, expected)

        expected = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]], dtype=np.float32)
        test_over_different_reso(reso=1.0, expected=expected)

        # Note that hazard and robot_radius keepout is 0.4 in diameter
        expected = np.array(
            [[0.0, 0.0], [0.0, 0.5], [0.5, 1.0], [1.0, 1.0], [1.5, 1.0], [2.0, 0.5], [2.0, 0.0]], dtype=np.float32
        )
        test_over_different_reso(reso=0.5, expected=expected)


if __name__ == "__main__":
    unittest.main()
