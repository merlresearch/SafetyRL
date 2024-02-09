# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import unittest

import numpy as np

from safety_rl.path_planner.a_star import AStarNode, AStarObsMap


class TestAStar(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """
        Construct a simple map for test as:
        ---
        |x|
        | |
        |s|
        ---
        , where s (0,0) is start and (0,2) is goal.
        ---
        :return:
        """
        cls.obstacles = np.array(
            [[-1, -1], [0, -1], [1, -1], [-1, 0], [1, 0], [-1, 1], [1, 1], [-1, 2], [1, 2], [-1, 3], [0, 3], [1, 3]]
        )
        cls.planner = AStarObsMap(obstacles=cls.obstacles, resolution=1.0)
        cls._start = np.array([0, 0])
        cls._goal = np.array([0, 2])

    def test_is_collided(self):
        planner = AStarObsMap(obstacles=self.obstacles, resolution=1.0)
        expected_falses = [AStarNode(*pos, 0, 0) for pos in [[0, 0], [0, 1], [0, 2]]]
        expected_trues = [AStarNode(*pos, 0, 0) for pos in self.obstacles]

        for node in expected_falses:
            self.assertFalse(planner._is_collided(node))

        for node in expected_trues:
            self.assertTrue(planner._is_collided(node))

    def test_calc_index(self):
        def test(planner, inputs, expected_indices):
            for input, expected_idx in zip(inputs, expected_indices):
                node = AStarNode(x=input[0], y=input[1], cost=0.0, parent=0)
                idx = planner._calc_index(node)
                self.assertEqual(expected_idx, idx)

        planner = AStarObsMap(obstacles=self.obstacles, resolution=1.0)
        inputs = np.array(
            [
                [-1, -1],
                [0, -1],
                [1, -1],
                [-1, 0],
                [0, 0],
                [1, 0],
                [-1, 1],
                [0, 1],
                [1, 1],
                [-1, 2],
                [0, 2],
                [1, 2],
                [-1, 3],
                [0, 3],
                [1, 3],
            ],
            dtype=np.float32,
        )
        expected_indices = np.arange(inputs.shape[0])
        test(planner, inputs, expected_indices)

        planner = AStarObsMap(obstacles=self.obstacles, resolution=0.5)
        expected_indices = np.array([0, 2, 4, 10, 12, 14, 20, 22, 24, 30, 32, 34, 40, 42, 44])
        test(planner, inputs, expected_indices)

        planner = AStarObsMap(obstacles=self.obstacles, resolution=2.0)
        inputs = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1], [-1, 3], [1, 3]], dtype=np.float32)
        expected_indices = np.arange(inputs.shape[0])
        test(planner, inputs, expected_indices)

    def test_planning(self):
        """
        Test A* planning over different resolutions
        :return:
        """

        def test_over_different_reso(reso, expected, obstacles=None):
            obstacles = self.obstacles if obstacles is None else obstacles
            planner = AStarObsMap(obstacles=obstacles, resolution=reso)
            results = planner.planning(start=self._start, goal=self._goal)
            np.testing.assert_array_equal(results, expected)

        test_over_different_reso(
            reso=0.5, expected=np.array([[0, 0], [0, 0.5], [0, 1.0], [0, 1.5], [0, 2]], dtype=np.float32)
        )
        test_over_different_reso(reso=1.0, expected=np.array([[0, 0], [0, 1], [0, 2]]))
        test_over_different_reso(
            reso=2.0,
            expected=np.array([[0, 0], [0, 2]], dtype=np.float32),
            obstacles=np.array([[-2.0, -2.0], [2.0, -2.0], [-2.0, 4.0], [2.0, 4.0]]),
        )

    def test_is_devisible(self):
        from safety_rl.path_planner.a_star import is_devisible

        self.assertTrue(is_devisible(1.0, 0.5))
        self.assertFalse(is_devisible(1.0, 0.3))

    def test_is_devisible_array(self):
        from safety_rl.path_planner.a_star import is_devisible_2darray

        self.assertTrue(is_devisible_2darray(np.array([1.0, 1.0]), np.array([0.5, 0.25])))
        self.assertFalse(is_devisible_2darray(np.array([1.0, 1.0]), np.array([0.4, 0.5])))


if __name__ == "__main__":
    unittest.main()
