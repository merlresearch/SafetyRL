# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import unittest

import numpy as np

from safety_rl.experiments.utils import to_abs_way_points, to_rel_way_points


class TestGenerateDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.abs_way_points = np.array([[1, 0], [1, 1], [2, 2], [3, 1], [3, 1]], dtype=np.float32)
        cls.rel_way_points = np.array([[1, 0], [0, 1], [1, 1], [1, -1], [0, 0]], dtype=np.float32)

    def test_to_abs_way_points(self):
        abs_way_points = to_abs_way_points(self.rel_way_points)
        np.array_equal(abs_way_points, self.abs_way_points)

    def test_to_rel_way_points(self):
        rel_way_points = to_rel_way_points(self.abs_way_points)
        np.array_equal(rel_way_points, self.rel_way_points)


if __name__ == "__main__":
    unittest.main()
