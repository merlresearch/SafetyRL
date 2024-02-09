# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


class AbstractPathPlanner:
    def __init__(self, start, goal):
        if start.shape[0] != goal.shape[0]:
            raise ValueError("Dimension mismatch")
        self.start_node = Node(start)
        self.goal_node = Node(goal)
        self.dim = start.shape[0]


class Node:
    def __init__(self, pos, cost=0.0, parent=None):
        """
        :param pos (np.ndarray): n-dimensional array
        :cost (float): Cost to connect path
        :parent (int): Index of parent node
        """
        self.pos = pos
        self.cost = cost
        self.parent = parent
