# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import decimal
import logging
import math
import sys

import matplotlib.patches as pat
import matplotlib.pyplot as plt
import numpy as np

from safety_rl.path_planner.planner_base import AbstractPathPlanner, Node
from safety_rl.path_planner.utils import roundint


def is_devisible(a, b):
    return decimal.Decimal(float(a)) % decimal.Decimal(float(b)) == decimal.Decimal("0.00")


def is_devisible_2darray(array_a, array_b):
    assert isinstance(array_a, np.ndarray) and isinstance(array_b, np.ndarray)
    assert array_a.shape == array_b.shape == (2,)
    return is_devisible(array_a[0], array_b[0]) & is_devisible(array_a[1], array_b[1])


class AStarNode(Node):
    def __init__(self, x, y, cost, parent):
        super().__init__(pos=np.array([x, y]), cost=cost, parent=parent)
        self.x = x
        self.y = y


class AStarBase(AbstractPathPlanner):
    def __init__(self, minx, maxx, miny, maxy, resolution=1.0, show_process=False):
        assert minx < maxx and miny < maxy
        self._reso = resolution
        self._show_process = show_process
        self._minx = minx
        self._maxx = maxx
        self._miny = miny
        self._maxy = maxy
        self._nx = roundint((self._maxx - self._minx) / self._reso) + 1
        self._ny = roundint((self._maxy - self._miny) / self._reso) + 1
        self._logger = logging.getLogger("safety_rl")

    def _init_plt(self):
        plt.close()
        if self._show_process:
            self._fig, self._ax = plt.subplots()
            self._ax.set_aspect(1.0)
            self._ax.set_xlim(self._minx, self._maxx)
            self._ax.set_ylim(self._miny, self._maxy)

    def _init_setting(self, init_start, init_goal):
        if not ((init_start is not None) or (init_goal is not None)):
            raise NotImplementedError
        else:
            start = init_start.copy()
            goal = init_goal.copy()
        return start, goal

    def planning(self, init_start=None, init_goal=None, n_trial=100):
        motion = self._get_motion_model()
        specify_start_goal = (init_start is not None) and (init_goal is not None)

        for cur_trial in range(n_trial):
            start, goal = self._init_setting(init_start, init_goal)

            assert isinstance(start, np.ndarray) and isinstance(goal, np.ndarray)
            assert start.shape[0] == 2 and goal.shape[0] == 2

            start_node = AStarNode(start[0], start[1], 0.0, -1)
            goal_node = AStarNode(goal[0], goal[1], 0.0, -1)
            open_set, closed_set = dict(), dict()
            open_set[self._calc_index(start_node)] = start_node

            if self._show_process:
                self._init_plt()
                self._ax.add_patch(pat.Circle(xy=start_node.pos, radius=0.25, color="g"))
                self._ax.add_patch(pat.Circle(xy=goal_node.pos, radius=0.25, color="r"))

            failed = False

            while not failed:
                # Find the node with the least f (the movement cost + the estimated cost) on the open list
                try:
                    current_id = min(
                        open_set, key=lambda o: open_set[o].cost + self._estimate_cost(goal_node, open_set[o])
                    )
                except ValueError as e:
                    self._logger.debug(
                        "Failed to generate path at trial={} with error msg '{}', start={} goal={}".format(
                            cur_trial, e.with_traceback(sys.exc_info()[2]), start_node.pos, goal_node.pos
                        )
                    )
                    failed = True
                    continue
                current_node = open_set[current_id]

                # Show process
                if self._show_process:
                    self.render(current_node)

                if self._is_goal(current_node, goal_node):
                    goal_node.parent = current_node.parent
                    goal_node.cost = current_node.cost
                    break

                # Remove the item from the open set and add it to the closed set
                del open_set[current_id]
                closed_set[current_id] = current_node

                # Expand search grid based on motion model
                for i, _ in enumerate(motion):
                    node = AStarNode(
                        x=current_node.x + motion[i][0] * self._reso,
                        y=current_node.y + motion[i][1] * self._reso,
                        cost=current_node.cost + motion[i][2] * self._reso,
                        parent=current_id,
                    )
                    n_id = self._calc_index(node)
                    if n_id in closed_set:
                        continue

                    if not self._verify_node(node):
                        continue

                    if n_id not in open_set:
                        open_set[n_id] = node  # Discover a new node
                    else:
                        if open_set[n_id].cost >= node.cost:
                            open_set[n_id] = node  # Record the best path

            if not failed:
                return self._calc_final_path(closed_set, goal_node)
            elif specify_start_goal:
                return None

    def _is_goal(self, current_node, goal_node):
        return np.linalg.norm(current_node.pos - goal_node.pos) <= 1e-6

    def render(self, current_node):
        self._ax.plot(current_node.x, current_node.y, "xc")
        plt.pause(0.001)

    def _get_motion_model(self):
        # dx, dy, cost
        motion = [
            [1, 0, 1],
            [1, 1, math.sqrt(2)],
            [0, 1, 1],
            [-1, 1, math.sqrt(2)],
            [-1, 0, 1],
            [-1, -1, math.sqrt(2)],
            [0, -1, 1],
            [1, -1, math.sqrt(2)],
        ]
        return motion

    def _calc_index(self, node):
        """Compute index of a node. Each node is indexed from top left to right down"""
        assert isinstance(node, AStarNode), f"{type(node)} should be AStarNode object"
        return roundint((node.y - self._miny) / self._reso * self._nx + (node.x - self._minx) / self._reso)

    def _verify_node(self, node):
        """Check if specified node is invalid or not"""
        if self._is_violate_area(node):
            return False

        if self._is_collided(node):
            return False

        return True

    def _is_collided(self, pos):
        return False

    def _is_violate_area(self, node):
        return (
            (node.x < self._minx - 1e-6)
            or (node.y < self._miny - 1e-6)
            or (node.x > self._maxx + 1e-6)
            or (node.y > self._maxy + 1e-6)
        )

    def _compute_distance_cost(self, pos1, pos2, w=1.0):
        return w * np.linalg.norm(pos1 - pos2)

    def _estimate_cost(self, node1, node2, w_distance=1.0):
        return self._compute_distance_cost(node1.pos, node2.pos, w_distance)

    def _calc_final_path(self, closed_set, goal):
        # Generate final course
        parent = goal.parent
        vertices = goal.pos
        while parent != -1:
            node = closed_set[parent]
            vertices = np.vstack((node.pos, vertices))
            parent = node.parent
        return vertices


class AStarObsMap(AStarBase):
    def __init__(self, obstacles, *args, **kwargs):
        assert isinstance(obstacles, np.ndarray)
        assert obstacles.ndim == 2 and obstacles.shape[1] == 2
        minx = np.min(obstacles[:, 0])
        maxx = np.max(obstacles[:, 0])
        miny = np.min(obstacles[:, 1])
        maxy = np.max(obstacles[:, 1])
        super().__init__(minx, maxx, miny, maxy, *args, **kwargs)
        self._obstacles = obstacles
        self._obstacles_indices = (obstacles[:, 1] - self._miny) / self._reso * self._nx + (
            obstacles[:, 0] - self._minx
        ) / self._reso

    def _init_plt(self):
        super()._init_plt()
        if self._show_process:
            self._ax.plot(self._obstacles.T[0], self._obstacles.T[1], ".k")

    def _is_collided(self, node):
        return np.isin(self._calc_index(node), self._obstacles_indices)


def main(show_animation=True):
    print(__file__ + " start!!")

    # start and goal position
    start = np.array([0, 0])
    goal = np.array([20, 20])
    resolution = 1.0

    # set obstable positions
    ox, oy = [], []
    for i in range(-4, 24):
        ox.append(i)
        oy.append(-4.0)
    for i in range(-4, 24):
        ox.append(24.0)
        oy.append(i)
    for i in range(-4, 24):
        ox.append(i)
        oy.append(24.0)
    for i in range(-4, 24):
        ox.append(-4.0)
        oy.append(i)
    for i in range(-4, 16):
        ox.append(4.0)
        oy.append(i)
    for i in range(0, 20):
        ox.append(16.0)
        oy.append(24.0 - i)
    obstacles = np.array([ox, oy]).T

    a_star = AStarObsMap(obstacles, resolution, show_process=True)
    path = a_star.planning(start, goal)

    if show_animation:  # pragma: no cover
        plt.plot(path[:, 0], path[:, 1], "-r")
        plt.show()

    print("Generated path is : \r\n{}".format(path))


if __name__ == "__main__":
    main()
