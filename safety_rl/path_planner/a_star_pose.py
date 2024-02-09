# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import sys

import matplotlib.patches as pat
import matplotlib.pyplot as plt
import numpy as np

from safety_rl.path_planner.a_star import AStarBase, AStarNode
from safety_rl.path_planner.utils import roundint


class AStarPoseNode(AStarNode):
    def __init__(self, x, y, pose, cost, parent):
        super().__init__(x=x, y=y, cost=cost, parent=parent)
        assert isinstance(pose, int), f"Type of pose must be int, not {type(pose)}"
        self.pose = pose % 8


class AStarPose(AStarBase):
    def planning(self, init_start=None, init_goal=None, n_trial=100):
        motions = self._get_motion_model()
        specify_start_goal = (init_start is not None) and (init_goal is not None)

        for cur_trial in range(n_trial):
            start, goal = self._init_setting(init_start, init_goal)

            assert isinstance(start, np.ndarray) and isinstance(goal, np.ndarray)
            assert start.shape[0] == 2 and goal.shape[0] == 2

            start_node = AStarPoseNode(start[0], start[1], 6, 0.0, -1)
            goal_node = AStarPoseNode(goal[0], goal[1], 0, 0.0, -1)
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
                for next_pose, motion in enumerate(motions):
                    next_pos = np.array(
                        (current_node.x + motion[0] * self._reso, current_node.y + motion[1] * self._reso),
                        dtype=np.float32,
                    )
                    dist_cost = self._compute_distance_cost(current_node.pos, next_pos)
                    pose_cost = self._compute_pose_cost(current_node.pose, next_pose)
                    node = AStarPoseNode(
                        x=next_pos[0],
                        y=next_pos[1],
                        cost=current_node.cost + dist_cost + pose_cost,
                        pose=next_pose,
                        parent=current_id,
                    )
                    n_id = self._calc_index(node)
                    if n_id in closed_set:
                        continue

                    if not self._verify_node(node, current_node):
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

    def _verify_node(self, node, parent_node):
        if super()._verify_node(node) is False:
            return False
        elif np.cos(np.pi / 4.0 * (node.pose - parent_node.pose)) <= 0:
            return False
        return True

    def _calc_index(self, node):
        """Compute index of a node. Each node is indexed from top left to right down"""
        assert isinstance(node, AStarPoseNode)
        pose_idx = node.pose
        pos_idx = roundint((node.y - self._miny) / self._reso * self._nx + (node.x - self._minx) / self._reso)
        return pos_idx * 8 + pose_idx

    def _compute_pose_cost(self, pose1, pose2):
        """
        Suppose the pose is in a set of (0, np.pi/4, ..., np.pi*7/4),
        and compute the cost by its cosine distance but add 1 to change scale
        so that minimum cost will be 0, and maximum to be 2.
        """
        rel_pose = min(abs(pose1 - pose2), abs(pose1 - pose2 - 8), abs(pose1 - pose2 + 8))
        costs = np.array([0, 3, 5, 10, 15])
        return costs[rel_pose]
        # angle = (pose1 - pose2) * np.pi / 4.
        # return np.abs(np.cos(angle) - 1)

    def _is_goal(self, current_node, goal_node):
        is_reached = np.linalg.norm(current_node.pos - goal_node.pos) <= 1e-6
        # is_same_pose = current_node.pose == goal_node.pose
        return is_reached  # and is_same_pose

    def _estimate_cost(self, node1, node2, w_distance=1.0, w_pose=1.0):
        dist_cost = self._compute_distance_cost(node1.pos, node2.pos, w_distance)
        pose_cost = self._compute_pose_cost(node1.pose, node2.pose)
        return dist_cost + pose_cost


class AStarObsMap(AStarPose):
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
        return np.isin(self._calc_index(node) // 8, self._obstacles_indices)

    def _calc_final_path(self, closed_set, goal):
        # Generate final course
        parent = goal.parent
        vertices = goal.pos
        poses = [goal.pose]
        costs = []
        while parent != -1:
            node = closed_set[parent]
            vertices = np.vstack((node.pos, vertices))
            poses.append(node.pose)
            parent = node.parent
            if parent != -1:
                print(
                    f"{closed_set[parent].pos},{closed_set[parent].pose} to {node.pos},{node.pose} with cost={self._estimate_cost(node, closed_set[parent])}"
                )
                costs.append(self._estimate_cost(node, closed_set[parent]))
        return vertices, poses


def main(show_animation=True):
    print(__file__ + " start!!")

    # start and goal position
    start = np.array([0, 0])
    goal = np.array([20, 20])
    # goal = np.array([3, 3])
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
    path, pose = a_star.planning(start, goal)

    if show_animation:  # pragma: no cover
        plt.plot(path[:, 0], path[:, 1], "-r")
        plt.show()

    print("Generated path is : \n{}\n{}".format(path, pose))


if __name__ == "__main__":
    main()
