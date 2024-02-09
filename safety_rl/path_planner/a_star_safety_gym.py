# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from time import sleep

import numpy as np
from safety_gym.envs.engine import ResamplingError

from safety_rl.envs.engine_wrapper import EngineWrapper
from safety_rl.misc.logger import initialize_logger
from safety_rl.path_planner.a_star import AStarBase, AStarNode
from safety_rl.path_planner.utils import divide_path


class AStarSafetyGym(AStarBase):
    """
    Generate optimal path using A*
    """

    def __init__(self, env, *args, **kwargs):
        assert isinstance(env, EngineWrapper), "Please use EngineWrapper class"
        minx, miny, maxx, maxy = env.placements_extents
        super().__init__(minx, maxx, miny, maxy, *args, **kwargs)
        self.env = env
        self.reset_sim()
        self._logger = initialize_logger(logging_level=logging.getLevelName("INFO"), save_log=False)

    def to_discrete_pos(self, pos):
        return (pos / self._reso).astype(np.int32)

    def to_continuous_pos(self, pos):
        return (pos * self._reso).astype(np.float32)

    def reset_sim(self):
        self.env.reset()

    def _init_setting(self, init_start, init_goal):
        if not (init_start is not None) or (init_goal is not None):
            start = self.sample_collision_free_pos()
            for _ in range(1000):
                goal = self.sample_collision_free_pos()
                if np.linalg.norm(start - goal) < 3.0:
                    continue
                if not np.array_equal(start, goal):
                    break
        else:
            start = init_start.copy()
            goal = init_goal.copy()
        self.env.set_robot_pos(start)
        self.env.set_goal_pos(goal)
        if hasattr(self.env, "_initialize_map"):
            self.env._initialize_map()
        return start, goal

    def _is_collided(self, node):
        return self._is_collided_from_pos(node.pos)

    def _is_collided_from_pos(self, pos):
        if self.env._is_collided_with_pillars_from_keepout(robot_pos=pos):
            return True
        if self.env._is_collided_with_hazards(robot_pos=pos)[0]:
            return True
        if self.env._is_collided_with_room_walls(robot_pos=pos):
            return True
        if self.env._is_collided_with_gremlins(robot_pos=pos):
            return True
        return False

    def _calc_final_path(self, closed_set, goal_node):
        final_path = super()._calc_final_path(closed_set, goal_node)
        final_path = self.shortcut(final_path)
        return final_path

    def shortcut(
        self,
        vertices,
        max_iteration=100,
        min_points=5,
        max_points=20,
        min_distance=0.01,
        n_initial_divide=3,
        divide_equally=True,
    ):
        """
        Based on "Fast smoothing of manipulator trajectories
                  using optimal bounded-acceleration shortcuts"
                  https://ieeexplore.ieee.org/document/5509683
        Args:
            vertices (np.ndarray): Vertices that consist reference path
            max_iteration:
            min_points (int): Minimum number of vertices
                If generated path consists of less than this number of vertices, force stop iteration
            max_points (int): Maximum number of vertices allowed to skip in one iteration.
            min_distance (float): Distance to check collision.
                If this value is 1, and distance between two points is 3, then we check collision with 3 (=3/1) points.
            n_initial_divide:
            divide_equally:
        Returns:

        """
        vertices = divide_path(vertices, n_initial_divide)
        start_vertice, goal_vertice = vertices[0], vertices[-1]
        self._logger.debug("Started with following {} vertices\r\n{}".format(vertices.shape[0], vertices))

        for _ in range(max_iteration):
            if vertices.shape[0] < min_points:
                break
            # Choose vertice to start shortcut
            start_vertice_idx = np.random.randint(vertices.shape[0] - 2)
            # Choose target vertice to shortcut
            last_vertice_idx = min(
                start_vertice_idx + min(max_points, np.random.randint(2, vertices.shape[0] - start_vertice_idx)),
                vertices.shape[0],
            )

            # Check collision between chosen two points
            n_way_points = int(np.linalg.norm(vertices[last_vertice_idx] - vertices[start_vertice_idx]) / min_distance)
            diff = (vertices[last_vertice_idx] - vertices[start_vertice_idx]) / n_way_points
            # Remove two points, i.e. start and finish point since they are ensured not to be collided
            n_way_points -= 2
            collided = False
            for i in range(n_way_points):
                target_vertice = vertices[start_vertice_idx] + diff * (i + 1)
                # collided_with_hazards = self.env._is_collided_with_hazards(robot_pos=target_vertice)[0]
                # collided_with_pillars = self.env._is_collided_with_pillars(robot_pos=target_vertice)
                # collided_with_walls = self.env._is_collided_with_room_walls(robot_pos=target_vertice)
                if self._is_collided_from_pos(pos=target_vertice):
                    self._logger.debug("Collided\t from {} to {}".format(start_vertice_idx, last_vertice_idx))
                    collided = True
                    break

            # Update reference path
            if not collided:
                vertices = np.vstack((vertices[: start_vertice_idx + 1], vertices[last_vertice_idx:]))
                assert (
                    vertices[0] == start_vertice
                ).all(), "[error] resulted start: {}\t expected: {}\t (start, last)=({}, {})".format(
                    vertices[0], start_vertice, start_vertice_idx, last_vertice_idx
                )
                assert (
                    vertices[-1] == goal_vertice
                ).all(), "[error] resulted goal: {}\t expected: {}\t (start, last)=({}, {})".format(
                    vertices[-1], goal_vertice, start_vertice_idx, last_vertice_idx
                )
                self._logger.debug(
                    "short cut succeeded! current n_vertices: {}\t from {} to {}".format(
                        vertices.shape[0], start_vertice_idx, last_vertice_idx
                    )
                )

        if divide_equally:
            vertices = divide_path(
                vertices,
                3,
                divide_equally=True,
                max_divide_distance=self._reso * 1.1,
                default_min_divide_distance=self._reso * 0.9,
            )
            self._logger.debug("Divided into {} points".format(vertices.shape[0]))
        self._logger.debug("Resulted path: \n{}".format(np.linalg.norm(vertices[1:] - vertices[:-1], axis=1)))
        return vertices

    def sample_collision_free_pos(self):
        for _ in range(10000):
            pos = np.array(
                [
                    np.random.randint(self._nx) * self._reso + self._minx,
                    np.random.randint(self._ny) * self._reso + self._miny,
                ],
                dtype=np.float32,
            )
            node = AStarNode(x=pos[0], y=pos[1], cost=0.0, parent=0)
            if self._verify_node(node):
                return pos
        raise ResamplingError("Failed to sample layout of objects")

    def visualize_path(self, results):
        self.env.set_goal_pos(results[-1])
        for idx, current_pos in enumerate(results[1:-1]):
            self.env.set_robot_pos(current_pos)
            self.env.set_reference_path_world_coord(results[idx + 1 :])
            for _ in range(10):
                self.env.render()
                sleep(0.01)


if __name__ == "__main__":
    config = {
        "robot_base": "xmls/car.xml",
        "task": "goal",
        "hazards_num": 10,
    }
    env = EngineWrapper(config)
    planner = AStarSafetyGym(env=env, resolution=0.1, show_process=True)
    results = planner.planning()
    print("Generated path: \r\n", results)
    planner.visualize_path(results=results)
