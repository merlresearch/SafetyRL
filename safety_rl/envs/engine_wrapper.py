# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from collections import OrderedDict

import numpy as np
from safety_gym.envs.engine import Engine

from safety_rl.misc.logger import initialize_logger
from safety_rl.path_planner.utils import divide_path

COLOR_WAYPOINT = np.array([0, 0.5, 0.5, 1])

GROUP_WAYPOINT = 7


class EngineWrapper(Engine):
    def __init__(
        self,
        *args,
        n_waypoints=10,
        visualize_waypoints=False,
        visualize_only_raw_waypoints=False,
        logging_level="INFO",
        **kwargs,
    ):
        assert isinstance(n_waypoints, int)
        self._allow_confliction = False
        self._logger = initialize_logger(logging_level=logging.getLevelName(logging_level))
        self._n_waypoints = n_waypoints
        self._n_way_points_for_vis = n_waypoints
        if not visualize_only_raw_waypoints:
            self._n_way_points_for_vis *= 50

        self._way_point_size = 0.05
        self._visualize_waypoints = visualize_waypoints
        self._visualize_only_raw_waypoints = visualize_only_raw_waypoints
        self._raw_reference_path_world_coord = np.zeros(shape=(self._n_waypoints, 2), dtype=np.float32)
        self._reference_path_world_coord = np.zeros(shape=(self._n_waypoints, 2), dtype=np.float32)
        super().__init__(*args, **kwargs)

    def step(self, action):
        next_obs, rew, done, info = super().step(action)
        info["collided"] = self._is_collided()
        return next_obs, rew, done, info

    def pillar_cost(self):
        pillar_cost = 0.0
        for contact in self.data.contact[: self.data.ncon]:
            geom_ids = [contact.geom1, contact.geom2]
            geom_names = sorted([self.model.geom_id2name(g) for g in geom_ids])
            if any(n in self.robot.geom_names for n in geom_names):
                pillar_cost += self.pillars_cost
        return pillar_cost

    def hazard_cost(self):
        hazard_cost = 0.0
        for h_pos in self.hazards_pos:
            h_dist = self.dist_xy(h_pos)
            if h_dist <= self.hazards_size:
                hazard_cost += self.hazards_cost * (self.hazards_size - h_dist)
        return hazard_cost

    def _is_collided(self, robot_pos=None):
        collided_with_hazards = self._is_collided_with_hazards(robot_pos)[0]
        collided_with_pillars = self._is_collided_with_pillars(robot_pos)
        collided_with_walls = self._is_collided_with_room_walls(robot_pos)
        return collided_with_hazards or collided_with_pillars or collided_with_walls

    def _is_collided_with_hazards(self, robot_pos=None):
        if len(self.hazards_pos) == 0:
            return False, -1

        if robot_pos is None:
            robot_pos = np.copy(self.robot_pos[:2])
        assert isinstance(robot_pos, np.ndarray)
        assert robot_pos.ndim == 1 and robot_pos.shape[0] == 2

        distances = np.linalg.norm(np.array(self.hazards_pos)[:, :2] - robot_pos, axis=1)
        for idx, distance in enumerate(distances):
            if distance < self.hazards_keepout:
                return True, idx
        return False, -1

    def _is_collided_with_pillars_from_keepout(self, robot_pos=None):
        if len(self.pillars_pos) == 0:
            return False

        if robot_pos is None:
            robot_pos = np.copy(self.robot_pos[:2])
        assert isinstance(robot_pos, np.ndarray)
        assert robot_pos.ndim == 1 and robot_pos.shape[0] == 2

        distances = np.linalg.norm(np.array(self.pillars_pos)[:, :2] - robot_pos, axis=1)
        return np.any(distances <= self.pillars_keepout)  # + self.robot_keepout * 0.4)

    def _is_collided_with_pillars(self, robot_pos=None):
        if len(self.pillars_pos) == 0:
            return False

        if robot_pos is not None:
            self.set_robot_pos(robot_pos)
        for contact in self.data.contact[: self.data.ncon]:
            geom_ids = [contact.geom1, contact.geom2]
            geom_names = sorted([self.model.geom_id2name(g) for g in geom_ids])
            if any(n.startswith("pillar") for n in geom_names):
                if any(n in self.robot.geom_names for n in geom_names):
                    return True
        return False

    def _is_collided_with_gremlins(self, robot_pos=None):
        if robot_pos is not None:
            self.set_robot_pos(robot_pos)
        for contact in self.data.contact[: self.data.ncon]:
            geom_ids = [contact.geom1, contact.geom2]
            geom_names = sorted([self.model.geom_id2name(g) for g in geom_ids])
            if any(n.startswith("gremlin") for n in geom_names):
                if any(n in self.robot.geom_names for n in geom_names):
                    return True
        return False

    def _is_collided_with_room_walls(self, robot_pos=None):
        if robot_pos is not None:
            self.set_robot_pos(robot_pos)
        for contact in self.data.contact[: self.data.ncon]:
            geom_ids = [contact.geom1, contact.geom2]
            geom_names = sorted([self.model.geom_id2name(g) for g in geom_ids])
            if any(n.startswith("room") for n in geom_names):
                if any(n in self.robot.geom_names for n in geom_names):
                    return True
        return False

    def _set_pos(self, name, pos):
        assert isinstance(pos, np.ndarray)
        assert pos.ndim == 1 and pos.shape[0] == 2
        self.model.body_pos[self.model._body_name2id[name]][:2] = pos

    def set_goal_pos(self, pos):
        assert self.task in ["goal", "push"]
        self._set_pos("goal", pos)
        self.update_layout()

    def set_robot_pos(self, pos):
        assert isinstance(pos, np.ndarray)
        assert pos.ndim == 1 and pos.shape[0] == 2
        if self.robot_base == "xmls/car.xml" or self.robot_base == "xmls/doggo.xml":
            self.sim.data.qpos[:2] = pos
        else:
            self.model.body_pos[self.model._body_name2id["robot"]][:2] = pos
        self.update_layout()

    def set_hazards_pos(self, hazards_pos):
        """
        Force to set hazards position.
        :param hazards_pos: 2D array of hazards position. The second dimension must be 2 because the z-position should be fixed to 0.02
        :return:
        """
        assert isinstance(hazards_pos, np.ndarray)
        assert hazards_pos.ndim == 2 and hazards_pos.shape[1] == 2
        assert hazards_pos.shape[0] == self.hazards_num
        for i, hazard_pos in enumerate(hazards_pos):
            self._set_pos("hazard{}".format(i), hazard_pos)
        self.update_layout()

    def set_pillars_pos(self, pillars_pos):
        """
        Force to set pillars position.
        :param pillars_pos: 2D array of hazards position. The second dimension must be 2 because the z-position should be fixed to 0.02
        :return:
        """
        assert isinstance(pillars_pos, np.ndarray)
        assert pillars_pos.ndim == 2 and pillars_pos.shape[1] == 2
        assert pillars_pos.shape[0] == self.pillars_num
        for i, pillar_pos in enumerate(pillars_pos):
            self._set_pos("pillar{}".format(i), pillar_pos)
        self.update_layout()

    def set_gremlins_obj_pos(self, gremlins_obj_pos):
        """
        Force to set pillars position.
        :param pillars_pos: 2D array of hazards position. The second dimension must be 2 because the z-position should be fixed to 0.02
        :return:
        """
        assert isinstance(gremlins_obj_pos, np.ndarray)
        assert gremlins_obj_pos.ndim == 2 and gremlins_obj_pos.shape[1] == 2
        assert gremlins_obj_pos.shape[0] == self.gremlins_num
        for i, gremlin_obj_pos in enumerate(gremlins_obj_pos):
            self._set_pos("gremlin{}obj".format(i), gremlin_obj_pos)
        self.update_layout()

    def build_world_config(self):
        world_config = super().build_world_config()
        if self._visualize_waypoints:
            for i in range(self._n_way_points_for_vis):
                name = f"way_point_{i}"
                geom = {
                    "name": name,
                    "size": [self._way_point_size, 1e-2],  # self.hazards_size / 2,
                    "pos": np.array([0.0, 0.0, 2e-2]),  # dummy location
                    "rot": self.random_rot(),
                    "type": "cylinder",
                    "contype": 0,
                    "conaffinity": 0,
                    "group": GROUP_WAYPOINT,
                    "rgba": COLOR_WAYPOINT * [1, 1, 1, 1],
                }  # transparent
                world_config["geoms"][name] = geom
        return world_config

    def set_reference_path_world_coord(
        self, abs_way_points_world_coord, max_divide_distance=0.01, default_min_divide_distance=0.005
    ):
        """
        Set way points to the reference path
        Args:
            abs_way_points_world_coord: Reference path to goal location defined in world coordinate
        """
        assert (
            abs_way_points_world_coord.ndim == 2
            and abs_way_points_world_coord.shape[1] == 2
            and abs_way_points_world_coord.shape[0] > 1
        )

        self._raw_reference_path_world_coord = abs_way_points_world_coord.copy()

        _way_points = np.insert(abs_way_points_world_coord.copy(), 0, self.robot_pos[:2], axis=0)
        _way_points = divide_path(
            _way_points,
            divide_num=self._n_waypoints,
            divide_equally=True,
            max_divide_distance=max_divide_distance,
            default_min_divide_distance=default_min_divide_distance,
        )

        n_input_way_points = min(_way_points.shape[0], self._n_waypoints)
        if n_input_way_points == self._n_waypoints:
            self._reference_path_world_coord = _way_points.copy()
        else:
            self._reference_path_world_coord[:n_input_way_points, :] = _way_points.copy()
            for idx in range(n_input_way_points, self._n_waypoints):
                self._reference_path_world_coord[idx] = _way_points[-1].copy()

        if self._visualize_waypoints:
            n_input_way_points = min(_way_points.shape[0], self._n_way_points_for_vis)
            waypoints = self._raw_reference_path_world_coord if self._visualize_only_raw_waypoints else _way_points
            for i in range(self._n_way_points_for_vis):
                if i < n_input_way_points or self._visualize_only_raw_waypoints:
                    self._set_pos("way_point_{}".format(i), waypoints[i])
                else:
                    self._set_pos("way_point_{}".format(i), np.zeros(shape=(2,), dtype=np.float32))
            self.update_layout()

    @property
    def goal_dist(self):
        return np.linalg.norm(self.robot_pos[:2] - self.goal_pos[:2])
