# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import math

import numpy as np


def compute_deg_in_polar_coord(origin, query):
    assert isinstance(origin, np.ndarray) and isinstance(query, np.ndarray)
    assert origin.ndim == query.ndim == 1
    assert origin.shape[0] == query.shape[0] == 2, "dimension mismatch: origin: {} query: {} expected {}".format(
        origin.shape, query.shape, "(2, x)"
    )
    # x position of origin and query is the same
    if query[0] == origin[0]:
        if query[1] > origin[1]:
            return 90
        else:
            return 270

    theta = math.atan((float(query[1]) - float(origin[1])) / (float(query[0]) - float(origin[0]))) * 180 / math.pi

    # If the origin is left side to the query, then plus 180 deg
    # because the output range of `math.atan` is [-90, 90]
    if float(query[0]) - float(origin[0]) < 0.0:
        theta += 180

    while theta > 360:
        theta -= 360

    return theta


def rotate(theta, vertices, flip=False):
    """

    Args:
        theta (float):  Rotation angle in **RADIAN**
        vertices (np.ndarray): Vertices to rotate
        flip (bool): If True, suppose Z-axis is flipping.

    Returns:

    """
    # assert theta <= 2 * np.pi + 1e-6 and theta >= -2 * np.pi - 1e-6

    single_input = vertices.ndim == 1
    if single_input:
        vertices = np.expand_dims(vertices, axis=0)
    assert vertices.shape[1] == 2
    if flip:
        theta *= -1
    S = np.sin(theta)
    C = np.cos(theta)
    rotation_mtx = np.array([[C, -S], [S, C]])
    if single_input:
        return np.matmul(rotation_mtx, vertices.T).T[0]
    else:
        return np.matmul(rotation_mtx, vertices.T).T


def find_nearest_point(query, points):
    """
    Find nearest point and its distance between query and points
    Args:
        query: [1, {# of dimensions}] array
        points: [N, {# of dimensions}] array

    Returns:
        min_index: Way point index that is closest to current agent position
        min_distance: Distance between agent and the closest way point

    """
    distances = np.linalg.norm(points - query, axis=1)
    min_index = np.argmin(distances)
    min_distance = distances[min_index]
    return min_index, min_distance
