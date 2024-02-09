# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import csv
import math

import numpy as np


def write_path(filename, vertices):
    """
    Write vertices to a file

    :param filename (str): Filename to write vertices
    :param vertices (np.ndarray): Vertices to write to file
    """
    with open(filename, "w") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(vertices)


def load_path(filename):
    """
    Load vertices from file

    :param filename (str): File name from which vertices will be loaded
    :return (np.ndarray): Vertices
    """
    with open(filename, "r") as f:
        reader = csv.reader(f)

        vertices = []
        for row in reader:
            vertices.append(row)

    return np.asarray(vertices, np.float)


def roundint(data):
    """
    Round number to nearest integer.
    roundint(0.5) = 0
    :param data:
    :return:
    """
    return int(round(data))


def divide_path(
    vertices, divide_num, divide_equally=False, max_divide_distance=0.01, default_min_divide_distance=0.005
):
    """
    Divide vertices more detail
    Args:
        vertices: Vertices that consist a trajectory
        divide_num: Number of points to divide trajectory
        divide_equally: Flag whether to divide vertices with equal distance
        max_divide_distance: Ensure distance between adjacent angles is less than this value
        default_min_divide_distance: Ensure distance between adjacent locations is more than this value

    Returns:
        vertices_new: Divided vertices
    """
    assert isinstance(divide_num, int) and divide_num > 0
    if divide_num == 1:
        return vertices

    if divide_equally:
        # Compute minimum distance to divide vertices
        vecs_norm = []
        for i in range(vertices.shape[0] - 1):
            dist = np.linalg.norm(vertices[i + 1] - vertices[i])
            vecs_norm.append(dist)
        # Ensure minimum distance is bigger than `min_divide_distance` and
        # smaller than `max_divide_distance`
        min_divide_distance = max(min(min(vecs_norm), max_divide_distance), default_min_divide_distance)

    vertices_new = vertices[0]
    for i in range(vertices.shape[0] - 1):
        if divide_equally:
            # divide_num = int(vecs_norm[i] / min_divide_distance)
            divide_num = math.ceil(vecs_norm[i] / min_divide_distance)
            if vecs_norm[i] == 0:
                continue
            elif divide_num == 0:
                divide_num = 1
        vec_unit = (vertices[i + 1] - vertices[i]) / divide_num
        if i != 0:
            vertices_new = np.vstack((vertices_new, vertices[i]))
        for j in range(1, divide_num):
            point_new = vertices[i] + j * vec_unit
            vertices_new = np.vstack((vertices_new, point_new))
    vertices_new = np.vstack((vertices_new, vertices[-1]))

    return vertices_new


def in_collision_area(query, obstacles):
    """
    Check collision
    Args:
        query (np.ndarray): Query
        obstacles (np.ndarray): Obstacles

    Returns:

    """
    if obstacles is None:
        return False
    return ((obstacles[:, 0, :] <= query) & (query <= obstacles[:, 1, :])).all(axis=1).any()


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


def find_kth_nearest_point(query, points, k=0):
    assert isinstance(k, int)
    if k == 0:
        return find_nearest_point(query, points)
    else:
        distances = np.linalg.norm(points - query, axis=1)
        min_index = np.argmin(distances)
        kth_index = max(0, min(min_index + k, distances.shape[0] - 1))
        kth_distance = distances[kth_index]
        return kth_index, kth_distance
