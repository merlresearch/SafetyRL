# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


def to_abs_way_points(rel_way_points):
    assert rel_way_points.ndim == 2 and rel_way_points.shape[1] == 2

    abs_way_points = rel_way_points.copy()
    for i in range(abs_way_points.shape[0] - 1):
        abs_way_points[i + 1] = rel_way_points[i + 1] + abs_way_points[i]

    assert rel_way_points.shape == abs_way_points.shape
    return abs_way_points


def to_rel_way_points(abs_way_points):
    assert abs_way_points.ndim == 2 and abs_way_points.shape[1] == 2

    rel_way_points = abs_way_points.copy()
    for i in range(rel_way_points.shape[0] - 1):
        rel_way_points[i + 1] = abs_way_points[i + 1] - abs_way_points[i]

    assert abs_way_points.shape == rel_way_points.shape
    return rel_way_points
