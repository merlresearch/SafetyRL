# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import datetime
import logging
import os
from time import sleep

import numpy as np
from tqdm import tqdm

from safety_rl.envs.goselo_env import GoseloEnv
from safety_rl.experiments.dataset import save_path
from safety_rl.misc.utils import compute_deg_in_polar_coord, rotate
from safety_rl.path_planner.a_star_safety_gym import AStarSafetyGym

np.set_printoptions(precision=4)


def generate_dataset(
    global_dataset,
    lock,
    n_generated_transitions,
    n_max_transition,
    save_data,
    output_dim,
    img_size,
    n_way_points,
    idx_core,
    config,
    show_process=False,
    resolution=0.1,
    skip_points=2,
):
    assert output_dim > 1
    np.random.seed(idx_core)

    logger = logging.getLogger("safety_rl")

    goselo_env = GoseloEnv(config, img_reso=0.5, out_img_size=img_size, visualize_way_points=show_process)

    path_planner = AStarSafetyGym(goselo_env, resolution=resolution, show_process=False)

    while n_generated_transitions.value < n_max_transition:
        # Generate optimal path using A*
        way_points_world_coord = path_planner.planning()

        # Convert to GOSELO expression
        inputs, way_points, goal_dists = to_goselo_format(
            goselo_env, way_points_world_coord, n_way_points, show_process
        )

        if len(inputs) == 0:
            continue

        indices = np.arange(inputs.shape[0])
        indices = indices[indices % skip_points == 0]
        inputs = inputs[indices]
        way_points = way_points[indices]
        goal_dists = goal_dists[indices]

        n_generated_transitions.value += inputs.shape[0]

        # Write locally collected data to global dataset
        with lock:
            if save_data:
                global_dataset.add(inputs=inputs, way_points=way_points, goal_dists=goal_dists)
            stored_size = global_dataset.get_stored_size()
        logger.info(
            "Processed {0: 6d}, stored {1: 6d} transitions on {2}".format(
                n_generated_transitions.value, stored_size, idx_core
            )
        )


def decode_way_points_to_world_coord(current_pos, goal_pos, abs_vertices):
    assert isinstance(current_pos, np.ndarray) and isinstance(goal_pos, np.ndarray)
    abs_vertices = np.copy(abs_vertices)

    # Flip horizontally
    current_pos[1] *= -1
    goal_pos[1] *= -1

    # Compute rotation angle
    theta = compute_deg_in_polar_coord(origin=current_pos, query=goal_pos)

    # Apply rotation matrix to opposite direction
    way_points = rotate(theta=np.deg2rad(90 - theta) * (-1), vertices=abs_vertices)

    # Multiply by -1
    way_points[:, 1] *= -1

    return way_points


def to_goselo_format(goselo_env, vertices_world_coord, n_way_points, show_process, way_points_interval=1):
    """
    A*で生成された経路をもらってGOSELO形式のデータを生成する
    Args:
        goselo_env:
        vertices_world_coord:
        n_way_points:
        show_process:
        way_points_interval:

    Returns:

    """
    assert isinstance(goselo_env, GoseloEnv)

    # Append vertices to compute way points
    goal_vertice = vertices_world_coord[-1]
    assert np.array_equal(goal_vertice, goselo_env.goal_pos[:2])

    for _ in range(n_way_points * way_points_interval):
        vertices_world_coord = np.append(vertices_world_coord, np.array([goal_vertice]), axis=0)
    assert vertices_world_coord.ndim == 2, "`vertices` shape {} is wrong. Should be ndim==2".format(
        vertices_world_coord.shape
    )

    inputs, way_points, goal_dists = [], [], []

    for idx in range(0, vertices_world_coord.shape[0] - n_way_points * way_points_interval, way_points_interval):
        # for idx, current_pos in enumerate(vertices_world_coord[:-1 - n_way_points * way_points_interval]):
        current_pos = vertices_world_coord[idx]

        # Update agent position
        goselo_env.set_robot_pos(current_pos)

        # Remove vertex that is too close each other when discretized
        discrete_robot_pos = goselo_env.to_img_pos(goselo_env.robot_pos)
        discrete_goal_pos = goselo_env.to_img_pos(goselo_env.goal_pos)
        if np.array_equal(discrete_robot_pos, discrete_goal_pos):
            continue

        # Update path log
        goselo_env.save_current_pos_to_pathlog()

        # Get vertices to rotate
        target_vertices = vertices_world_coord[
            idx + way_points_interval : idx + (n_way_points + 1) * way_points_interval : way_points_interval
        ].copy()
        target_vertices -= current_pos

        # Convert world coord. to GOSELO coord.
        _way_points = to_goselo_coordinate(goselo_env, target_vertices, way_points_interval)

        # Store observation and best action computed from A*
        inputs.append(goselo_env._get_obs())
        way_points.append(np.ravel(_way_points))
        goal_dists.append([np.linalg.norm(goselo_env.goal_pos[:2] - goselo_env.robot_pos[:2])])

        if show_process:
            goselo_env.set_reference_path_world_coord(target_vertices + current_pos)
            goselo_env.render()

    return np.array(inputs), np.array(way_points), np.array(goal_dists)


def to_goselo_coordinate(env, vertices, way_points_interval):
    goal_pos = np.copy(env.goal_pos[:2])
    agent_pos = np.copy(env.robot_pos[:2])
    vertices = np.copy(vertices)

    # Flip horizontally
    agent_pos[1] *= -1
    goal_pos[1] *= -1
    vertices[:, 1] *= -1

    # Compute rotation angle
    theta = compute_deg_in_polar_coord(origin=agent_pos, query=goal_pos)

    # Apply rotation matrix
    abs_way_points = rotate(theta=np.deg2rad(90 - theta), vertices=vertices)

    return abs_way_points


def dump_data(global_dataset, lock, output_dir, is_mp=True, n_data_per_file=1000):
    logger = logging.getLogger("safety_rl")

    logger.debug("start dumping data...")
    """Dump data to .pkl file"""
    # Wait until data generation done
    while True and is_mp:
        with lock:
            is_collected_data = global_dataset.get_stored_size() == global_dataset.get_buffer_size()
        if is_collected_data:
            break
        else:
            sleep(3)

    logger.info("Start to generate dataset...")

    # output_path_prefix = os.path.join(
    #     output_dir,
    #     datetime.datetime.now().strftime("%Y%m%dT%H%M%S"))
    # output_path_prefix += "_waypoints"
    # output_path_prefix += "_dataset"

    with lock:
        sample_size = global_dataset.get_stored_size()
        n_iter = int(sample_size / n_data_per_file)

        for i in tqdm(range(n_iter)):
            idx = np.arange(n_data_per_file) + i * n_data_per_file
            filename = os.path.join(output_dir, "{}.pkl".format(n_data_per_file * i))
            save_path(samples=global_dataset.encode_sample(idx), filename=filename)
            logger.info("Saving {}".format(filename))

    logger.info("Finished generating dataset.")


def get_dataset_argument(parser):
    parser.add_argument("--dataset-size", type=int, default=10000)
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--show-process", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--logging-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO", help="Logging level")
    parser.add_argument("--n-cpu", type=int, default=None)
    return parser
