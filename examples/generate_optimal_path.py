# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse

from examples.config import get_config, get_config_argument
from safety_rl.envs.engine_wrapper import EngineWrapper
from safety_rl.path_planner.a_star_safety_gym import AStarSafetyGym


def generate_path(env, resolution, show_process=False):
    while True:
        env.reset()
        path_planner = AStarSafetyGym(env, resolution=resolution, show_process=False)
        results = path_planner.planning()
        if show_process and results is not None:
            path_planner.visualize_path(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=float, default=0.1)
    parser = get_config_argument(parser)
    args = parser.parse_args()

    config, _ = get_config(
        robot_type="doggo",
        field_size=args.field_size,
        no_obs=args.no_obs,
        place_room=args.place_room,
        room_type=args.room_type,
        hazards_num=args.hazards_num,
        pillars_num=args.pillars_num,
        gremlins_num=args.gremlins_num,
    )
    env = EngineWrapper(config, visualize_waypoints=True)
    generate_path(env, args.resolution, show_process=True)
