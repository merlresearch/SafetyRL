# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import glob
import logging
import os

import numpy as np
from safety_gym.envs.engine import Engine


def get_config(
    robot_type="point",
    field_size=2,
    no_obs=False,
    remove_lidar=False,
    hazards_num=0,
    observe_hazards=False,
    hazards_order=False,
    pillars_num=10,
    observe_pillars=True,
    pillars_order=False,
    gremlins_num=0,
    observe_gremlins=False,
    dummy_gremlins=False,
    point_straight=False,
    place_room=False,
    room_type=0,
    **kwargs
):
    assert robot_type in ["point", "car", "doggo", "point_straight"]

    logger = logging.getLogger("safety_rl")

    if place_room:
        no_obs = True
        observe_hazards = False
        observe_pillars = False
        observe_gremlins = False
        observe_room_walls = True
        name = "two_room" if room_type == 0 else "four_room"
    elif no_obs:
        hazards_num, pillars_num, gremlins_num = 0, 0, 0
        observe_hazards, observe_pillars, observe_gremlins, observe_room_walls = False, False, False, False
        name = "no_obs"
    else:
        observe_room_walls = False
        # Assume we don't use hazards and pillars at the same time
        if hazards_num * pillars_num * gremlins_num > 0:
            logger.warning(
                """
                You specified more than two kinds of obstacles from following {gremlins, hazards, pillars}.\n
                The priority is gremlins > hazards > pillars."""
            )

        if gremlins_num > 0:
            name = "gremlins"
            hazards_num, pillars_num = 0, 0
            observe_gremlins = True
            observe_hazards, observe_pillars = False, False
        elif hazards_num > 0:
            name = "gremlins" if dummy_gremlins else "hazards"
            pillars_num, gremlins_num = 0, 0
            hazards_order = True
            observe_hazards = True
            observe_pillars, observe_gremlins = False, False
        elif pillars_num > 0:
            name = "pillars"
            hazards_num, gremlins_num = 0, 0
            observe_hazards, observe_gremlins = False, False
        else:
            name = "no_obs"
            hazards_num, pillars_num, gremlins_num = 0, 0, 0
            observe_hazards, observe_pillars, observe_gremlins = False, False, False

    name += "_{}_{}".format(int(field_size), int(hazards_num + gremlins_num + pillars_num))

    if dummy_gremlins:
        hazards_size = Engine.DEFAULT["gremlins_size"] * np.sqrt(2)
    else:
        hazards_size = Engine.DEFAULT["hazards_size"]

    if remove_lidar:
        observe_hazards = False
        observe_pillars = False
        observe_room_walls = False

    if robot_type == "point_straight":
        robot_type = "point"
        point_straight = True

    config = {
        "robot_base": "xmls/{}.xml".format(robot_type),
        "task": "goal",
        "continue_goal": False,
        "point_straight": point_straight,
        "hazards_num": 0 if no_obs else hazards_num,
        "gremlins_num": 0 if no_obs else gremlins_num,
        "pillars_num": 0 if no_obs else pillars_num,
        "hazards_size": hazards_size,
        "observe_hazards": observe_hazards,
        "observe_gremlins": observe_gremlins,
        "observe_pillars": observe_pillars,
        "observe_room_walls": observe_room_walls,
        "place_room": place_room,
        "room_type": room_type,
        "hazards_order": hazards_order,
        "pillars_order": pillars_order,
        "placements_extents": [-1 * field_size, -1 * field_size, field_size, field_size],
    }
    return config, name


def get_max_steps():
    max_steps = {"point": int(1e6), "car": int(1e6), "doggo": int(1e6)}
    return max_steps


def get_frequencies():
    return 4, 8, 16, 32, 64


def get_common_argument(parser):
    if parser is None:
        parser = argparse.ArgumentParser(conflict_handler="resolve")
    parser.add_argument("--root-dir", type=str, default=".")
    return parser


def get_config_argument(parser):
    if parser is None:
        parser = argparse.ArgumentParser(conflict_handler="resolve")
    parser.add_argument("--place-room", action="store_true")
    parser.add_argument("--room-type", type=int, default=0)
    parser.add_argument("--remove-lidar", action="store_true")
    parser.add_argument("--hazards-num", type=int, default=0)
    parser.add_argument("--pillars-num", type=int, default=10)
    parser.add_argument("--gremlins-num", type=int, default=0)
    parser.add_argument("--field-size", type=float, default=2)
    parser.add_argument("--no-obs", action="store_true")
    parser.add_argument("--dummy-gremlins", action="store_true")
    parser.add_argument("--robot-type", default="point", choices=["point", "car", "doggo", "point_straight"])
    return parser


DEFAULT_EXP_NAME = "pillars_2_10"


def get_all_exp_settings(is_cmd_args=True):
    if is_cmd_args:
        return [
            {"name": "pillars_2_10", "cmd_args": "--field-size 2 --pillars-num 10"},
            {"name": "pillars_3_25", "cmd_args": "--field-size 3 --pillars-num 25"},
            {"name": "pillars_4_40", "cmd_args": "--field-size 4 --pillars-num 40"},
            {"name": "gremlins_2_10", "cmd_args": "--field-size 2 --hazards-num 10 --dummy-gremlins"},
            {"name": "two_room", "cmd_args": "--field-size 2 --place-room --room-type 0"},
            {"name": "four_room", "cmd_args": "--field-size 2 --place-room --room-type 1"},
        ]
    else:
        return [
            {"name": "pillars_2_10", "kwargs": {"field_size": 2, "pillars_num": 10}},
            {"name": "pillars_3_25", "kwargs": {"field_size": 3, "pillars_num": 25}},
            {"name": "pillars_4_40", "kwargs": {"field_size": 4, "pillars_num": 40}},
            {"name": "gremlins_2_10", "kwargs": {"field_size": 2, "hazards-num": 10, "dummy_gremlins": True}},
            {"name": "two_room", "kwargs": {"field_size": 2, "place_room": True, "room_type": 0}},
            {"name": "four_room", "kwargs": {"field-size": 2, "place_room": True, "room_type": 1}},
        ]


def get_policy_model_directories(root_dir, exp_type=DEFAULT_EXP_NAME, method="ours", robot_type="point"):
    logdir = os.path.join(root_dir, "dataset", exp_type, method.replace("_finetune", ""))
    checkpoints = glob.glob("{}/{}/*/checkpoint".format(logdir, robot_type))
    return [os.path.dirname(checkpoint) for checkpoint in checkpoints]
