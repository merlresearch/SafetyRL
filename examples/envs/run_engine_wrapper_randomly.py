# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


import argparse

import numpy as np

from examples.config import get_config
from safety_rl.envs.engine_wrapper import EngineWrapper

np.set_printoptions(precision=3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hazards-num", type=int, default=10)
    parser.add_argument("--field-size", type=int, default=2)
    parser.add_argument("--resolution", type=float, default=0.1)
    args = parser.parse_args()

    config, _ = get_config(robot_type="point", field_size=args.field_size, place_room=True, room_type=1)
    env = EngineWrapper(config)

    while True:
        env.reset()
        # env.set_goal_pos(np.zeros(shape=(2,), dtype=np.float32))
        for _ in range(1000):
            action = env.action_space.sample()
            _, _, done, _ = env.step(action)

            print(
                "magnetometer: {} orientation: {}".format(
                    env.world.get_sensor("magnetometer"),
                    np.rad2deg(
                        np.arctan2(env.world.get_sensor("magnetometer")[1], env.world.get_sensor("magnetometer")[0])
                    ),
                )
            )
            env.render()
            if done:
                break
