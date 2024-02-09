# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
from time import sleep, time

from examples.config import get_config
from safety_rl.envs.goselo_env import GoseloEnv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hazards-num", type=int, default=10)
    parser.add_argument("--field-size", type=int, default=2)
    parser.add_argument("--resolution", type=float, default=0.1)
    parser.add_argument("--wait", type=float, default=0.0)
    args = parser.parse_args()

    config, _ = get_config(robot_type="point", field_size=args.field_size, place_room=True, room_type=1)
    env = GoseloEnv(config, img_reso=0.5)
    while True:
        env.reset()
        start_time = time()
        n_step = 0
        for _ in range(100):
            _, _, done, _ = env.step(env.action_space.sample())
            env.render("debug")
            n_step += 1
            if done:
                break
        elapsed = time() - start_time
        print("{} FPS (elapsed: {}, step: {})".format(n_step / elapsed, elapsed, n_step))
        sleep(args.wait)
