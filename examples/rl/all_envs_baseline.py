# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


import argparse
import logging
import os

from safety_rl.misc.task_manager import run_in_concurrent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrent", type=int, default=1)
    parser.add_argument("--start_gpu_idx", type=int, default=0)
    parser.add_argument("--dir-out", type=str, default="output")
    args = parser.parse_args()

    concurrent = args.concurrent
    start_gpu_idx = args.start_gpu_idx
    dir_out = args.dir_out

    # python examples/rl/run_sac_waypoints_generator.py --periodically-update-way-points --way-points-model-dir <> --robot-type point --max-steps 1000000 --test-interval 50000
    args = ["python3", "examples/rl/run_sac_baseline.py"]
    args += ["--max-steps", "1000000"]
    args += ["--test-interval", "10000"]

    list_args = []
    robot_types = ["point", "car", "doggo"]

    for cur_seed in range(3):
        for robot_type in robot_types:
            cur_args = args.copy()
            cur_args += ["--logdir", os.path.join(dir_out, "baseline", robot_type, str(cur_seed))]
            cur_args += ["--robot-type", robot_type]
            list_args.append(cur_args)

    run_in_concurrent(list_args, concurrent=concurrent, start_gpu_idx=start_gpu_idx)


if __name__ == "__main__":
    logging.basicConfig(
        datefmt="%d/%Y %I:%M:%S",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)s) %(message)s",
    )
    main()
