# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import os
import subprocess

from examples.config import get_max_steps


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-experiments", type=int, default=5)
    parser.add_argument("--run", default=False, action="store_true")
    parser.add_argument("--dir-root", default="output", type=str)
    return parser


def run_all_envs(template, dir_out, is_run, n_experiments):
    max_steps = get_max_steps()

    for idx, robot_type in enumerate(["doggo", "car", "point"]):
        max_step = max_steps[robot_type]
        gpu = "TitanV" if idx == 0 else "QuadroRTX6000"
        cur_batch = template.format(n_experiments=n_experiments, robot_type=robot_type, steps=max_step, gpu=gpu)
        cur_out = os.path.join(dir_out, f"{robot_type}.srun")

        with open(cur_out, "w") as f:
            f.write(cur_batch)

        print(f"run {cur_out}")
        if is_run:
            args = ["sbatch", cur_out]
            subprocess.run(args)
