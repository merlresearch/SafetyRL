# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


import logging
import os
import subprocess

from examples.config import get_all_exp_settings
from examples.rl.all_envs_core import get_argument_parser

template = r"""#!/bin/bash
#SBATCH --array=1-1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task 2
#SBATCH --gres=gpu:{gpu}:1
#SBATCH -J train_cnn_{exp_type}
#SBATCH -o train_cnn_{exp_type}.log
#SBATCH -e train_cnn_{exp_type}_error.log

export PYTHONPATH=.
python examples/train_cnn.py --epochs 1000 --lr 0.001 {cmd_args}
"""


def main():
    parser = get_argument_parser()
    args = parser.parse_args()

    is_run = args.run
    dir_out = os.path.join(args.dir_root, "ours")
    os.makedirs(dir_out, exist_ok=True)

    settings = get_all_exp_settings()

    for idx, setting in enumerate(settings):
        exp_name = setting["name"]
        cmd_args = setting["cmd_args"]
        gpu = "TitanV" if idx == 0 else "QuadroRTX6000"
        cur_batch = template.format(gpu=gpu, exp_type=exp_name, cmd_args=cmd_args)
        cur_out = os.path.join(dir_out, f"{exp_name}.srun")

        with open(cur_out, "w") as f:
            f.write(cur_batch)

        print(f"run {cur_out}")
        if is_run:
            args = ["sbatch", cur_out]
            subprocess.run(args)


if __name__ == "__main__":
    logging.basicConfig(
        datefmt="%d/%Y %I:%M:%S",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)s) %(message)s",
    )
    main()
