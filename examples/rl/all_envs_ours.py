# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


import logging
import os

from examples.rl.all_envs_core import get_argument_parser, run_all_envs

template = r"""#!/bin/bash

#SBATCH --array=1-{n_experiments}
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task 2
#SBATCH --gres=gpu:{gpu}:1
#SBATCH -J pillars_2_10_ours_{robot_type}
#SBATCH -o pillars_2_10_ours_{robot_type}.log
#SBATCH -e pillars_2_10_ours_{robot_type}_error.log

export PYTHONPATH=.
python examples/rl/run_sac_waypoints_generator.py --max-steps {steps} \
                                                   --robot-type {robot_type} \
                                                   --test-interval 10000 \
                                                   --test-episodes 10
"""


def main():
    parser = get_argument_parser()
    args = parser.parse_args()

    dir_out = os.path.join(args.dir_root, "ours")
    os.makedirs(dir_out, exist_ok=True)
    run_all_envs(template=template, is_run=args.run, dir_out=dir_out, n_experiments=args.n_experiments)


if __name__ == "__main__":
    logging.basicConfig(
        datefmt="%d/%Y %I:%M:%S",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)s) %(message)s",
    )
    main()
