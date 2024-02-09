# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


import logging
import os
import subprocess

from examples.config import get_frequencies
from examples.rl.all_envs_core import get_argument_parser

template = r"""#!/bin/bash

#SBATCH --array=1-{n_experiments}
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task 2
#SBATCH --gres=gpu:1
#SBATCH -J hazards_all_frequency_{frequency}
#SBATCH -o hazards_all_frequency_{frequency}.log
#SBATCH -e hazards_all_frequency_{frequency}_error.log


export PYTHONPATH=.
python examples/rl/run_sac_waypoints_generator.py --max-steps 1000000 \
                                                   --robot-type doggo \
                                                   --test-interval 10000 \
                                                   --test-episodes 10 \
                                                   --robot-type doggo \
                                                   --periodically-update-way-points \
                                                   --remove-lidar \
                                                   --update-interval {frequency} \
                                                   --way-points-model-dir expert_data/pillar \
                                                   --logdir {dir_out}/{frequency}${{SLURM_ARRAY_TASK_ID}}
"""


def main():
    parser = get_argument_parser()
    args = parser.parse_args()

    n_experiments = args.n_experiments
    is_run = args.run

    dir_out = os.path.join(args.dir_root, "compare_frequency")
    os.makedirs(dir_out, exist_ok=True)

    frequencies = get_frequencies()

    for frequency in frequencies:
        cur_batch = template.format(n_experiments=n_experiments, dir_out=dir_out, frequency=frequency)
        cur_out = os.path.join(dir_out, f"{frequency}.srun")

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
