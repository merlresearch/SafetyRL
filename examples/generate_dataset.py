# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import logging
import multiprocessing
import os
import platform
import shutil
from multiprocessing import Lock, Process, Value
from multiprocessing.managers import SyncManager

import numpy as np
from cpprb import ReplayBuffer as CppReplayBuffer

from examples.config import get_config, get_config_argument
from safety_rl.envs.goselo_env import GoseloEnv
from safety_rl.experiments.generate_dataset import dump_data, generate_dataset, get_dataset_argument
from safety_rl.misc.logger import initialize_logger


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser = get_dataset_argument(parser)
    parser.add_argument("--n-way-points", type=int, default=10)
    parser.add_argument("--out-img-size", type=int, default=64)
    parser.add_argument("--resolution", type=float, default=0.1)
    return parser


class ReplayBuffer(CppReplayBuffer):
    def encode_sample(self, idx):
        return self._encode_sample(idx)


def main():
    parser = get_argument_parser()
    parser = get_config_argument(parser)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--logdir", type=str, default="dataset")
    parser.add_argument("--skip-points", type=int, default=2)
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()

    config, exp_name = get_config(
        robot_type="doggo",
        field_size=args.field_size,
        no_obs=args.no_obs,
        place_room=args.place_room,
        room_type=args.room_type,
        hazards_num=args.hazards_num,
        pillars_num=args.pillars_num,
        dummy_gremlins=args.dummy_gremlins,
    )

    logdir = os.path.join(args.logdir, exp_name, "waypoints_generator")
    if args.evaluate:
        logdir += "_eval"

    if os.path.exists(logdir):
        if args.force:
            shutil.rmtree(logdir)
        else:
            raise RuntimeError("{} already exists. Please manually remove it or specify --force")
    else:
        os.makedirs(logdir, exist_ok=True)

    logger = initialize_logger(
        logging_level=logging.getLevelName(args.logging_level), output_dir=logdir, save_log=False
    )

    output_dim = 2 * args.n_way_points
    logger.info("Output is {} way points".format(output_dim))

    img_size = (args.out_img_size, args.out_img_size)

    # Global buffer
    SyncManager.register("ReplayBuffer", ReplayBuffer)
    manager = SyncManager()
    manager.start()

    goselo_env = GoseloEnv(config, img_reso=0.5, out_img_size=img_size)

    replay_buffer_dict = {
        "size": args.dataset_size,
        "default_dtype": np.float32,
        "env_dict": {
            "inputs": {"shape": goselo_env.observation_space.shape},
            "goal_dists": {"shape": (1,)},
            "way_points": {"shape": (output_dim,)},
        },
    }

    global_dataset = manager.ReplayBuffer(**replay_buffer_dict)

    # Share number of generated paths
    n_generated_transitions = Value("i", 0)

    # Lock
    lock = Lock()

    if platform.system() == "Darwin" or args.debug:
        idx_core = 0
        # multiprocessing with GUI does not seem to work on MAC
        generate_dataset(
            global_dataset,
            lock,
            n_generated_transitions,
            args.dataset_size,
            args.save_data,
            output_dim,
            img_size,
            args.n_way_points,
            idx_core,
            config,
            args.show_process,
            args.resolution,
            args.skip_points,
        )
        dump_data(global_dataset, lock, logdir, is_mp=False)
    else:
        tasks = []
        n_cpu = multiprocessing.cpu_count() if args.n_cpu is None else args.n_cpu
        for idx_core in range(n_cpu):
            tasks.append(
                Process(
                    target=generate_dataset,
                    args=[
                        global_dataset,
                        lock,
                        n_generated_transitions,
                        args.dataset_size,
                        args.save_data,
                        output_dim,
                        img_size,
                        args.n_way_points,
                        idx_core,
                        config,
                        args.show_process,
                        args.resolution,
                        args.skip_points,
                    ],
                )
            )
        tasks.append(Process(target=dump_data, args=[global_dataset, lock, logdir, True]))

        for task in tasks:
            task.start()
        for task in tasks:
            task.join()


if __name__ == "__main__":
    main()
