# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
import os

from tf2rl.algos.sac import SAC

from examples.config import get_config, get_config_argument
from safety_rl.envs.engine_wrapper import EngineWrapper
from safety_rl.envs.rl_env_wrapper import RLEnvWrapper
from safety_rl.experiments.rl_trainer import RLTrainer
from safety_rl.misc.logger import initialize_logger

if __name__ == "__main__":
    parser = RLTrainer.get_argument()
    parser = SAC.get_argument(parser)
    parser = get_config_argument(parser)
    parser.set_defaults(batch_size=100)
    parser.set_defaults(n_warmup=10000)
    parser.set_defaults(episode_max_steps=300)
    parser.set_defaults(test_episodes=20)
    parser.set_defaults(max_steps=int(5e6))
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--sac-units", type=int, default=256)
    parser.add_argument("--root-dir", type=str, default=".")
    args = parser.parse_args()

    logger = initialize_logger(logging_level=logging.INFO, save_log=False)

    # Settings
    config, exp_type = get_config(
        robot_type=args.robot_type,
        field_size=2.0,
        remove_lidar=args.remove_lidar,
        no_obs=args.no_obs,
        place_room=args.place_room,
        room_type=args.room_type,
    )
    # e.g. "dataset/pillars_2_10/baseline/point"
    args.logdir = os.path.join(args.root_dir, "dataset", exp_type, "baseline", args.robot_type)

    # env
    env = RLEnvWrapper(env=EngineWrapper(config=config))
    test_env = RLEnvWrapper(env=EngineWrapper(config=config))

    # test environment by visualization
    if args.debug:
        for _ in range(100):
            test_env.reset()
            test_env.render()
            for _ in range(100):
                _, _, done, _ = test_env.step(test_env.action_space.sample())
                test_env.render()
                if done:
                    break
        exit()

    # policy
    policy = SAC(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.high.size,
        gpu=args.gpu,
        actor_units=[args.sac_units, args.sac_units],
        critic_units=[args.sac_units, args.sac_units],
        memory_capacity=args.memory_capacity,
        max_action=env.action_space.high[0],
        batch_size=args.batch_size,
        n_warmup=args.n_warmup,
        alpha=args.alpha,
        auto_alpha=args.auto_alpha,
    )

    # trainer
    trainer = RLTrainer(policy, env, args, test_env=test_env)
    if args.evaluate:
        trainer.evaluate_policy_continuously()
    else:
        trainer()
