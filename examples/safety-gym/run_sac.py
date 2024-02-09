# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import gym
import safety_gym
from tf2rl.algos.sac import SAC
from tf2rl.experiments.trainer import Trainer

if __name__ == "__main__":
    parser = Trainer.get_argument()
    parser = SAC.get_argument(parser)
    parser.set_defaults(batch_size=100)
    parser.set_defaults(n_warmup=10000)
    args = parser.parse_args()

    env = gym.make("Safexp-PointGoal1-v0")
    test_env = gym.make("Safexp-PointGoal1-v0")

    policy = SAC(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.high.size,
        gpu=args.gpu,
        memory_capacity=args.memory_capacity,
        max_action=env.action_space.high[0],
        batch_size=args.batch_size,
        n_warmup=args.n_warmup,
        alpha=args.alpha,
        auto_alpha=args.auto_alpha,
    )
    trainer = Trainer(policy, env, args, test_env=test_env)
    if args.evaluate:
        trainer.evaluate_policy(0)
    else:
        trainer()
