# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
from tf2rl.algos.sac import SAC

from examples.config import get_config
from safety_rl.envs.engine_wrapper import EngineWrapper
from safety_rl.envs.rl_env_wrapper import RLEnvWrapper
from safety_rl.experiments.rl_trainer import RLTrainer


class FixedObsRLEnv(RLEnvWrapper):
    def __init__(self, *args, default_goal_pos=np.array([1.5, 0.0], dtype=np.float32), curve_reference=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._default_goal_pos = default_goal_pos
        self._curve_reference = curve_reference

    def reset(self):
        self._env.reset()
        if self._curve_reference:
            self.set_goal_pos(np.zeros(shape=(2,), dtype=np.float32))
            rad = np.random.uniform(-np.pi, np.pi)
            xs = np.linspace(0.0, 2 * np.pi, 30)
            ys = np.sin(xs)
            xs /= np.pi
            ys /= np.pi / 2.0
            reference_path = np.array([np.cos(rad) * xs + np.sin(rad) * ys, -np.sin(rad) * xs + np.cos(rad) * ys]).T
            reference_path = np.flip(reference_path)
            self.set_robot_pos(reference_path[0])
        else:
            self.set_goal_pos(self._default_goal_pos)
            reference_path = np.linspace(self.robot_pos[:2], self.goal_pos[:2], num=self._n_way_points)
        self.set_reference_path_world_coord(reference_path)
        return self._get_obs()

    def step(self, action):
        next_obs, rew, done, info = super().step(action)
        done = done or info["goal_met"]
        return next_obs, rew, done, info


if __name__ == "__main__":
    parser = RLTrainer.get_argument()
    parser = SAC.get_argument(parser)
    parser.set_defaults(batch_size=100)
    parser.set_defaults(n_warmup=10000)
    parser.add_argument("--curve-reference", action="store_true")
    parser.add_argument("--update-reference", action="store_true")
    parser.add_argument("--robot-type", default="point", choices=["point", "car", "doggo"])
    args = parser.parse_args()

    config, _ = get_config(robot_type=args.robot_type, field_size=2.0, no_obs=True)
    env = FixedObsRLEnv(EngineWrapper(config=config, visualize_waypoints=True), curve_reference=args.curve_reference)
    test_env = FixedObsRLEnv(
        EngineWrapper(config=config, visualize_waypoints=True), curve_reference=args.curve_reference
    )

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

    trainer = RLTrainer(policy, env, args, test_env=test_env)
    if args.evaluate:
        trainer.evaluate_policy(0)
    else:
        trainer()
