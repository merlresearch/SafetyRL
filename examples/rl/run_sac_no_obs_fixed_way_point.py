# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
from tf2rl.algos.sac import SAC

from examples.config import get_config
from safety_rl.envs.engine_wrapper import EngineWrapper
from safety_rl.envs.rl_env_wrapper import RLEnvWrapper
from safety_rl.experiments.rl_trainer import RLTrainer
from safety_rl.path_planner.utils import find_nearest_point


class NoObsRLEnvWrapper(RLEnvWrapper):
    def __init__(self, *args, threshold_min_dist_to_reference_path=0.1, **kwargs):
        self._threshold_min_dist_to_reference_path = threshold_min_dist_to_reference_path
        super().__init__(*args, **kwargs)

    def reset(self):
        obs = super().reset()
        self._update_reference_path()
        return obs

    def _update_reference_path(self):
        reference_path = np.linspace(self.robot_pos[:2], self.goal_pos[:2], num=self._n_way_points)
        self.set_reference_path_world_coord(reference_path)

    def _is_update_reference(self):
        min_idx, min_dist = find_nearest_point(self.robot_pos[:2], self._reference_path_world_coord)
        if min_idx > int(self._reference_path_world_coord.shape[0] / 2.0):
            return True
        if min_dist > self._threshold_min_dist_to_reference_path:
            return True
        return False

    def step(self, action):
        next_obs, rew, done, info = super().step(action)
        if self._is_update_reference():
            self._update_reference_path()
        done = done or info["goal_met"]
        return next_obs, rew, done, info


if __name__ == "__main__":
    parser = RLTrainer.get_argument()
    parser = SAC.get_argument(parser)
    parser.set_defaults(batch_size=100)
    parser.set_defaults(n_warmup=10000)
    parser.set_defaults(episode_max_steps=300)
    parser.set_defaults(max_steps=int(5e6))
    parser.add_argument("--curve-reference", action="store_true")
    parser.add_argument("--robot-type", default="point", choices=["point", "car", "doggo"])
    args = parser.parse_args()

    config, _ = get_config(robot_type=args.robot_type, field_size=2.0, no_obs=True)
    env = NoObsRLEnvWrapper(EngineWrapper(config=config, visualize_waypoints=True))
    test_env = NoObsRLEnvWrapper(EngineWrapper(config=config, visualize_waypoints=True))

    policy = SAC(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.high.size,
        gpu=args.gpu,
        actor_units=[64, 64],
        critic_units=[64, 64],
        memory_capacity=args.memory_capacity,
        max_action=env.action_space.high[0],
        batch_size=args.batch_size,
        n_warmup=args.n_warmup,
        alpha=args.alpha,
        auto_alpha=args.auto_alpha,
    )

    trainer = RLTrainer(policy, env, args, test_env=test_env)
    if args.evaluate:
        trainer.evaluate_policy_continuously()
    else:
        trainer()
