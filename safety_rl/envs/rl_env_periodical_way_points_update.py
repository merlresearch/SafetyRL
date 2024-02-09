# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from safety_rl.envs.rl_env_way_points_generator_wrapper import RLEnvWayPointsGeneratorWrapper


class RLEnvWayPointsGeneratorPeriodicalUpdateWrapper(RLEnvWayPointsGeneratorWrapper):
    def __init__(self, *args, update_interval=2, **kwargs):
        self._update_interval = update_interval
        self._n_steps = 0
        super().__init__(*args, **kwargs)

    def reset(self):
        self._n_steps = 0
        return super().reset()

    def step(self, action):
        self._n_steps += 1
        return super().step(action)

    def _is_update_reference(self):
        if self._n_steps >= self._update_interval:
            self._n_steps = 0
            return True
        else:
            return False


class RLEnvWayPointsGeneratorPeriodicalUpdateEvalWrapper(RLEnvWayPointsGeneratorPeriodicalUpdateWrapper):
    def _compute_reward(self):
        return self._baseline_reward()
