# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from safety_rl.envs.engine_wrapper import EngineWrapper
from safety_rl.envs.goselo_env import GoseloEnv


class RLInterfaceEnv(GoseloEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Overwrite observation space by calling Engine.build_observation_space
        self.build_observation_space()

    def _get_obs(self):
        return EngineWrapper._get_obs(self)

    def get_goselo_img(self):
        return GoseloEnv._get_obs(self)
