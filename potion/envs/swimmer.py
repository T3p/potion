"""Swimmer environment compatible with MuJoCo 3 and Gymnasium 0.29."""

from pathlib import Path

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.mujoco.swimmer_v4 import SwimmerEnv as GymnasiumSwimmerEnv
from gymnasium.spaces import Box


_MODEL_PATH = Path(__file__).with_name("assets") / "swimmer.xml"


class SwimmerEnv(GymnasiumSwimmerEnv):
    """Gymnasium's Swimmer-v4 using an MJCF model accepted by MuJoCo 3."""

    def __init__(
        self,
        forward_reward_weight=1.0,
        ctrl_cost_weight=1e-4,
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            forward_reward_weight,
            ctrl_cost_weight,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        observation_size = 8 if exclude_current_positions_from_observation else 10
        observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_size,),
            dtype=np.float64,
        )
        MujocoEnv.__init__(
            self,
            str(_MODEL_PATH),
            4,
            observation_space=observation_space,
            **kwargs,
        )
