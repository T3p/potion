from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from potion.simulation.action_wrappers import TanhActionWrapper
from potion.simulation.trajectory_generators import generate_trajectory


class TrackingBoxEnv(gym.Env):
    observation_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(2,), dtype=np.float32
    )
    action_space = gym.spaces.Box(
        low=np.array([-2.0, 1.0], dtype=np.float32),
        high=np.array([4.0, 5.0], dtype=np.float32),
        dtype=np.float32,
    )

    def __init__(self):
        self.actions = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return np.array([0.25, -0.5], dtype=np.float32), {}

    def step(self, action):
        self.actions.append(action.copy())
        return np.zeros(2, dtype=np.float32), 1.0, True, False, {}


def test_tanh_action_wrapper_exposes_latent_space_and_squashes_controls():
    base_env = TrackingBoxEnv()
    env = TanhActionWrapper(base_env)
    env.reset(seed=3)

    latent_action = np.array([0.0, np.arctanh(0.5)])
    env.step(latent_action)

    assert env.action_space.dtype == np.float64
    assert np.all(np.isneginf(env.action_space.low))
    assert np.all(np.isposinf(env.action_space.high))
    assert base_env.actions[0].dtype == np.float32
    assert np.allclose(base_env.actions[0], np.array([1.0, 4.0]))


def test_tanh_action_wrapper_penalizes_nonfinite_action_without_stepping():
    base_env = TrackingBoxEnv()
    env = TanhActionWrapper(base_env, invalid_action_penalty=-123.0)
    observation, _ = env.reset(seed=3)

    transition = env.step(np.array([np.nan, 0.0]))

    assert base_env.actions == []
    assert np.array_equal(transition[0], observation)
    assert transition[1:] == (-123.0, True, False, {"invalid_action": True})


def test_trajectory_keeps_latent_action_while_env_receives_bounded_control():
    base_env = TrackingBoxEnv()
    env = TanhActionWrapper(base_env)
    latent_action = np.array([20.0, -20.0])
    policy = SimpleNamespace(
        act=lambda state, rng, t: latent_action.copy(),
        log_prob=lambda state, action, t: -0.5 * np.sum(action**2),
    )

    _, actions, _, alive, _ = generate_trajectory(env, policy, 1, seed=7)

    assert actions.dtype == np.float64
    assert np.array_equal(actions[alive][0], latent_action)
    assert np.all(base_env.actions[0] >= base_env.action_space.low)
    assert np.all(base_env.actions[0] <= base_env.action_space.high)


def test_tanh_action_wrapper_rejects_unsupported_spaces_and_penalties():
    class DiscreteEnv(gym.Env):
        observation_space = gym.spaces.Box(-1.0, 1.0, shape=(1,))
        action_space = gym.spaces.Discrete(2)

    discrete_env = DiscreteEnv()
    with pytest.raises(TypeError, match="Box action space"):
        TanhActionWrapper(discrete_env)

    with pytest.raises(ValueError, match="must be finite"):
        TanhActionWrapper(TrackingBoxEnv(), invalid_action_penalty=np.nan)
