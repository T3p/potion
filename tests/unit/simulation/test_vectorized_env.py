import numpy as np
import pytest
import gymnasium as gym

from potion.simulation.vectorized_env import VectorizedBatchEnv
from potion.simulation.trajectory_generators import generate_batch, unpack


class CountingEnv(gym.Env):
    observation_space = gym.spaces.Box(-1., 1., shape=(2,), dtype=np.float32)
    action_space = gym.spaces.Box(-1., 1., shape=(1,), dtype=np.float32)

    def __init__(self, closed):
        self.closed = closed

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        return np.ones(2, dtype=np.float32), 1., False, False, {}

    def close(self):
        self.closed.append(self)


def test_vectorized_batch_env_keeps_scalar_and_vector_apis():
    closed = []
    env = VectorizedBatchEnv(lambda: CountingEnv(closed), num_envs=3)

    scalar_state, _ = env.reset(seed=7)
    vector_states, _ = env.vector_env.reset(seed=[7, 8, 9])

    assert scalar_state.shape == (2,)
    assert vector_states.shape == (3, 2)
    assert env.num_envs == 3
    assert env.observation_space == CountingEnv.observation_space
    assert env.action_space == CountingEnv.action_space

    env.close()
    env.close()
    assert len(closed) == 4


@pytest.mark.parametrize("num_envs", [0, -1])
def test_vectorized_batch_env_rejects_nonpositive_size(num_envs):
    with pytest.raises(ValueError, match="positive"):
        VectorizedBatchEnv(lambda: CountingEnv([]), num_envs)


def test_vectorized_batch_env_requires_factory():
    with pytest.raises(TypeError, match="callable"):
        VectorizedBatchEnv(CountingEnv([]), 2)


class SeededFiniteEnv(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)
    action_space = gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        if seed is not None:
            self.limit = 1 + seed % 4
        return self.np_random.normal(size=2).astype(np.float32), {}

    def step(self, action):
        self.t += 1
        state = self.np_random.normal(size=2).astype(np.float32)
        reward = float(action[0] + self.t)
        return state, reward, False, self.t >= self.limit, {}


class BatchAwarePolicy:
    def __init__(self):
        self.batch_calls = 0

    def act_and_log_prob(self, state, rng, t=None):
        action = rng.normal(size=1) + np.sum(state)
        return action, -np.sum(action ** 2)

    def act_batch_and_log_prob(self, states, rngs, t=None):
        self.batch_calls += 1
        samples = [
            self.act_and_log_prob(state, rng, t)
            for state, rng in zip(states, rngs)
        ]
        actions, logps = zip(*samples)
        return np.asarray(actions), np.asarray(logps)


@pytest.mark.parametrize("continual", [False, True])
def test_vectorized_collection_matches_scalar_collection(continual):
    n_episodes = 5
    horizon = None if continual else 6
    discount = 0.75 if continual else 1.
    scalar_policy = BatchAwarePolicy()
    vector_policy = BatchAwarePolicy()
    scalar_batch = generate_batch(
        SeededFiniteEnv(), scalar_policy, n_episodes, horizon,
        np.random.default_rng(123), discount=discount,
    )
    vector_env = VectorizedBatchEnv(SeededFiniteEnv, num_envs=3)
    try:
        vector_batch = generate_batch(
            vector_env, vector_policy, n_episodes, horizon,
            np.random.default_rng(123), discount=discount,
        )
    finally:
        vector_env.close()

    assert scalar_policy.batch_calls == 0
    assert vector_policy.batch_calls > 0
    for scalar_values, vector_values in zip(
            unpack(scalar_batch), unpack(vector_batch)):
        assert np.allclose(scalar_values, vector_values)


def test_vectorized_collection_rejects_joblib_parallelism():
    env = VectorizedBatchEnv(SeededFiniteEnv, num_envs=2)
    try:
        with pytest.raises(ValueError, match="cannot be combined"):
            generate_batch(
                env, BatchAwarePolicy(), 2, 3, np.random.default_rng(1),
                parallel=True,
            )
    finally:
        env.close()
