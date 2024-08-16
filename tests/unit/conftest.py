import pytest
import numpy as np
import gymnasium as gym


@pytest.fixture
def max_trajectory_len():
    return 5


@pytest.fixture
def seed():
    return 42


@pytest.fixture
def rng(seed):
    return np.random.default_rng(seed)


@pytest.fixture
def state_d():
    return 3


@pytest.fixture
def action_d():
    return 2


@pytest.fixture
def discount():
    return 0.99


@pytest.fixture
def horizon(max_trajectory_len):
    return 3 * max_trajectory_len // 4


@pytest.fixture
def n_jobs():
    return 2


@pytest.fixture
def n_episodes():
    return 2


@pytest.fixture
def env(state_d, action_d, max_trajectory_len, horizon):
    class MockEnv(gym.Env):
        rng = None
        observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(state_d,), dtype=float)
        action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(action_d,), dtype=float)
        t = 0

        def reset(self, seed=None, options=None):
            self.rng = np.random.default_rng(seed)
            self.t = 0
            return self.rng.normal(size=state_d), dict()

        def step(self, action):
            next_state = self.rng.normal(size=state_d)
            reward = -1
            terminated = False
            info = dict()
            self.t += 1
            truncated = (self.t >= horizon)
            return next_state, reward, terminated, truncated, info

    return MockEnv()


@pytest.fixture
def env_1d(max_trajectory_len, horizon):
    class MockEnv(gym.Env):
        rng = None
        observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, dtype=float)
        action_space = gym.spaces.Box(low=-np.inf, high=np.inf, dtype=float)
        t = 0

        def reset(self, seed=None, options=None):
            self.rng = np.random.default_rng(seed)
            self.t = 0
            return self.rng.normal(), dict()

        def step(self, action):
            next_state = self.rng.normal()
            reward = -1
            terminated = False
            info = dict()
            self.t += 1
            truncated = (self.t >= horizon)
            return next_state, reward, terminated, truncated, info

    return MockEnv()


@pytest.fixture
def env_stochastic_reward(state_d, action_d, max_trajectory_len, horizon):
    class MockEnv(gym.Env):
        rng = None
        observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(state_d,), dtype=float)
        action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(action_d,), dtype=float)
        t = 0

        def reset(self, seed=None, options=None):
            self.rng = np.random.default_rng(seed)
            self.t = 0
            return np.zeros(state_d), dict()

        def step(self, action):
            next_state = np.zeros(state_d)
            reward = self.rng.uniform(0., 1.)
            terminated = False
            info = dict()
            self.t += 1
            truncated = (self.t >= horizon)
            return next_state, reward, terminated, truncated, info

    return MockEnv()


@pytest.fixture
def policy(action_d):
    class MockPolicy:
        def act(self, state, rng):
            return rng.normal(size=action_d)

    return MockPolicy()


@pytest.fixture
def policy_1d():
    class MockPolicy:
        def act(self, state, rng):
            return rng.normal(size=1)

    return MockPolicy()