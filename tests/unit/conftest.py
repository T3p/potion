import pytest
import numpy as np
import gymnasium as gym
from potion.policies import Policy, ParametricStochasticPolicy


@pytest.fixture
def n_traj():
    return 7


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
def n_params():
    return 6


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
def policy(state_d, action_d, n_params):
    class MockStochasticPolicy:
        state_dim = state_d
        action_dim = action_d

        @property
        def parameters(self):
            return np.ones(self.num_params)

        @property
        def num_params(self):
            return n_params

        def set_params(self, params):
            pass

        def act(self, state, rng, t=None):
            return rng.normal(size=action_d)

        def score(self, state, action, t=None):
            return np.concatenate((state,
                                   action,
                                   np.ones(shape=state.shape[:-1] + (1, ))),
                                  -1)

    return MockStochasticPolicy()


@pytest.fixture
def policy_1d():
    class MockPolicy():
        def act(self, state, rng, t=None):
            return rng.normal(size=1)

    return MockPolicy()


@pytest.fixture
def batch(n_traj, max_trajectory_len, state_d, action_d, discount, horizon, rng):
    states = rng.normal(size=(n_traj, max_trajectory_len, state_d))
    actions = rng.normal(size=(n_traj, max_trajectory_len, action_d))
    rewards = rng.normal(size=(n_traj, max_trajectory_len))
    alive = np.full(shape=(n_traj, max_trajectory_len), fill_value=True)
    alive[:, horizon:] = False

    batch = []
    for i in range(n_traj):
        batch.append((states[i], actions[i], rewards[i], alive[i]))

    return batch
