from abc import ABC, abstractmethod
from gymnasium.spaces import Box


class Policy(ABC):
    def __init__(self, state_dim, action_dim):
        self._state_dim = state_dim
        self._action_dim = action_dim

    @property
    def state_dim(self):
        return self._state_dim

    @property
    def action_dim(self):
        return self._action_dim

    @abstractmethod
    def act(self, state, rng, t=None):
        pass

    @classmethod
    def make(cls, env, **kwargs):
        if not (isinstance(env.observation_space, Box) and len(env.observation_space.shape) == 1
                and isinstance(env.action_space, Box) and len(env.action_space.shape) == 1):
            raise ValueError("Environment must have 1-dimensional Boxes as state space and action space. "
                             "Consider wrapping.")
        return cls(state_dim=env.observation_space.shape[0],
                   action_dim=env.action_space.shape[0],
                   **kwargs)


class StochasticPolicy(Policy):
    @abstractmethod
    def log_pdf(self, s, a, t=None):
        pass

    @abstractmethod
    def entropy(self, s, t=None):
        pass


class ParametricStochasticPolicy(StochasticPolicy):
    @property
    @abstractmethod
    def parameters(self):
        pass

    @property
    @abstractmethod
    def num_params(self):
        pass

    @abstractmethod
    def set_params(self, params):
        pass

    @abstractmethod
    def score(self, s, a, t=None):
        pass

    @abstractmethod
    def entropy_grad(self, s, t=None):
        pass

