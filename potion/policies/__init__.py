from abc import ABC, abstractmethod
from gymnasium.spaces import Box, Discrete


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
    def act(self, state, rng, t=None):  # pragma: no cover
        pass

    @classmethod
    def make(cls, env, **kwargs):
        if not (isinstance(env.observation_space, Box) and len(env.observation_space.shape) == 1
                and (isinstance(env.action_space, Box) and len(env.action_space.shape) == 1)
                or isinstance(env.action_space, Discrete)):
            raise ValueError("Environment must have 1-dimensional Box as state space, "
                             "and either a 1-dimensional Box or a Discrete as action space. "
                             "Consider wrapping.")
        return cls(state_dim=env.observation_space.shape[0],
                   action_dim=env.action_space.shape[0],
                   **kwargs)

    def check_state(self, s):
        if s.shape[-1] != self.state_dim:
            raise ValueError("Bad shape: expected %d-dimensional state(s)" % self.state_dim)

    @abstractmethod
    def _check_action(self, a):  # pragma: no cover
        pass

    @staticmethod
    def check_matching(s, a):
        if not s.shape[:-1] == a.shape[:-1]:
            raise ValueError("Bad shape: all state and action dimensions should match except the last")


class StochasticPolicy(Policy):
    @abstractmethod
    def log_prob(self, s, a, t=None):  # pragma: no cover
        pass

    @abstractmethod
    def entropy(self, s, t=None):  # pragma: no cover
        pass


class ParametricStochasticPolicy(StochasticPolicy):
    @property
    @abstractmethod
    def parameters(self):  # pragma: no cover
        pass

    @property
    def num_params(self):
        return len(self.parameters)

    @abstractmethod
    def set_params(self, params):  # pragma: no cover
        pass

    @abstractmethod
    def score(self, s, a, t=None):  # pragma: no cover
        pass

    @abstractmethod
    def entropy_grad(self, s, t=None):  # pragma: no cover
        pass
