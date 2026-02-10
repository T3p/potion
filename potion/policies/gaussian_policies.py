import numpy as np
from abc import abstractmethod
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from potion.policies import ParametricStochasticPolicy


class GaussianPolicy(ParametricStochasticPolicy):
    def __init__(self, state_dim, action_dim, std_init=None, learn_std=False):
        super().__init__(state_dim, action_dim)
        self._learn_std = learn_std

        # Log of standard deviation
        if std_init is not None:
            if not np.isscalar(std_init) and (std_init.ndim != 1 or len(std_init) != self._action_dim):
                raise ValueError("Bad shape: std_init should be a scalar or a 1d array of size action_dim")
            if self._action_dim == 1 and not np.isscalar(std_init):
                raise ValueError("Scalar std should not be an array")
            if not np.all(std_init > 0.):
                raise ValueError("Std should be positive")
            self._std_params = np.log(std_init)
        else:
            self._std_params = 0.

        if not learn_std:
            self._n_std_params = 0
        elif np.isscalar(self._std_params):
            self._n_std_params = 1
        else:
            self._n_std_params = len(self._std_params)

    def _check_action(self, a):
        if a.shape[-1] != self.action_dim:
            raise ValueError("Bad shape: expected %d-dimensional action(s)" % self.action_dim)

    @property
    def learn_std(self):
        return self._learn_std

    @property
    def parameters(self):
        if not self._learn_std:
            return self._flat_mean_params
        else:
            return np.concatenate((self._flat_mean_params, np.atleast_1d(self._std_params)))

    @property
    def num_mean_params(self):
        return self.num_params - self._n_std_params

    @property
    def num_std_params(self):
        return self._n_std_params

    @property
    def std(self):
        if np.isscalar(self._std_params) and self.action_dim > 1:
            return np.exp(self._std_params) * np.ones(self.action_dim)
        else:
            return np.exp(self._std_params)

    def set_params(self, params):
        if np.isscalar(params):  # Broadcast
            self._set_mean_params(params)
            if self.learn_std:
                self._std_params = self._std_params * 0. + params
            return

        if params.ndim > 1:
            if params.ndim != 2:
                raise ValueError("Bad shape, params should be 1d (flat) or 2d (matrix, only for LinearGaussianPolicy)")
            params = np.ravel(params)
        if len(params) > self.num_params:
            raise ValueError("Too many params")
        if len(params) < self.num_params:
            raise ValueError("Too few params")
        if not self._learn_std:
            self._set_mean_params(params)
        else:
            mean_params = params[:-self._n_std_params]
            self._set_mean_params(mean_params)
            self._std_params = params[-self._n_std_params:] if self.num_std_params > 1 else params[-1]

    def set_std(self, std):
        if self.learn_std:
            raise RuntimeError("Cannot set directly a learnable std")
        if not np.all(std > 0):
            raise ValueError("Std should be positive")
        if np.isscalar(std) and not np.isscalar(self._std_params):
            self._std_params = std + np.zeros_like(self._std_params)
        if not np.isscalar(std) and std.shape != (self.action_dim,):
            raise ValueError("Bad shape, std should be a scalar or a 1d array of size action_dim")
        if self.action_dim == 1 and not np.isscalar(std):
            raise ValueError("Scalar std should not be an array")
        self._std_params = np.log(std)

    def mean(self, s):
        self.check_state(s)
        return self._mean(s)

    def act(self, s, rng, t=None):
        self.check_state(s)
        noise = rng.normal(size=self.action_dim)
        return self.mean(s) + noise * self.std

    def log_prob(self, s, a, t=None):
        self.check_state(s)
        self._check_action(a)
        self.check_matching(s, a)
        log_p = -((a - self.mean(s)) ** 2) / (2 * self.std ** 2) - self._std_params - 0.5 * np.log(2 * np.pi)
        return np.sum(log_p, -1)

    def score(self, s, a, t=None):
        self.check_state(s)
        self._check_action(a)
        self.check_matching(s, a)
        if self._learn_std:
            return np.concatenate((self._mean_score(s, a), self._log_std_score(s, a)), axis=-1)
        else:
            return self._mean_score(s, a)

    def entropy(self, s, t=None):
        self.check_state(s)
        ent = self._std_params + 0.5 * (1. + np.log(2 * np.pi)) * np.ones(self.action_dim)
        return np.sum(ent, -1) * np.ones(s.shape[:-1])

    def entropy_grad(self, s, t=None):
        self.check_state(s)
        if self._learn_std:
            std_score = np.ones(s.shape[:-1] + (self._action_dim,))
            if np.isscalar(self._std_params):
                std_score = np.sum(std_score, -1, keepdims=True)
            return np.concatenate((np.zeros(s.shape[:-1] + (self._state_dim * self._action_dim,)),
                                   std_score), -1)
        else:
            return np.zeros(s.shape[:-1] + (self._state_dim * self._action_dim,))

    @property
    @abstractmethod
    def _flat_mean_params(self):  # pragma: no cover
        pass

    @abstractmethod
    def _set_mean_params(self, params):  # pragma: no cover
        pass

    @abstractmethod
    def _mean(self, s):  # pragma: no cover
        pass

    @abstractmethod
    def _mean_score(self, s, a):  # pragma: no cover
        pass

    def _log_std_score(self, s, a):
        score = (self.mean(s) - a) ** 2 * np.exp(-2 * self._std_params) - 1.
        if np.isscalar(self._std_params):
            return np.sum(score, -1, keepdims=True)
        return score


class LinearGaussianPolicy(GaussianPolicy):
    def __init__(self, state_dim, action_dim, mean_params_init=None, std_init=None, learn_std=False):

        # Mean
        super().__init__(state_dim, action_dim, std_init, learn_std)

        if mean_params_init is not None:
            if np.isscalar(mean_params_init):
                self._mean_params = mean_params_init + np.zeros((self.action_dim, self.state_dim))
            elif mean_params_init.ndim == 1 and len(mean_params_init) == self.action_dim * self.state_dim:
                self._mean_params = mean_params_init.reshape((self.action_dim, self.state_dim))
            else:
                if mean_params_init.shape != (self.action_dim, self.state_dim):
                    raise ValueError("Bad shape: mean_init should be a scalar, "
                                     "a 1d array of size action_dim * state_dim, "
                                     "or a 2d array of size state_dim times action_dim")
                self._mean_params = mean_params_init
        else:
            self._mean_params = np.zeros((self._action_dim, self._state_dim))

    @property
    def _flat_mean_params(self):
        return np.ravel(self._mean_params)

    def _set_mean_params(self, params):
        if np.isscalar(params):  # Broadcast
            self._mean_params = self._mean_params * 0 + params
            return
        self._mean_params = params.reshape((self._action_dim, self._state_dim))

    def _mean(self, s):
        return s @ self._mean_params.T

    def _mean_score(self, s, a):
        score = np.einsum('...k,...h->...hk', s, (a - self._mean(s)) / self.std ** 2)
        score = score.reshape(score.shape[:-2] + (score.shape[-2] * score.shape[-1],))
        return score


class DeepGaussianPolicy(GaussianPolicy):
    def __init__(self, state_dim, action_dim, mean_network=None, std_init=None, learn_std=False):
        super().__init__(state_dim, action_dim, std_init, learn_std)

        if mean_network is None:
            self._mean_network = nn.Linear(self._state_dim, self._action_dim, bias=False)
            self._mean_network.weight.data.fill_(0.)
        else:
            try:
                with torch.no_grad():
                    s = torch.ones(self.state_dim, dtype=torch.float)
                    a = mean_network(s).numpy()
                    if not a.shape == (self.action_dim,):
                        raise ValueError("Network output should match action dimension")
            except Exception as e:
                raise ValueError("Network could not process state, likely bad input shape") from e
            self._mean_network = mean_network

    @property
    def _flat_mean_params(self):
        return parameters_to_vector(self._mean_network.parameters()).detach().numpy()

    def _set_mean_params(self, params):
        if np.isscalar(params):  # Broadcast
            params = params * torch.ones(size=(self.num_params - self.num_std_params,), dtype=torch.float,
                                         requires_grad=False)
        else:
            params = torch.tensor(params, dtype=torch.float, requires_grad=False)

        vector_to_parameters(params, self._mean_network.parameters())

    def _mean(self, s, requires_grad=False):
        if not torch.is_tensor(s):
            s = torch.tensor(s, dtype=torch.float, requires_grad=False)
        if requires_grad:
            return self._mean_network(s)
        else:
            with torch.no_grad():
                return self._mean_network(s).numpy()

    def _mean_score(self, s, a):
        s = torch.tensor(s, dtype=torch.float, requires_grad=False)
        a = torch.tensor(a, dtype=torch.float, requires_grad=False)
        std = torch.tensor(self.std, dtype=torch.float, requires_grad=False)
        val = torch.sum(-((a - self._mean(s, requires_grad=True)) ** 2) / (2 * std ** 2), -1)
        grad = torch.autograd.grad(val, self._mean_network.parameters())
        return parameters_to_vector(grad).numpy()
