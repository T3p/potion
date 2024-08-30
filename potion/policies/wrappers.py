from potion.policies import ParametricStochasticPolicy
import numpy as np


class Staged(ParametricStochasticPolicy):
    def __init__(self, base_policy, horizon):
        if not isinstance(base_policy, ParametricStochasticPolicy):
            raise ValueError("Can only wrap a ParametricPolicy")
        super().__init__(base_policy.state_dim, base_policy.action_dim)
        self.base_policy = base_policy
        self.horizon = horizon

        self._params = np.tile(base_policy.parameters, (self.horizon, 1))

    @property
    def num_params(self):
        return self.base_policy.num_params * self.horizon

    @property
    def parameters(self):
        return np.ravel(self._params)

    def parameters_at(self, t):
        self._check_index(t)
        return self._params[t]

    def set_params(self, params):
        self._params = np.reshape(params, (self.horizon, self.base_policy.num_params))

    def set_params_at(self, params, t):
        self._check_index(t)
        if not params.shape == self._params[t].shape:
            raise ValueError("Bad shape for base-policy parameters")
        self._params[t] = params

    def act(self, state, rng, t=None):
        self._select(t)
        return self.base_policy.act(state, rng, t)

    def score(self, s, a, t=None):
        if t is None and s.ndim > 1:
            scores = []
            for h in range(s.shape[-2]):
                self._select(h)
                scores.append(self.base_policy.score(s[:, h, :], a[:, h, :], h))
            return np.stack(scores, axis=1)

        self._select(t)
        return self.base_policy.score(s, a, t)

    def entropy_grad(self, s, t=None):
        self._select(t)
        return self.base_policy.entropy_grad(s, t)

    def log_pdf(self, s, a, t=None):
        self._select(t)
        return self.base_policy.log_pdf(s, a, t)

    def entropy(self, s, t=None):
        self._select(t)
        return self.base_policy.entropy(s, t)

    def _check_index(self, t):
        if t is None or not isinstance(t, int) or t < 0 or t >= self.horizon:
            raise ValueError("t must be a valid index from 0 to horizon-1")

    def _select(self, t):
        self._check_index(t)
        self.base_policy.set_params(self._params[t])
