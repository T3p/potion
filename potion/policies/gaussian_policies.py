import numpy as np
from abc import abstractmethod
import torch
from torch import nn
from torch.func import functional_call, grad, vmap
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from potion.policies import ParametricStochasticPolicy


_SQUASH_QUADRATURE_NODES, _SQUASH_QUADRATURE_WEIGHTS = (
    np.polynomial.hermite.hermgauss(32)
)
_SQUASH_QUADRATURE_NOISE = np.sqrt(2.) * _SQUASH_QUADRATURE_NODES
_SQUASH_QUADRATURE_WEIGHTS = _SQUASH_QUADRATURE_WEIGHTS / np.sqrt(np.pi)


class GaussianPolicy(ParametricStochasticPolicy):
    def __init__(self, state_dim, action_dim, std_init=None, learn_std=False,
                 squash_actions=False, action_low=None, action_high=None):
        super().__init__(state_dim, action_dim)
        self._learn_std = learn_std
        self._squash_actions = bool(squash_actions)
        self._action_low = None
        self._action_high = None
        self._action_midpoint = None
        self._action_scale = None

        if self._squash_actions:
            self._action_low = self._validate_action_bound(action_low, "action_low")
            self._action_high = self._validate_action_bound(action_high, "action_high")
            if not np.all(self._action_low < self._action_high):
                raise ValueError("action_low must be strictly less than action_high")
            self._action_midpoint = (self._action_low + self._action_high) / 2.
            self._action_scale = (self._action_high - self._action_low) / 2.

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

    @classmethod
    def make(cls, env, **kwargs):
        if kwargs.get("squash_actions", False):
            kwargs.setdefault("action_low", env.action_space.low)
            kwargs.setdefault("action_high", env.action_space.high)
        return super().make(env, **kwargs)

    def _validate_action_bound(self, bound, name):
        if bound is None:
            raise ValueError("{} is required when squash_actions=True".format(name))
        value = np.asarray(bound, dtype=float)
        if value.ndim == 0:
            value = np.full(self.action_dim, value.item())
        if value.shape != (self.action_dim,):
            raise ValueError("{} must have shape ({},)".format(name, self.action_dim))
        if not np.isfinite(value).all():
            raise ValueError("{} must contain only finite values".format(name))
        return value

    def _check_action(self, a):
        if a.shape[-1] != self.action_dim:
            raise ValueError("Bad shape: expected %d-dimensional action(s)" % self.action_dim)

    @property
    def learn_std(self):
        return self._learn_std

    @property
    def squash_actions(self):
        return self._squash_actions

    @property
    def action_low(self):
        return None if self._action_low is None else self._action_low.copy()

    @property
    def action_high(self):
        return None if self._action_high is None else self._action_high.copy()

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
        """Return the latent Gaussian mean, before optional action squashing."""
        self.check_state(s)
        return self._mean(s)

    def _squash(self, latent_action):
        return self._action_midpoint + self._action_scale * np.tanh(latent_action)

    def _unsquash(self, action):
        normalized = (action - self._action_midpoint) / self._action_scale
        tolerance = 10. * np.finfo(float).eps
        if np.any(normalized < -1. - tolerance) or np.any(normalized > 1. + tolerance):
            raise ValueError("Squashed actions must lie within the action bounds")
        open_unit = np.nextafter(1., 0.)
        normalized = np.clip(normalized, -open_unit, open_unit)
        latent_action = np.arctanh(normalized)
        log_sech_squared = 2. * (
            np.log(2.) - latent_action - np.logaddexp(0., -2. * latent_action)
        )
        log_abs_det = np.log(self._action_scale) + log_sech_squared
        return latent_action, log_abs_det

    def act(self, s, rng, t=None):
        self.check_state(s)
        noise = rng.normal(size=self.action_dim)
        latent_action = self.mean(s) + noise * self.std
        if not self.squash_actions:
            return latent_action
        return self._squash(latent_action)

    def act_and_log_prob(self, s, rng, t=None):
        """Sample an action and reuse its mean to compute the log probability."""
        self.check_state(s)
        mean = self.mean(s)
        latent_action = mean + rng.normal(size=self.action_dim) * self.std
        action = (
            self._squash(latent_action)
            if self.squash_actions else latent_action
        )
        if self.squash_actions:
            latent_action, log_abs_det = self._unsquash(action)
        else:
            log_abs_det = 0.
        log_p = (
            -((latent_action - mean) ** 2) / (2 * self.std ** 2)
            - self._std_params
            - 0.5 * np.log(2 * np.pi)
            - log_abs_det
        )
        return action, np.sum(log_p, -1)

    def log_prob(self, s, a, t=None):
        self.check_state(s)
        self._check_action(a)
        self.check_matching(s, a)
        if self.squash_actions:
            latent_action, log_abs_det = self._unsquash(a)
        else:
            latent_action = a
            log_abs_det = 0.
        log_p = (-((latent_action - self.mean(s)) ** 2) / (2 * self.std ** 2)
                 - self._std_params - 0.5 * np.log(2 * np.pi)
                 - log_abs_det)
        return np.sum(log_p, -1)

    def score(self, s, a, t=None):
        self.check_state(s)
        self._check_action(a)
        self.check_matching(s, a)
        latent_action = self._unsquash(a)[0] if self.squash_actions else a
        if self._learn_std:
            return np.concatenate((self._mean_score(s, latent_action),
                                   self._log_std_score(s, latent_action)), axis=-1)
        else:
            return self._mean_score(s, latent_action)

    def entropy(self, s, t=None):
        self.check_state(s)
        ent = self._std_params + 0.5 * (1. + np.log(2 * np.pi)) * np.ones(self.action_dim)
        if self.squash_actions:
            mean = self.mean(s)
            noise = _SQUASH_QUADRATURE_NOISE.reshape(
                (1,) * len(s.shape[:-1]) + (-1, 1)
            )
            weights = _SQUASH_QUADRATURE_WEIGHTS.reshape(
                (1,) * len(s.shape[:-1]) + (-1, 1)
            )
            latent_actions = mean[..., None, :] + self.std * noise
            log_sech_squared = 2. * (
                np.log(2.) - latent_actions - np.logaddexp(0., -2. * latent_actions)
            )
            expected_log_jacobian = np.sum(
                weights * log_sech_squared,
                axis=-2,
            )
            ent = ent + np.log(self._action_scale) + expected_log_jacobian
        return np.sum(ent, -1) * np.ones(s.shape[:-1])

    def entropy_grad(self, s, t=None):
        self.check_state(s)
        if not self.squash_actions:
            mean_score = np.zeros(s.shape[:-1] + (self.num_mean_params,))
            if not self._learn_std:
                return mean_score
            std_score = np.ones(s.shape[:-1] + (self._action_dim,))
            if np.isscalar(self._std_params):
                std_score = np.sum(std_score, -1, keepdims=True)
            return np.concatenate((mean_score, std_score), -1)

        mean = self.mean(s)
        noise = _SQUASH_QUADRATURE_NOISE.reshape(
            (1,) * len(s.shape[:-1]) + (-1, 1)
        )
        weights = _SQUASH_QUADRATURE_WEIGHTS.reshape(
            (1,) * len(s.shape[:-1]) + (-1, 1)
        )
        latent_actions = mean[..., None, :] + self.std * noise
        tanh_actions = np.tanh(latent_actions)
        entropy_mean_derivative = np.sum(weights * -2. * tanh_actions, axis=-2)
        pseudo_action = mean + self.std ** 2 * entropy_mean_derivative
        mean_score = self._mean_score(s, pseudo_action)
        if not self._learn_std:
            return mean_score

        std_score = 1. + np.sum(
            weights * -2. * tanh_actions * self.std * noise,
            axis=-2,
        )
        if np.isscalar(self._std_params):
            std_score = np.sum(std_score, axis=-1, keepdims=True)
        return np.concatenate((mean_score, std_score), axis=-1)

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
    def __init__(self, state_dim, action_dim, mean_params_init=None, std_init=None,
                 learn_std=False, squash_actions=False, action_low=None,
                 action_high=None):

        # Mean
        super().__init__(state_dim, action_dim, std_init, learn_std,
                         squash_actions, action_low, action_high)

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
    def __init__(self, state_dim, action_dim, mean_network=None, std_init=None,
                 learn_std=False, squash_actions=False, action_low=None,
                 action_high=None):
        super().__init__(state_dim, action_dim, std_init, learn_std,
                         squash_actions, action_low, action_high)

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

    def _mean_scores_and_means(self, s, a):
        leading_shape = s.shape[:-1]
        states = torch.as_tensor(s, dtype=torch.float).reshape(-1, self.state_dim)
        actions = torch.as_tensor(a, dtype=torch.float).reshape(-1, self.action_dim)
        std = torch.tensor(self.std, dtype=torch.float, requires_grad=False)
        params = dict(self._mean_network.named_parameters())

        def log_prob(parameters, state, action):
            mean = functional_call(self._mean_network, parameters, (state,))
            value = torch.sum(-((action - mean) ** 2) / (2 * std ** 2))
            return value, mean

        grads, means = vmap(
            grad(log_prob, has_aux=True), in_dims=(None, 0, 0)
        )(
            params, states, actions
        )
        flat_grads = torch.cat(
            [param_grad.reshape(len(states), -1) for param_grad in grads.values()],
            dim=-1,
        )
        mean_scores = flat_grads.reshape(
            leading_shape + (self.num_mean_params,)
        ).detach().numpy()
        means = means.reshape(
            leading_shape + (self.action_dim,)
        ).detach().numpy()
        return mean_scores, means

    def _mean_score(self, s, a):
        return self._mean_scores_and_means(s, a)[0]

    def score(self, s, a, t=None):
        """Return batched scores with a shared forward pass for learned stds."""
        self.check_state(s)
        self._check_action(a)
        self.check_matching(s, a)
        latent_action = self._unsquash(a)[0] if self.squash_actions else a
        mean_scores, means = self._mean_scores_and_means(s, latent_action)
        if not self.learn_std:
            return mean_scores

        actions = np.asarray(latent_action, dtype=mean_scores.dtype)
        std = np.asarray(self.std, dtype=mean_scores.dtype)
        std_scores = ((means - actions) / std) ** 2 - 1.
        if np.isscalar(self._std_params):
            std_scores = np.sum(std_scores, axis=-1, keepdims=True)
        return np.concatenate((mean_scores, std_scores), axis=-1)

    def weighted_score_sum(self, s, a, weights):
        """Sum scalar-weighted log-policy scores without a score Jacobian."""
        self.check_state(s)
        self._check_action(a)
        self.check_matching(s, a)
        weights = np.asarray(weights)
        if weights.shape != s.shape[:-1]:
            raise ValueError("weights should match state and action leading dimensions")

        latent_action = self._unsquash(a)[0] if self.squash_actions else a
        parameters = tuple(self._mean_network.parameters())
        reference = parameters[0]
        states = torch.as_tensor(
            s, dtype=reference.dtype, device=reference.device
        )
        actions = torch.as_tensor(
            latent_action, dtype=reference.dtype, device=reference.device
        )
        coefficients = torch.as_tensor(
            weights, dtype=reference.dtype, device=reference.device
        )
        std = torch.as_tensor(
            self.std, dtype=reference.dtype, device=reference.device
        )

        means = self._mean_network(states)
        mean_log_prob = torch.sum(
            -((actions - means) ** 2) / (2. * std ** 2), dim=-1
        )
        objective = torch.sum(coefficients * mean_log_prob)
        mean_grads = torch.autograd.grad(objective, parameters)
        flat_mean_grad = torch.cat(
            [parameter_grad.reshape(-1) for parameter_grad in mean_grads]
        )
        if not self.learn_std:
            return flat_mean_grad.detach().cpu().numpy()

        with torch.no_grad():
            std_scores = ((means - actions) / std) ** 2 - 1.
            if np.isscalar(self._std_params):
                std_scores = torch.sum(std_scores, dim=-1, keepdim=True)
            reduction_dims = tuple(range(std_scores.ndim - 1))
            std_grad = torch.sum(
                coefficients[..., None] * std_scores,
                dim=reduction_dims,
            )
            gradient = torch.cat((flat_mean_grad, std_grad))
        return gradient.detach().cpu().numpy()

    def weighted_score_samples(self, s, a, weights):
        """Return one scalar-weighted score sum for each trajectory.

        This uses vectorized reverse-mode differentiation over trajectories and
        avoids materializing the time-by-parameter score tensor.
        """
        self.check_state(s)
        self._check_action(a)
        self.check_matching(s, a)
        weights = np.asarray(weights)
        if weights.shape != s.shape[:-1]:
            raise ValueError("weights should match state and action leading dimensions")

        latent_action = self._unsquash(a)[0] if self.squash_actions else a
        parameters = dict(self._mean_network.named_parameters())
        reference = next(self._mean_network.parameters())
        states = torch.as_tensor(
            s, dtype=reference.dtype, device=reference.device
        )
        actions = torch.as_tensor(
            latent_action, dtype=reference.dtype, device=reference.device
        )
        coefficients = torch.as_tensor(
            weights, dtype=reference.dtype, device=reference.device
        )
        std = torch.as_tensor(
            self.std, dtype=reference.dtype, device=reference.device
        )

        def trajectory_log_prob(parameters, states, actions, coefficients):
            means = functional_call(self._mean_network, parameters, (states,))
            log_probs = torch.sum(
                -((actions - means) ** 2) / (2. * std ** 2), dim=-1
            )
            return torch.sum(coefficients * log_probs), means

        grads, means = vmap(
            grad(trajectory_log_prob, has_aux=True),
            in_dims=(None, 0, 0, 0),
        )(parameters, states, actions, coefficients)
        flat_mean_grads = torch.cat(
            [parameter_grad.reshape(len(states), -1)
             for parameter_grad in grads.values()],
            dim=-1,
        )
        if not self.learn_std:
            return flat_mean_grads.detach().cpu().numpy()

        with torch.no_grad():
            std_scores = ((means - actions) / std) ** 2 - 1.
            if np.isscalar(self._std_params):
                std_scores = torch.sum(std_scores, dim=-1, keepdim=True)
            std_grads = torch.sum(
                coefficients[..., None] * std_scores, dim=1
            )
            gradients = torch.cat((flat_mean_grads, std_grads), dim=-1)
        return gradients.detach().cpu().numpy()
