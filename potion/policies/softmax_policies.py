import numpy as np
from abc import abstractmethod
import torch
from torch import nn
from torch.func import functional_call, grad, vmap
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from potion.policies import ParametricStochasticPolicy
from scipy.special import softmax, logsumexp


class SoftmaxPolicy(ParametricStochasticPolicy):
    def __init__(self, state_dim, num_actions, temperature):
        super().__init__(state_dim, 1)
        if temperature < 0:
            raise ValueError("Temperature must be positive")
        self._num_actions = num_actions
        self._temp = temperature

    def _check_action(self, a):
        a = np.asarray(a)
        if not np.issubdtype(a.dtype, np.integer) or np.any(a < 0) or np.any(a >= self.num_actions):
            raise ValueError("Illegal action(s): expected index between 0 and %d"
                             % (self.num_actions - 1))
        return a

    @property
    def num_actions(self):
        return self._num_actions

    @property
    def temperature(self):
        return self._temp

    def set_temperature(self, temperature):
        if temperature < 0:
            raise ValueError("Temperature must be positive")
        self._temp = temperature

    def logits(self, s):
        self.check_state(s)
        return self._logits(s)

    def act(self, s, rng, t=None):
        self.check_state(s)
        return int(rng.choice(a=self._num_actions, p=self._probs(s)))

    def act_and_log_prob(self, s, rng, t=None):
        """Sample an action and reuse its logits to compute the log probability."""
        self.check_state(s)
        scaled_logits = self._logits(s) / self._temp
        probabilities = softmax(scaled_logits, axis=-1)
        action = int(rng.choice(a=self._num_actions, p=probabilities))
        log_prob = scaled_logits[action] - logsumexp(scaled_logits, axis=-1)
        return action, log_prob

    def act_batch_and_log_prob(self, states, rngs, t=None):
        """Sample independent actions using one batched logits evaluation."""
        states = np.asarray(states)
        rngs = tuple(rngs)
        self.check_state(states)
        if states.ndim != 2 or len(states) != len(rngs):
            raise ValueError("states and rngs should contain equally many episodes")

        scaled_logits = self._logits(states) / self._temp
        probabilities = softmax(scaled_logits, axis=-1)
        actions = np.asarray([
            rng.choice(a=self._num_actions, p=probability)
            for rng, probability in zip(rngs, probabilities)
        ], dtype=int)
        selected_logits = np.take_along_axis(
            scaled_logits, actions[:, None], axis=-1
        )[:, 0]
        log_probs = selected_logits - logsumexp(scaled_logits, axis=-1)
        return actions, log_probs

    def log_prob(self, s, a, t=None):
        self.check_state(s)
        a = self._check_action(a)
        self.check_matching(s, a)
        scaled_logits = self._logits(s) / self._temp  # ns x da
        logZ = logsumexp(scaled_logits, axis=-1)
        if a.ndim == scaled_logits.ndim - 1:
            a = np.expand_dims(a, axis=-1)
        selected_logits = np.take_along_axis(scaled_logits, a, axis=-1)
        return np.squeeze(selected_logits, axis=-1) - logZ

    @abstractmethod
    def score(self, s, a, t=None):  # pragma: no cover
        pass

    def entropy(self, s, t=None):
        self.check_state(s)
        scaled_logits = self._logits(s) / self._temp
        probs = softmax(scaled_logits, axis=-1)
        logZ = logsumexp(scaled_logits, axis=-1, keepdims=True)
        log_probs = scaled_logits - logZ
        ent = -probs * log_probs
        return np.sum(ent, -1)

    @abstractmethod
    def entropy_grad(self, s, t=None):  # pragma: no cover
        pass

    @abstractmethod
    def _logits(self, s):  # pragma: no cover
        pass

    def _probs(self, s):
        logits = self._logits(s)
        return softmax(logits / self._temp, axis=-1)


class LinearSoftmaxPolicy(SoftmaxPolicy):
    def __init__(self, state_dim, num_actions, temperature=1., params_init=None):

        # Mean
        super().__init__(state_dim, num_actions, temperature)

        if params_init is not None:
            if np.isscalar(params_init):
                self._params = params_init + np.zeros((self.num_actions, self.state_dim))
            elif params_init.ndim == 1 and len(params_init) == self.num_actions * self.state_dim:
                self._params = params_init.reshape((self.num_actions, self.state_dim))
            else:
                if params_init.shape != (self.num_actions, self.state_dim):
                    raise ValueError("Bad shape: param_init should be a scalar, "
                                     "a 1d array of size num_actions * state_dim, "
                                     "or a 2d array of size num_actions times state_dim")
                self._params = params_init
        else:
            self._params = np.zeros((self.num_actions, self.state_dim))

    def _logit_grads(self, s):
        x = np.reshape(s, s.shape[:-1] + (1, s.shape[-1]))  # ns x 1 x ds
        return np.kron(np.eye(self.num_actions), x)  # ns x da x d=da*ns

    def _logits(self, s):
        return s @ self._params.T  # ns x da

    @property
    def parameters(self):
        return np.ravel(self._params)

    def set_params(self, params):
        if np.isscalar(params):
            self._params = params + np.zeros((self.num_actions, self.state_dim))
        elif params.ndim == 1 and len(params) == self.num_actions * self.state_dim:
            self._params = params.reshape((self.num_actions, self.state_dim))
        else:
            if params.shape != (self.num_actions, self.state_dim):
                raise ValueError("Bad shape: params should be a scalar, "
                                 "a 1d array of size num_actions * state_dim, "
                                 "or a 2d array of size num_actions times state_dim")
            self._params = params

    def score(self, s, a, t=None):
        self.check_state(s)
        a = self._check_action(a)
        self.check_matching(s, a)
        logit_grads = self._logit_grads(s)  # ns x da x d
        transposed = logit_grads.swapaxes(-1, -2)
        logit_grad_a = transposed[tuple(np.indices(transposed.shape[:-1])) + (a,)]
        logit_grad_mean = self._probs(s)[..., None] * logit_grads
        logit_grad_mean = np.sum(logit_grad_mean, axis=-2)
        return (logit_grad_a - logit_grad_mean) / self._temp

    def entropy_grad(self, s, t=None):
        self.check_state(s)
        scaled_logits = self._logits(s) / self._temp
        probs = softmax(scaled_logits, axis=-1)
        logZ = logsumexp(scaled_logits, axis=-1, keepdims=True)
        log_probs = scaled_logits - logZ
        coeff = probs * log_probs  # ns x da
        logit_grads = self._logit_grads(s)  # ns x da x d
        logit_grad_mean = self._probs(s)[..., None] * logit_grads
        logit_grad_mean = np.sum(logit_grad_mean, axis=-2, keepdims=True)
        scores = (logit_grads - logit_grad_mean) / self._temp
        ent_grad = coeff[..., None] * scores  # ns x da x d
        return - np.sum(ent_grad, -2)


class DeepSoftmaxPolicy(SoftmaxPolicy):
    def __init__(self, state_dim, num_actions, logit_network, temperature=1.):
        super().__init__(state_dim, num_actions, temperature)

        if logit_network is None:
            self._logit_network = nn.Linear(self.state_dim, self.num_actions, bias=False)
            self._logit_network.weight.data.fill_(0.)
        else:
            try:
                with torch.no_grad():
                    s = torch.ones(self.state_dim, dtype=torch.float)
                    a = logit_network(s).numpy()
                    if not a.shape == (self.num_actions,):
                        raise ValueError("Network output should match action dimension")
            except Exception as e:
                raise ValueError("Network could not process state, likely bad input shape") from e
            self._logit_network = logit_network

    @property
    def _flat_params(self):
        return parameters_to_vector(self._logit_network.parameters()).detach().numpy()

    def _set_params(self, params):
        if np.isscalar(params):  # Broadcast
            params = params * torch.ones(size=(self.num_params,), dtype=torch.float,
                                         requires_grad=False)
        else:
            params = torch.tensor(params, dtype=torch.float, requires_grad=False)

        vector_to_parameters(params, self._logit_network.parameters())

    @property
    def parameters(self):
        return self._flat_params

    def set_params(self, params):
        if not np.isscalar(params) and params.shape != (self.num_params,):
            raise ValueError("Bad shape: params should be a scalar, "
                             "or a 1d array of size {}".format(self.num_params))
        self._set_params(params)

    def _logits(self, s, requires_grad=False):
        if not torch.is_tensor(s):
            s = torch.tensor(s, dtype=torch.float, requires_grad=False)
        if requires_grad:
            return self._logit_network(s)
        else:
            with torch.no_grad():
                return self._logit_network(s).numpy()

    def score(self, s, a, t=None):
        self.check_state(s)
        a = self._check_action(a)
        self.check_matching(s, a)

        leading_shape = s.shape[:-1]
        states = torch.as_tensor(s, dtype=torch.float).reshape(-1, self.state_dim)
        actions = torch.as_tensor(a, dtype=torch.long).reshape(-1)
        params = dict(self._logit_network.named_parameters())

        def log_prob(parameters, state, action):
            logits = functional_call(self._logit_network, parameters, (state,)) / self._temp
            log_probs = torch.log_softmax(logits, dim=-1)
            return torch.gather(log_probs, 0, action.unsqueeze(0)).squeeze(0)

        grads = vmap(grad(log_prob), in_dims=(None, 0, 0))(params, states, actions)
        flat_grads = torch.cat(
            [param_grad.reshape(len(states), -1) for param_grad in grads.values()],
            dim=-1,
        )
        return flat_grads.reshape(leading_shape + (self.num_params,)).detach().numpy()

    def weighted_score_sum(self, s, a, weights):
        """Sum scalar-weighted log-policy scores without a score tensor."""
        self.check_state(s)
        a = self._check_action(a)
        self.check_matching(s, a)
        weights = np.asarray(weights)
        if weights.shape != s.shape[:-1]:
            raise ValueError("weights should match state and action leading dimensions")

        parameters = tuple(self._logit_network.parameters())
        reference = parameters[0]
        states = torch.as_tensor(
            s, dtype=reference.dtype, device=reference.device
        )
        actions = torch.as_tensor(
            a, dtype=torch.long, device=reference.device
        ).reshape(s.shape[:-1])
        coefficients = torch.as_tensor(
            weights, dtype=reference.dtype, device=reference.device
        )

        logits = self._logit_network(states) / self._temp
        log_probs = torch.log_softmax(logits, dim=-1)
        selected_log_probs = torch.gather(
            log_probs, -1, actions.unsqueeze(-1)
        ).squeeze(-1)
        objective = torch.sum(coefficients * selected_log_probs)
        gradients = torch.autograd.grad(objective, parameters)
        return torch.cat(
            [parameter_grad.reshape(-1) for parameter_grad in gradients]
        ).detach().cpu().numpy()

    def fused_weighted_score_sum(self, s, a, coefficient_builder):
        """Build detached weights from log probabilities and reuse the forward."""
        self.check_state(s)
        a = self._check_action(a)
        self.check_matching(s, a)
        if not callable(coefficient_builder):
            raise TypeError("coefficient_builder should be callable")

        parameters = tuple(self._logit_network.parameters())
        reference = parameters[0]
        states = torch.as_tensor(
            s, dtype=reference.dtype, device=reference.device
        )
        actions = torch.as_tensor(
            a, dtype=torch.long, device=reference.device
        ).reshape(s.shape[:-1])

        raw_logits = self._logit_network(states)
        logits = raw_logits / self._temp
        log_probs = torch.log_softmax(logits, dim=-1)
        selected_log_probs = torch.gather(
            log_probs, -1, actions.unsqueeze(-1)
        ).squeeze(-1)
        detached_logits = raw_logits.detach().cpu().numpy() / self._temp
        detached_actions = np.asarray(a).reshape(s.shape[:-1])
        detached_log_probs = (
            np.take_along_axis(
                detached_logits, detached_actions[..., None], axis=-1
            )[..., 0]
            - logsumexp(detached_logits, axis=-1)
        ).astype(np.float32, copy=False)
        weights = np.asarray(coefficient_builder(detached_log_probs))
        if weights.shape != s.shape[:-1]:
            raise ValueError(
                "coefficient_builder should return one weight per state-action pair"
            )
        coefficients = torch.as_tensor(
            weights, dtype=reference.dtype, device=reference.device
        )

        objective = torch.sum(coefficients * selected_log_probs)
        gradients = torch.autograd.grad(objective, parameters)
        return torch.cat(
            [parameter_grad.reshape(-1) for parameter_grad in gradients]
        ).detach().cpu().numpy()

    def weighted_score_samples(self, s, a, weights):
        """Return one scalar-weighted score sum for each trajectory.

        This uses vectorized reverse-mode differentiation over trajectories and
        avoids materializing the time-by-parameter score tensor.
        """
        self.check_state(s)
        a = self._check_action(a)
        self.check_matching(s, a)
        weights = np.asarray(weights)
        if weights.shape != s.shape[:-1]:
            raise ValueError("weights should match state and action leading dimensions")

        reference = next(self._logit_network.parameters())
        states = torch.as_tensor(
            s, dtype=reference.dtype, device=reference.device
        )
        actions = torch.as_tensor(
            a, dtype=torch.long, device=reference.device
        ).reshape(s.shape[:-1])
        coefficients = torch.as_tensor(
            weights, dtype=reference.dtype, device=reference.device
        )
        parameters = dict(self._logit_network.named_parameters())

        def trajectory_log_prob(parameters, states, actions, coefficients):
            logits = functional_call(self._logit_network, parameters, (states,))
            log_probs = torch.log_softmax(logits / self._temp, dim=-1)
            selected_log_probs = torch.gather(
                log_probs, -1, actions.unsqueeze(-1)
            ).squeeze(-1)
            return torch.sum(coefficients * selected_log_probs)

        grads = vmap(
            grad(trajectory_log_prob), in_dims=(None, 0, 0, 0)
        )(parameters, states, actions, coefficients)
        flat_grads = torch.cat(
            [parameter_grad.reshape(len(states), -1)
             for parameter_grad in grads.values()],
            dim=-1,
        )
        return flat_grads.detach().cpu().numpy()

    def entropy_grad(self, s, t=None):
        self.check_state(s)

        leading_shape = s.shape[:-1]
        states = torch.as_tensor(s, dtype=torch.float).reshape(-1, self.state_dim)
        params = dict(self._logit_network.named_parameters())

        def entropy(parameters, state):
            logits = functional_call(self._logit_network, parameters, (state,)) / self._temp
            log_probs = torch.log_softmax(logits, dim=-1)
            return -torch.sum(torch.exp(log_probs) * log_probs)

        grads = vmap(grad(entropy), in_dims=(None, 0))(params, states)
        flat_grads = torch.cat(
            [param_grad.reshape(len(states), -1) for param_grad in grads.values()],
            dim=-1,
        )
        return flat_grads.reshape(leading_shape + (self.num_params,)).detach().numpy()
