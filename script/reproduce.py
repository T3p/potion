import math
import warnings

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.distributions import Normal
from potion.actors.continuous_policies import ShallowGaussianPolicy as sgp

"""
Most of this code is taken from: https://github.com/T3p/potion/
"""


def num_params(params):
    with torch.no_grad():
        return sum(p.numel() for p in params)


def flatten(params):
    """
    Turns a module's parameters (or gradients) into a flat numpy array

    params: the module's parameters (or gradients)
    """
    with torch.no_grad():
        return torch.cat([p.data.view(-1) for p in params])


def set_from_flat(params, values):
    """
    Sets a module's parameters from a flat array or tensor

    params: the module's parameters
    values: a flat array or tensor
    """
    with torch.no_grad():
        if not torch.is_tensor(values):
            values = torch.tensor(values)
        k = 0
        for p in params:
            shape = tuple(list(p.shape))
            offset = torch.prod(torch.tensor(shape)).item()
            val = values[k: k + offset]
            val = val.view(shape)
            with torch.no_grad():
                p.copy_(val)
            k = k + offset


class FlatModule(nn.Module):
    """Module with flattened parameter management"""

    def num_params(self):
        """Number of parameters of the module"""
        return num_params(self.parameters())

    def get_flat(self):
        """Module parameters as flat array"""
        return flatten(self.parameters())

    def set_from_flat(self, values):
        """Set module parameters from flat array"""
        set_from_flat(self.parameters(), values)

    def save_flat(self, path):
        try:
            torch.save(self.get_flat(), path)
        except:
            warnings.warn('Could not save parameters!')

    def load_from_flat(self, path):
        try:
            values = torch.load(path)
        except:
            warnings.warn('Could not load parameters!')
            return
        self.set_from_flat(values)


def flat_gradients(module, loss, coeff=None):
    module.zero_grad()
    loss.backward(coeff, retain_graph=True)
    return torch.cat([p.grad.view(-1) for p in module.parameters()])


def jacobian(module, loss, coeff=None):
    """Inefficient! Use jacobian-vector product whenever possible
    (still useful for nonlinear functions of gradients, such as
    in Peter's baseline for REINFORCE)"""
    mask = torch.eye(loss.numel())

    jac = torch.stack([flat_gradients(module, loss, mask[i, :])
                       for i in range(loss.numel())],
                      dim=0)
    return jac


def tensormat(a, b):
    """
    a: NxHxm
    b: NxH
    a*b: NxHxm
    """
    return torch.einsum('ijk,ij->ijk', (a, b))


def complete_out(x, dim):
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    while x.dim() < dim:
        x = x.unsqueeze(0)
    return x


def complete_in(x, dim):
    while x.dim() < dim:
        x = x.unsqueeze(-1)
    return x


def atanh(x):
    return 0.5 * (torch.log(1 + x) - torch.log(1 - x))


def maybe_tensor(x):
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    return x


def discount(rewards, disc):
    """rewards: array or tensor"""
    i = 0 if rewards.dim() < 2 else 1
    discounts = torch.tensor(disc ** np.indices(rewards.shape)[i], dtype=torch.float)
    return rewards * discounts


class LinearMapping(FlatModule):
    def __init__(self, d_in, d_out, bias=False):
        super(LinearMapping, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.linear = nn.Linear(d_in, d_out, bias)

    def forward(self, x):
        return self.linear(x)


class ContinuousPolicy(FlatModule):
    """Alias"""
    pass


class ShallowGaussianPolicy(ContinuousPolicy):
    """
    Factored
    linear mean \mu_{\theta}(x)
    diagonal, state-independent std \sigma = e^{\omega}
    Provides closed-form scores
    N.B. the squashing function is NOT considered in gradients
    """

    def __init__(self, n_states, n_actions, feature_fun=None, squash_fun=None,
                 mu_init=None, logstd_init=None,
                 learn_std=True):
        super(ShallowGaussianPolicy, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.feature_fun = feature_fun
        self.squash_fun = squash_fun
        self.learn_std = learn_std
        self.mu_init = mu_init
        self.logstd_init = logstd_init

        # Mean
        if feature_fun is None:
            self.mu = LinearMapping(n_states, n_actions)
        else:
            self.mu = LinearMapping(len(feature_fun(torch.ones(n_states))), n_actions)
        if mu_init is not None:
            self.mu.set_from_flat(mu_init)

        # Log of standard deviation
        if logstd_init is None:
            logstd_init = torch.zeros(self.n_actions)
        elif not torch.is_tensor(logstd_init):
            logstd_init = torch.tensor(logstd_init)
        if learn_std:
            self.logstd = nn.Parameter(logstd_init)
        else:
            self.logstd = autograd.Variable(logstd_init)

        # Normal(0,1)
        self._pdf = Normal(torch.zeros_like(self.logstd.data),
                           torch.ones(n_actions))

    def log_pdf(self, s, a):
        log_sigma = self.logstd
        sigma = torch.exp(log_sigma)
        if self.feature_fun is not None:
            x = self.feature_fun(s)
        else:
            x = s

        logp = -((a - self.mu(x)) ** 2) / (2 * sigma ** 2) - \
               log_sigma - .5 * math.log(2 * math.pi)
        return torch.sum(logp, -1)

    def forward(self, s, a):
        return torch.exp(self.log_pdf(s, a))

    def act(self, s, deterministic=False):
        import numpy as np
        if isinstance(s, np.ndarray):
            s = torch.tensor(s, dtype=torch.float32)
        with torch.no_grad():
            if self.feature_fun is not None:
                x = self.feature_fun(s)
            else:
                x = s

            if deterministic:
                return self.mu(x)

            sigma = torch.exp(self.logstd.data)
            a = self.mu(x) + self._pdf.sample() * sigma
            if self.squash_fun is not None:
                a = self.squash_fun(a)
            return a

    def num_loc_params(self):
        return self.n_states

    def num_scale_params(self):
        return self.n_actions

    def get_loc_params(self):
        return self.mu.get_flat()

    def get_scale_params(self):
        return self.logstd.data

    def set_loc_params(self, val):
        self.mu.set_from_flat(val)

    def set_scale_params(self, val):
        with torch.no_grad():
            self.logstd.data = torch.tensor(val)

    def exploration(self):
        return torch.exp(torch.sum(self.logstd)).data

    def loc_score(self, s, a):
        sigma = torch.exp(self.logstd)
        if self.feature_fun is not None:
            x = self.feature_fun(s)
        else:
            x = s
        score = torch.einsum('ijk,ijh->ijhk', (x, (a - self.mu(x)) / sigma ** 2)) # TODO: bug found here!!! hk instead of kh
        score = score.reshape((score.shape[0], score.shape[1], score.shape[2] * score.shape[3]))
        return score

    def scale_score(self, s, a):
        sigma = torch.exp(self.logstd)
        if self.feature_fun is not None:
            x = self.feature_fun(s)
        else:
            x = s
        return ((a - self.mu(x)) / sigma) ** 2 - 1

    def score(self, s, a):
        s = complete_out(s, 3)
        a = complete_out(a, 3)
        if self.learn_std:
            return torch.cat((self.scale_score(s, a),
                              self.loc_score(s, a)),
                             2)
        else:
            return self.loc_score(s, a)

    def entropy(self, s):
        s = complete_out(s, 3)
        ent = torch.sum(self.logstd) + \
              1. / (2 * self.n_actions) * (1 + math.log(2 * math.pi))
        return torch.zeros(s.shape[:-1]) + ent

    def entropy_grad(self, s):
        s = complete_out(s, 3)
        if self.learn_std:
            return torch.cat((torch.ones(s.shape[:-1] + (self.n_actions,)),
                              torch.zeros(s.shape[:-1] + (self.n_states,))), -1)
        else:
            return torch.zeros(s.shape[:-1] + (self.n_states,))

    def info(self):
        return {'PolicyClass': self.__class__.__name__,
                'LearnStd': self.learn_std,
                'StateDim': self.n_states,
                'ActionDim': self.n_actions,
                'MuInit': self.mu_init,
                'LogstdInit': self.logstd_init,
                'FeatureFun': self.feature_fun,
                'SquashFun': self.squash_fun}


if __name__ == '__main__':
    params = np.array([1.0 for _ in range(3)] + [0.0 for _ in range(3)])
    print(params.reshape(2, 3))

    policy = sgp(3,
                                   2,
                                   feature_fun=None,
                                   squash_fun=None,
                                   mu_init=params.tolist(),
                                   logstd_init=None,
                                   learn_std=False)

    print("Parameters")
    print(policy.get_flat())

    print("Actions")
    state = torch.ones((3,))
    print(policy.act(state, deterministic=True))

    print("Scores!")
    a = torch.tensor([1., 0.])
    s1 = policy.score(torch.tensor(state, dtype=torch.float32), a)
    print(s1)
    print()

    print("Update")
    print(params + s1)
