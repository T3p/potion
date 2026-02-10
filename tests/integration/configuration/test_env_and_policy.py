import pytest
import potion.envs
from potion.policies.gaussian_policies import LinearGaussianPolicy, DeepGaussianPolicy
from potion.policies.softmax_policies import LinearSoftmaxPolicy, DeepSoftmaxPolicy


def test_default_gaussian_policies():
    env = potion.envs.LQR(init_mean=1., init_std=0.)
    pol1 = LinearGaussianPolicy.make(env, std_init=0.5, learn_std=True)
    pol2 = DeepGaussianPolicy.make(env, std_init=0.5, learn_std=True)

    assert isinstance(pol1, LinearGaussianPolicy)
    assert pol1.state_dim == sum(env.observation_space.shape)
    assert pol1.action_dim == sum(env.action_space.shape)
    assert pol1.std == 0.5
    assert pol1.learn_std

    assert isinstance(pol2, DeepGaussianPolicy)
    assert pol2.state_dim == sum(env.observation_space.shape)
    assert pol2.action_dim == sum(env.action_space.shape)
    assert pol2.std == 0.5
    assert pol2.learn_std
