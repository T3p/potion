from potion.algorithms import reinforce
import gymnasium as gym
import potion.envs
from potion.policies.gaussian_policies import LinearGaussianPolicy
from potion.evaluation.loggers import SilentLogger, EpisodicPerformanceLogger
from potion.simulation.trajectory_generators import estimate_average_return
import numpy as np
from potion.optimization.gradient_descent import Adam
import pytest
from potion.policies.wrappers import Staged


@pytest.mark.skip("Not now")
def test_policy_gradient_nonstationary():
    step_size = 1e-1
    seed = 42
    horizon = 3
    discount = 1.
    policy_std = 0.2
    env = potion.envs.LQR(init_mean=1., init_std=0.)  # gym.make("LQR-v0")
    policy = Staged(LinearGaussianPolicy.make(env, std_init=policy_std), horizon=horizon)
    reinforce(env, policy,
              step_size=step_size,
              batch_size=100,
              horizon=horizon,
              discount=discount,
              estimator="nonstationary",
              baseline="peters",
              max_iterations=100,
              seed=seed,
              logger=EpisodicPerformanceLogger(path=None, log_every=100, log_params=True),
              verbose=True)

    ret = estimate_average_return(env, policy,
                                  n_episodes=1000,
                                  horizon=horizon,
                                  discount=discount,
                                  rng=np.random.default_rng(seed))

    optimal_gains = env.unwrapped.optimal_gains(horizon)
    optimal_param = np.concatenate([optimal_gains[h].ravel() for h in range(horizon)])
    optimal_ret = env.unwrapped.optimal_return(horizon, policy_std)

    print("RESULT:")
    print(policy.parameters)
    print(ret)

    print("OPTIMAL:")
    print(optimal_param)
    print(optimal_ret)

    assert np.allclose(policy.parameters, optimal_param, atol=1e-1)
    assert np.isclose(ret, optimal_ret, atol=1e-1)


@pytest.mark.skip("Not now")
def test_policy_gradient_continual():
    step_size = 1e-2
    seed = 42
    discount = 0.9
    policy_std = 0.2
    env = potion.envs.LQR(init_mean=1., init_std=0.)
    policy = LinearGaussianPolicy.make(env, std_init=policy_std)
    reinforce(env, policy,
              step_size=step_size,
              batch_size=100,
              horizon=None,
              discount=0.9,
              estimator="gpomdp",
              baseline="peters",
              max_iterations=100,
              seed=seed,
              logger=EpisodicPerformanceLogger(path=None, log_every=100, log_params=True),
              verbose=True)

    ret = estimate_average_return(env, policy,
                                  n_episodes=1000,
                                  horizon=None,
                                  discount=discount,
                                  rng=np.random.default_rng(seed))

    optimal_param = env.discounted_optimal_gain(discount).ravel()
    optimal_ret = env.discounted_optimal_return(discount, policy_std)

    print("RESULT:")
    print(policy.parameters, ret)

    print("OPTIMAL:")
    print(optimal_param, optimal_ret)

    assert np.allclose(policy.parameters, optimal_param, atol=1e-1)
    assert np.isclose(ret, optimal_ret, atol=1e-1)

