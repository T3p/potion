from potion.algorithms import reinforce
import gymnasium as gym
import potion.envs
from potion.policies.gaussian_policies import LinearGaussianPolicy
from potion.evaluation.loggers import SilentLogger
from potion.simulation.trajectory_generators import estimate_average_return
import numpy as np
from potion.optimization.gradient_descent import Adam


def test_reinforce():
    seed = 42
    horizon = 10
    discount = 1.
    env = gym.make("LQR-v0")
    policy = LinearGaussianPolicy.make(env, std_init=0.1)
    reinforce(env, policy,
              step_size=1e-2,
              horizon=horizon,
              discount=discount,
              baseline="peters",
              max_iterations=20,
              seed=seed,
              logger=SilentLogger())

    ret = estimate_average_return(env, policy,
                                  n_episodes=100,
                                  max_trajectory_len=horizon,
                                  discount=discount,
                                  rng=np.random.default_rng(seed))

    assert ret > - 1.


def test_reinforce_with_adam():
    seed = 42
    horizon = 10
    discount = 1.
    env = gym.make("LQR-v0")
    policy = LinearGaussianPolicy.make(env, std_init=0.1)
    reinforce(env, policy,
              step_size=Adam(alpha=1e-1),
              horizon=horizon,
              discount=discount,
              baseline="peters",
              max_iterations=20,
              seed=seed,
              logger=SilentLogger())

    ret = estimate_average_return(env, policy,
                                  n_episodes=100,
                                  max_trajectory_len=horizon,
                                  discount=discount,
                                  rng=np.random.default_rng(seed))

    assert ret > -1
