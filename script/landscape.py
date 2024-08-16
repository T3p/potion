import gym
import torch
import numpy as np
from potion.simulation.play import play
from potion.policies.gaussian_policies import LinearGaussianPolicy
from potion.common.misc_utils import returns
import matplotlib.pyplot as plt
import potion.envs

disc = 0.9
horizon = 20
std= 1.

env = gym.make('lqr1d-v0')

m = sum(env.observation_space.shape)
d = sum(env.action_space.shape)
start = torch.zeros(m*d) - 2.
end = torch.zeros(m*d)

logstd = torch.log(torch.zeros(d) + std)
policy = LinearGaussianPolicy(m, d, learn_std=False, std_init=logstd)

points = np.linspace(0, 1, 100)
rets = []
for i, alpha in enumerate(points):
    print('%d/%d' % (i, len(points)))
    param = (1 - alpha) * start + alpha * end
    policy.set_from_flat(param)
    batch = play(env, policy, horizon=horizon, episodes=100, render=False, action_filter=None, fast=True)
    rets.append(np.mean(returns(batch, disc)))
    
plt.plot(points, rets)
plt.show()