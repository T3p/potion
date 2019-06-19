#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 14:47:33 2019

@author: Matteo Papini
"""
import torch
import gym
import potion.envs
import argparse
from potion.actors.continuous_policies import ShallowGaussianPolicy
from potion.common.logger import Logger
from potion.simulation.play import play
from potion.common.misc_utils import performance, clip
import re



# Command line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--seed', help='RNG seed', type=int, default=0)
parser.add_argument('--env', help='Gym environment id', type=str, default='ContCartPole-v0')
parser.add_argument('--horizon', help='Task horizon', type=int, default=1000)
parser.add_argument('--episodes', help='Number of trials', type=int, default=10)
parser.add_argument('--gamma', help='Discount factor', type=float, default=0.99)
parser.add_argument("--render", help="Render an episode",
                    action="store_true")
parser.add_argument("--no-render", help="Do not render any episode",
                    action="store_false")
parser.set_defaults(render=True) 

args = parser.parse_args()


#Safe params:
safe_params = [0.986962854862213, -0.0940133929252625, 2.56180429458618, 14.6990728378296, 13.2945461273193]
    
#Unsafe params:
unsafe_params = [ 1.1094,  0.7132,  6.0743, 23.7693, 12.0861]


params = unsafe_params

env = gym.make(args.env)
env.seed(args.seed)

m = sum(env.observation_space.shape)
d = sum(env.action_space.shape)
mu_init = params[1:]
logstd_init = params[0]
policy = ShallowGaussianPolicy(m, d, 
                               mu_init=mu_init, 
                               logstd_init=logstd_init)

envname = re.sub(r'[^a-zA-Z]', "", args.env)[:-1]
envname = re.sub(r'[^a-zA-Z]', "", args.env)[:-1].lower()
logname = envname + '_' + 'play' + '_' + str(args.seed)

logger = Logger(directory='../temp', name = logname)

# Run
for i in range(args.episodes):
    batch = play(env, policy, horizon=args.horizon, episodes=1, render=args.render, action_filter=clip(env))
    print("Episode %d:\nperformance: %f\n\n" % (i, performance(batch, disc=args.gamma)))