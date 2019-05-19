#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 14:47:33 2019

@author: Matteo Papini
"""
import torch
import gym
import potion.envs
from potion.actors.continuous_policies import ShallowGaussianPolicy
from potion.common.logger import Logger
from potion.algorithms.semisafe import adastep
import argparse
import re

#Command line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--name', help='Experiment name', type=str, default='ADASTEP')
parser.add_argument('--seed', help='RNG seed', type=int, default=0)
parser.add_argument('--env', help='Gym environment id', type=str, default='LQG1D-v0')
parser.add_argument('--max_rew', help='Max reward', type=float, default=28.8)
parser.add_argument('--max_feat', help='Max state feature', type=float, default=4.)
parser.add_argument('--action_vol', help='Volume of action space', type=float, default=8.)
parser.add_argument('--horizon', help='Task horizon', type=int, default=20)
parser.add_argument('--batchsize', help='Batch size', type=int, default=500)
parser.add_argument('--iterations', help='Iterations', type=int, default=1000)
parser.add_argument('--gamma', help='Discount factor', type=float, default=0.9)
parser.add_argument('--delta', help='Confidence parameter', type=float, default=0.2)
parser.add_argument('--sigmainit', help='Initial policy std', type=float, default=1.)
parser.add_argument("--render", help="Render an episode",
                    action="store_true")
parser.add_argument("--no-render", help="Do not render any episode",
                    action="store_false")
parser.add_argument("--temp", help="Save logs in temp folder",
                    action="store_true")
parser.add_argument("--no-temp", help="Save logs in logs folder",
                    action="store_false")
parser.add_argument("--test", help="Test on deterministic policy",
                    action="store_true")
parser.add_argument("--no-test", help="Online learning only",
                    action="store_false")
parser.set_defaults(render=False, temp=False, test=False) 

args = parser.parse_args()

#Prepare
env = gym.make(args.env)

m = sum(env.observation_space.shape)
d = sum(env.action_space.shape)
mu_init = torch.zeros(m)
logstd_init = torch.log(torch.zeros(1) + args.sigmainit)
policy = ShallowGaussianPolicy(m, d, 
                               mu_init=mu_init, 
                               logstd_init=logstd_init, 
                               learn_std=False)

env.seed(args.seed)

test_batchsize = 100 if args.test else 0

envname = re.sub(r'[^a-zA-Z]', "", args.env)[:-1]
envname = re.sub(r'[^a-zA-Z]', "", args.env)[:-1].lower()
logname = envname + '_' + args.name + '_' + str(args.seed)

if args.temp:
    logger = Logger(directory='../temp', name = logname)
else:
    logger = Logger(directory='../logs', name = logname)
    
#Run
adastep(env, policy,
            horizon = args.horizon,
            batchsize = args.batchsize,
            iterations = args.iterations,
            disc = args.gamma,
            max_feat = args.max_feat,
            max_rew = args.max_rew,
            action_vol = args.action_vol,
            conf = args.delta,
            seed = args.seed,
            logger = logger,
            render = args.render,
            test_batchsize=test_batchsize)