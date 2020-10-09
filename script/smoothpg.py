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
from potion.algorithms.smooth import smoothpg
import argparse
import re
from potion.common.rllab_utils import rllab_env_from_name, Rllab2GymWrapper
from gym.spaces.discrete import Discrete
from potion.actors.discrete_policies import ShallowGibbsPolicy
from potion.meta.smoothing_constants import gauss_lip_const, gauss_lip2_const


# Command line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--name', help='Experiment name', type=str, default='SmoPG')
parser.add_argument('--estimator', help='Policy gradient estimator (reinforce/gpomdp)', type=str, default='gpomdp')
parser.add_argument('--baseline', help='baseline for policy gradient estimator (avg/peters/zero)', type=str, default='peters')
parser.add_argument('--seed', help='RNG seed', type=int, default=0)
parser.add_argument('--env', help='Gym environment id', type=str, default='lqr1d-v0')
parser.add_argument('--horizon', help='Task horizon', type=int, default=20)
parser.add_argument('--min_batchsize', help='(Minimum) batch size', type=int, default=100)
parser.add_argument('--max_batchsize', help='Maximum batch size', type=int, default=1000)
parser.add_argument('--max_samples', help='Maximum total samples', type=int, default=1e6)
parser.add_argument('--disc', help='Discount factor', type=float, default=0.9)
parser.add_argument('--std', help='Policy std', type=float, default=1.)
parser.add_argument('--max_feat', help='Max value of state features', type=float, default=4.)
parser.add_argument('--max_rew', help='Max value of reward', type=float, default=14.5)
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
parser.add_argument("--fast", help="Test on deterministic policy",
                    action="store_true")
parser.add_argument("--no-fast", help="Online learning only",
                    action="store_false")
parser.add_argument("--oracle", help="Use curvature oracle",
                    action="store_true")
parser.add_argument("--no-oracle", help="Use curvature bound",
                    action="store_false")
parser.set_defaults(render=False, temp=False, test=False, fast=False, oracle=False) 

args = parser.parse_args()

# Prepare
if args.env.startswith('rllab'):
    env_rllab_class = rllab_env_from_name(args.env[6:])
    env_rllab = env_rllab_class()
    env = Rllab2GymWrapper(env_rllab)
else:
    env = gym.make(args.env)
env.seed(args.seed)

if type(env.action_space) is Discrete:
    raise NotImplementedError
    #policy = ShallowGibbsPolicy(env,temp=1.)
else:
    m = sum(env.observation_space.shape)
    d = sum(env.action_space.shape)
    mu_init = torch.zeros(m*d)
    logstd_init = torch.log(torch.zeros(d) + args.std)
    policy = ShallowGaussianPolicy(m, d, 
                               mu_init=mu_init, 
                               logstd_init=logstd_init, 
                               learn_std=False)
    lip_const1 = gauss_lip_const(args.max_feat, args.max_rew, args.disc, args.std)
    lip_const2 = gauss_lip2_const(args.max_feat, args.max_rew, args.disc, args.std)

test_batchsize = args.min_batchsize if args.test else 0

envname = re.sub(r'[^a-zA-Z]', "", args.env)[:-1]
envname = re.sub(r'[^a-zA-Z]', "", args.env)[:-1].lower()
logname = envname + '_' + args.name + '_' + str(args.seed)

if args.temp:
    logger = Logger(directory='../temp', name = logname)
else:
    logger = Logger(directory='../logs', name = logname)

# Run
smoothpg(env, policy, 
            horizon=args.horizon, 
            lip_const1 = lip_const1,
            lip_const2 = lip_const2,
            min_batchsize = args.min_batchsize,
            max_batchsize = args.max_batchsize,
            max_samples = args.max_samples,
            disc = args.disc,
            seed = args.seed,
            logger = logger,
            render = args.render,
            shallow = True,
            estimator = args.estimator,
            test_batchsize=test_batchsize)