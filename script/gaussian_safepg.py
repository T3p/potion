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
from potion.algorithms.safe import safepg, incr_safepg
import argparse
import re
from potion.common.rllab_utils import rllab_env_from_name, Rllab2GymWrapper
from potion.meta.smoothing_constants import gauss_smooth_const, gauss_lip_const
from potion.meta.variance_bounds import gpomdp_var_bound, reinforce_var_bound


# Command line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--name', help='Experiment name', type=str, default='safepg')
parser.add_argument('--estimator', help='Policy gradient estimator (reinforce/gpomdp)', type=str, default='gpomdp')
parser.add_argument('--baseline', help='baseline for policy gradient estimator (avg/peters/zero)', type=str, default='peters')
parser.add_argument('--seed', help='RNG seed', type=int, default=0)
parser.add_argument('--env', help='Gym environment id', type=str, default='LQG1D-v0')
parser.add_argument('--horizon', help='Task horizon', type=int, default=20)
parser.add_argument('--batchsize', help='Initial batch size', type=int, default=100)
parser.add_argument('--maxbatchsize', help='Maximum batch size', type=int, default=10000)
parser.add_argument('--iterations', help='Iterations', type=int, default=500)
parser.add_argument('--gamma', help='Discount factor', type=float, default=0.9)
parser.add_argument('--sigmainit', help='Initial policy std', type=float, default=1.)
parser.add_argument('--maxfeat', help='Maximum state feature', type=float, default=28.8)
parser.add_argument('--maxrew', help='Maximum reward', type=float, default=4.)
parser.add_argument('--stepper', help='Step size rule', type=str, default='constant')
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
parser.set_defaults(render=False, temp=False, learnstd=False, test=False) 

args = parser.parse_args()

# Prepare
if args.env.startswith('rllab'):
    env_rllab_class = rllab_env_from_name(args.env)
    env_rllab = env_rllab_class()
    env = Rllab2GymWrapper(env_rllab)
else:
    env = gym.make(args.env)
env.seed(args.seed)

m = sum(env.observation_space.shape)
d = sum(env.action_space.shape)
mu_init = torch.zeros(m)
logstd_init = torch.log(torch.zeros(1) + args.sigmainit)
policy = ShallowGaussianPolicy(m, d, 
                               mu_init=mu_init, 
                               logstd_init=logstd_init, 
                               learn_std=args.learnstd)

test_batchsize = args.batchsize if args.test else 0

envname = re.sub(r'[^a-zA-Z]', "", args.env)[:-1]
envname = re.sub(r'[^a-zA-Z]', "", args.env)[:-1].lower()
logname = envname + '_' + args.name + '_' + str(args.seed)

if args.temp:
    logger = Logger(directory='../temp', name = logname)
else:
    logger = Logger(directory='../logs', name = logname)

#Constants
_, kappa, _ = gauss_smooth_const(args.maxfeat, args.sigmainit)
lip_const = gauss_lip_const(args.maxfeat, args.maxrew, args.gamma, args.sigmainit)
if args.estimator == 'reinforce':
    var_bound = reinforce_var_bound(args.maxrew, args.gamma, kappa, args.horizon)
elif args.estimator == 'gpomdp':
    var_bound = gpomdp_var_bound(args.maxrew, args.gamma, kappa, args.horizon)
else:
    raise NotImplementedError


# Run
incr_safepg(env, policy,
            lip_const = lip_const,
            var_bound = var_bound,
            horizon = args.horizon,
            #init_batchsize = args.batchsize,
            max_batchsize = args.maxbatchsize,
            iterations = args.iterations,
            disc = args.gamma,
            seed = args.seed,
            logger = logger,
            render = args.render,
            shallow = True,
            estimator = args.estimator,
            baseline = args.baseline,
            test_batchsize=test_batchsize)