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
from potion.algorithms.safe import legacy_adabatch
import argparse
import re
from potion.common.rllab_utils import rllab_env_from_name, Rllab2GymWrapper
from potion.meta.smoothing_constants import pirotta_coeff, gauss_smooth_const
from potion.meta.variance_bounds import (gpomdp_var_bound, reinforce_var_bound, 
                                            gauss_gradient_range)


# Command line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--name', help='Experiment name', type=str, 
                    default='AdaBatch')
parser.add_argument('--estimator', help='PG estimator (reinforce/gpomdp)', 
                    type=str, default='gpomdp')
parser.add_argument('--bound', help='Statistical inequality', 
                    type=str, default='bernstein')
parser.add_argument('--baseline', help='control variate (avg/peters/zero)', 
                    type=str, default='peters')
parser.add_argument('--seed', help='RNG seed', type=int, default=0)
parser.add_argument('--env', help='Gym environment id', type=str, 
                    default='LQ-v0')
parser.add_argument('--horizon', help='Task horizon', type=int, default=10)
parser.add_argument('--max_samples', help='Maximum total samples', type=int, 
                    default=2e7)
parser.add_argument('--min_batchsize', help='(Minimum) batch size', type=int, 
                    default=100)
parser.add_argument('--max_batchsize', help='Maximum batch size', type=int, 
                    default=100000)
parser.add_argument('--disc', help='Discount factor', type=float, default=0.9)
parser.add_argument('--conf', help='Confidence', type=float, default=0.95)
parser.add_argument('--std_init', help='Initial policy std', type=float,
                    default=1.)
parser.add_argument('--max_feat', help='Maximum state feature', type=float,
                    default=1.)
parser.add_argument('--max_rew', help='Maximum reward', type=float,
                    default=2.)
parser.add_argument('--action_vol', help='Volume of action space', type=float,
                    default=2.)
parser.add_argument('--max_action', help='Radius of action space', 
                    type=float, default=1.)
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
parser.add_argument("--emp", help="Use empirical range",
                    action="store_true")
parser.add_argument("--no-emp", help="Use theoretical range",
                    action="store_false")
parser.set_defaults(render=False, temp=False, learnstd=False, test=False, 
                        emp=False) 
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
logstd_init = torch.log(torch.zeros(1) + args.std_init)
policy = ShallowGaussianPolicy(m, d, 
                               mu_init=mu_init, 
                               logstd_init=logstd_init, 
                               learn_std=args.learnstd)

test_batchsize = args.min_batchsize if args.test else 0

envname = re.sub(r'[^a-zA-Z]', "", args.env)[:-1]
envname = re.sub(r'[^a-zA-Z]', "", args.env)[:-1].lower()
logname = envname + '_' + args.name + '_' + str(args.seed)

if args.temp:
    logger = Logger(directory='../temp', name = logname)
else:
    logger = Logger(directory='../logs', name = logname)

#Constants
_, kappa, _ = gauss_smooth_const(args.max_feat, args.std_init)
pen_coeff = pirotta_coeff(args.max_feat, args.max_rew, args.disc, 
                          args.std_init, args.action_vol)
if args.estimator == 'reinforce':
    var_bound = reinforce_var_bound(args.max_rew, args.disc, kappa,
                                    args.horizon)
elif args.estimator == 'gpomdp':
    var_bound = gpomdp_var_bound(args.max_rew, args.disc, kappa, args.horizon)
else:
    raise NotImplementedError

if args.emp:
    grad_range = None
else:
    grad_range = gauss_gradient_range(args.max_rew, args.max_feat, args.disc, 
                                  args.horizon, args.max_action, 
                                  args.std_init, estimator=args.estimator)

# Run
legacy_adabatch(env, policy,
            pen_coeff = pen_coeff,
            bound = args.bound,
            var_bound = var_bound,
            grad_range = grad_range,
            horizon = args.horizon,
            min_batchsize = args.min_batchsize,
            max_batchsize = args.max_batchsize,
            max_samples = args.max_samples,
            disc = args.disc,
            fail_prob = 1. - args.conf,
            seed = args.seed,
            logger = logger,
            render = args.render,
            shallow = True,
            estimator = args.estimator,
            baseline = args.baseline,
            test_batchsize=test_batchsize,
            oracle = lambda x: env.computeJ(x, args.std_init),
            save_params=False)