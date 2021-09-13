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
from potion.algorithms.safe import safe_step
import argparse
import re
from potion.common.rllab_utils import rllab_env_from_name, Rllab2GymWrapper
from potion.meta.smoothing_constants import gauss_lip_const


# Command line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--name', help='Experiment name', type=str, 
                    default='SafeStepStrict')
parser.add_argument('--estimator', help='PG estimator (reinforce/gpomdp)', 
                    type=str, default='gpomdp')
parser.add_argument('--baseline', help='control variate (avg/peters/zero)', 
                    type=str, default='peters')
parser.add_argument('--batchsize', help='batch size', type=int, 
                    default=500)
parser.add_argument('--seed', help='RNG seed', type=int, default=0)
parser.add_argument('--env', help='Gym environment id', type=str, 
                    default='LQ-v0')
parser.add_argument('--horizon', help='Task horizon', type=int, default=10)
parser.add_argument('--max_samples', help='Maximum total samples', type=int, 
                    default=1e7)
parser.add_argument('--disc', help='Discount factor', type=float, default=0.9)
parser.add_argument('--conf', help='Confidence', type=float, default=0.2)
parser.add_argument('--std', help='Initial policy std', type=float,
                    default=1.)
parser.add_argument('--max_feat', help='Maximum state feature', type=float,
                    default=1.)
parser.add_argument('--max_rew', help='Maximum reward', type=float,
                    default=2.)
parser.add_argument("--render", help="Render an episode",
                    action="store_true")
parser.add_argument("--no-render", help="Do not render any episode",
                    action="store_false")
parser.add_argument("--temp", help="Save logs in temp folder",
                    action="store_true")
parser.add_argument("--no-temp", help="Save logs in logs folder",
                    action="store_false")
parser.set_defaults(render=False, temp=False) 
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
logstd = torch.log(torch.zeros(1) + args.std)
policy = ShallowGaussianPolicy(m, d, 
                               mu_init=mu_init, 
                               logstd_init=logstd, 
                               learn_std=False)

envname = re.sub(r'[^a-zA-Z]', "", args.env)[:-1]
envname = re.sub(r'[^a-zA-Z]', "", args.env)[:-1].lower()
logname = envname + '_' + args.name + '_' + str(args.seed)

if args.temp:
    logger = Logger(directory='../temp', name = logname)
else:
    logger = Logger(directory='../logs', name = logname)

#Constants
lip_const = gauss_lip_const(args.max_feat, args.max_rew, args.disc, args.std)


#Run
safe_step(env=env, 
            policy=policy, 
            disc=args.disc, 
            horizon=args.horizon, 
            lip_const=lip_const, 
            batchsize = args.batchsize,
            conf = args.conf,
            max_samples = args.max_samples,
            logger = logger,
            estimator = args.estimator,
            baseline = args.baseline,
            shallow = True,
            seed = args.seed,
            verbose = 1)