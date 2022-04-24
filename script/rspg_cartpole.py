#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 14:47:33 2019

@author: Matteo Papini
"""
import torch
import gym
import potion.envs
from potion.actors.discrete_policies import ShallowGibbsPolicy
from potion.common.logger import Logger
from potion.algorithms.safe import relaxed_spg
import argparse
import re
from potion.common.rllab_utils import rllab_env_from_name, Rllab2GymWrapper
from potion.meta.smoothing_constants import gibbs_lip_const
from potion.meta.error_bounds import emp_bernstein


# Command line arguments
parser = argparse.ArgumentParser(formatter_class
                                 =argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--name', help='Experiment name', type=str, default='RSPG')
parser.add_argument('--storage', help='root of log directories', type=str, default='..')
parser.add_argument('--estimator', help='PG estimator (reinforce/gpomdp)', 
                    type=str, default='gpomdp')
parser.add_argument('--baseline', help='control variate (avg/peters/zero)', 
                    type=str, default='peters')
parser.add_argument('--seed', help='RNG seed', type=int, default=0)
parser.add_argument('--env', help='Gym environment id', type=str, 
                    default='CartPole-v1')
parser.add_argument('--horizon', help='Task horizon', type=int, default=100)
parser.add_argument('--max_samples', help='Maximum total samples', type=int, 
                    default=2e6)
parser.add_argument('--mini_batchsize', help='(Minimum/mini) batch size', type=int, 
                    default=100)
parser.add_argument('--max_batchsize', help='Maximum batch size', type=int, 
                    default=100000)
parser.add_argument('--disc', help='Discount factor', type=float, default=0.9)
parser.add_argument('--conf', help='Confidence', type=float, default=0.8)
parser.add_argument('--std_init', help='Initial policy std', type=float, 
                    default=1.)
parser.add_argument('--max_feat', help='Maximum state feature', type=float, 
                    default=1.)
parser.add_argument('--degradation', help='0 means MI', type=float, 
                    default=0.2)
parser.add_argument('--max_rew', help='Maximum reward', type=float, 
                    default=1.)
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
logstd_init = torch.log(torch.zeros(1) + args.std_init)
policy = ShallowGibbsPolicy(env, 
                            temp=1.)

envname = re.sub(r'[^a-zA-Z]', "", args.env)[:-1]
envname = re.sub(r'[^a-zA-Z]', "", args.env)[:-1].lower()
logname = envname + '_' + args.name + '_' + str(args.seed)

if args.temp:
    logger = Logger(directory= args.storage + '/temp', name = logname, modes=['human', 'csv'])
else:
    logger = Logger(directory=args.storage + '/logs', name = logname, modes=['human', 'csv'])

#Constants
lip_const = gibbs_lip_const(args.max_feat, args.max_rew, args.disc, 
                            1.)
print(lip_const)
score_bound = 2 * args.max_feat
err_bound = emp_bernstein(args.max_rew, score_bound, args.disc, args.horizon, 
                            dim=policy.num_params(), estimator=args.estimator)


# Run
relaxed_spg(env, policy, args.horizon, lip_const, err_bound, args.max_rew,
            empirical = True,
            fail_prob = 1. - args.conf,
            mini_batchsize = args.mini_batchsize,
            max_batchsize = args.max_batchsize,
            max_samples = args.max_samples,
            disc = args.disc,
            seed = args.seed,
            logger = logger,
            render = args.render,
            shallow = True,
            estimator = args.estimator,
            baseline = args.baseline,
            log_params=False,
            save_params=False,
            degradation=args.degradation) #degradation should be close to zero
