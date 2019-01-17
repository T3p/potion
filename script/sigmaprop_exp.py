#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 14:47:33 2019

@author: Matteo Papini
"""
import torch
import gym
import potion.envs
from potion.meta.steppers import ConstantStepper, RMSprop
from potion.actors.continuous_policies import SimpleGaussianPolicy as Gauss
from potion.common.logger import Logger
from potion.algorithms.metaexplore import sigmaprop
from potion.common.misc_utils import clip
import argparse
import re

# Command line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--seed', help='RNG seed', type=int, default=0)
parser.add_argument('--explore', help='Kind of exploration', type=str, default='balanced')
parser.add_argument('--env', help='Gym environment id', type=str, default='ContCartPole-v0')
parser.add_argument('--alpha', help='Step size', type=float, default=1e-2)
parser.add_argument('--horizon', help='Task horizon', type=int, default=300)
parser.add_argument('--batchsize', help='Batch size', type=int, default=100)
parser.add_argument('--iterations', help='Task horizon', type=int, default=300)
parser.add_argument('--gamma', help='Discount factor', type=float, default=0.99)
parser.add_argument('--saveon', help='How often to save parameters', type=int, default=100)
parser.add_argument('--sigmainit', help='Initial policy std', type=float, default=1.)
parser.add_argument('--stepper', help='Step size rule', type=str, default='rmsprop')
parser.add_argument("--render", help="Render an episode",
                    action="store_true")
parser.add_argument("--no-render", help="Do not render any episode",
                    action="store_false")
parser.add_argument("--trial", help="Save logs in temp folder",
                    action="store_true")
parser.add_argument("--no-trial", help="Save logs in logs folder",
                    action="store_false")
parser.set_defaults(render=False, trial=False) 

args = parser.parse_args()

# Prepare
env = gym.make(args.env)
env.seed(args.seed)

m = sum(env.observation_space.shape)
d = sum(env.action_space.shape)
mu_init = torch.zeros(m)
logstd_init = torch.log(torch.zeros(1) + args.sigmainit)
policy = Gauss(m, d, mu_init=mu_init, logstd_init=logstd_init, learn_std=True)

if args.stepper == 'rmsprop':
    stepper = RMSprop(alpha = args.alpha)
else:
    stepper = ConstantStepper(args.alpha)

envname = re.sub(r'[^a-zA-Z]', "", args.env)[:-1]

if args.trial:
    logger = Logger(directory='../temp', name = envname + '_' + str(args.seed))
else:
    logger = Logger(directory='../logs', name = envname + '_' + '_' + str(args.seed))
    
# Run
sigmaprop(env,
            policy,
            horizon = args.horizon,
            batchsize = args.batchsize,
            iterations = args.iterations,
            gamma = args.gamma,
            stepper = stepper,
            seed = args.seed,
            action_filter = clip(env),
            logger = logger,
            save_params = args.saveon,
            render = args.render)