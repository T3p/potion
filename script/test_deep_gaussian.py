#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 14:47:33 2019

@author: Matteo Papini
"""
import torch
import gym
import potion.envs
from potion.actors.continuous_policies import DeepGaussianPolicy
from potion.common.logger import Logger
from potion.algorithms.reinforce import reinforce
from potion.algorithms.variance_reduced import svrpg, srvrpg, stormpg, pagepg
import argparse
import re
from potion.meta.steppers import ConstantStepper, RMSprop, Adam
from gym.spaces.discrete import Discrete
import numpy as np
from functools import partial

# Command line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--name', help='Experiment name', type=str, default='DEEP_TEST')
parser.add_argument('--estimator', help='Policy gradient estimator (reinforce/gpomdp)', type=str, default='gpomdp')
parser.add_argument('--baseline', help='baseline for policy gradient estimator (avg/peters/zero)', type=str, default='avg')
parser.add_argument('--seed', help='RNG seed', type=int, default=0)
parser.add_argument('--env', help='Gym environment id', type=str, default='Swimmer-v3')
parser.add_argument('--horizon', help='Task horizon', type=int, default=500)
parser.add_argument('--batchsize', help='Initial batch size', type=int, default=10)
parser.add_argument('--network', help='Neural network size as space-separated integers', type=str, default="32 32")
parser.add_argument('--activation', help='Neural network activation function', type=str, default="tanh")
parser.add_argument('--iterations', help='Iterations', type=int, default=200)
parser.add_argument('--disc', help='Discount factor', type=float, default=0.995)
parser.add_argument('--std_init', help='Initial policy std', type=float, default=1.)
parser.add_argument('--stepper', help='Step size rule', type=str, default='adam')
parser.add_argument('--step', help='Step size', type=float, default=1e-3)
parser.add_argument('--ent', help='Entropy bonus coefficient', type=float, default=0.)
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
parser.add_argument("--learnstd", help="Learn std",
                    action="store_true")
parser.add_argument("--no-learnstd", help="Don't learn std",
                    action="store_false")
parser.set_defaults(render=False, temp=False, learnstd=True, test=False) 

args = parser.parse_args()

# Prepare

env = gym.make(args.env)
env.seed(args.seed)

hidden_neurons = [int(x) for x in args.network.split(" ")] if args.network else []
if args.activation=="tanh":
    activation = torch.tanh
elif args.activation=="relu":
    activation = torch.relu
else:
    raise NotImplementedError("Only available activation functions are tanh and relu")
    
m = sum(env.observation_space.shape)
d = sum(env.action_space.shape)
mu_init = torch.zeros(m*d)
logstd_init = torch.log(torch.zeros(d) + args.std_init)

policy = DeepGaussianPolicy(m, d,
                           hidden_neurons=hidden_neurons,
                           activation=activation,
                           state_preproc=None,
                           mu_init=None, #random initialization 
                           logstd_init=logstd_init, 
                           learn_std=args.learnstd,
                           action_range=1.,
                           )#init=partial(torch.nn.init.constant_))

test_batchsize = args.batchsize if args.test else 0

envname = re.sub(r'[^a-zA-Z]', "", args.env)[:-1]
envname = re.sub(r'[^a-zA-Z]', "", args.env)[:-1].lower()
logname = envname + '_' + args.name + '_' + str(args.seed)

if args.temp:
    logger = Logger(directory='../temp', name = logname)
else:
    logger = Logger(directory='../logs', name = logname)

if args.stepper == 'rmsprop':
    stepper = RMSprop()
elif args.stepper == 'adam':
    stepper = Adam(alpha=args.step)
else:
    stepper = ConstantStepper(args.step)


# Run
"""
reinforce(env, policy,
            action_filter = None,#lambda a: np.array((a,)),
            horizon = args.horizon,
            stepper = stepper,
            batchsize = args.batchsize,
            iterations = args.iterations,
            disc = args.disc,
            entropy_coeff = args.ent,
            seed = args.seed,
            logger = logger,
            render = args.render,
            shallow = False,
            estimator = args.estimator,
            baseline = args.baseline,
            test_batchsize=test_batchsize,
            log_params=False)
"""
stormpg(env, policy,
            action_filter = None,#lambda a: np.array((a,)),
            horizon = args.horizon,
            stepper = stepper,
            iterations = args.iterations,
            disc = args.disc,
            seed = args.seed,
            logger = logger,
            render = args.render,
            shallow = False,
            estimator = args.estimator,
            baseline = args.baseline,
            test_batchsize=test_batchsize,
            log_params=False)
#"""