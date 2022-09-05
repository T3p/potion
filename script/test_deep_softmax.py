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
from potion.actors.discrete_policies import ShallowGibbsPolicy, DeepGibbsPolicy
from potion.common.logger import Logger
from potion.algorithms.reinforce import reinforce
from potion.algorithms.variance_reduced import stormpg, pagepg, svrpg, srvrpg
import argparse
import re
from potion.meta.steppers import ConstantStepper, RMSprop, Adam
from gym.spaces.discrete import Discrete
from functools import partial


# Command line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--name', help='Experiment name', type=str, default='TEST_DEEP_SOFTMAX')
parser.add_argument('--estimator', help='Policy gradient estimator (reinforce/gpomdp)', type=str, default='gpomdp')
parser.add_argument('--baseline', help='baseline for policy gradient estimator (avg/peters/zero)', type=str, default='avg')
parser.add_argument('--seed', help='RNG seed', type=int, default=0)
parser.add_argument('--env', help='Gym environment id', type=str, default='CartPole-v1')
parser.add_argument('--horizon', help='Task horizon', type=int, default=200)
parser.add_argument('--batchsize', help='Initial batch size', type=int, default=100)
parser.add_argument('--network', help='Neural network size as space-separated integers', type=str, default="32 32")
parser.add_argument('--iterations', help='Iterations', type=int, default=1000)
parser.add_argument('--disc', help='Discount factor', type=float, default=0.9999)
parser.add_argument('--tmp', help='Policy temperature', type=float, default=1.)
parser.add_argument('--stepper', help='Step size rule', type=str, default='constant')
parser.add_argument('--step', help='Step size', type=float, default=1e-4)
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
parser.add_argument("--log_params", help="Print policy parameters",
                    action="store_true")
parser.add_argument("--no-log_params", help="Do not print policy parameters",
                    action="store_false")
parser.set_defaults(render=False, temp=False, test=False, log_params=False) 

args = parser.parse_args()

# Prepare

env = gym.make(args.env)
env.seed(args.seed)

assert type(env.action_space) is Discrete

state_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

hidden_neurons = [int(x) for x in args.network.split(" ")] if args.network else []
pref_init = torch.zeros(state_dim * n_actions) if hidden_neurons == [] else None

policy = DeepGibbsPolicy(state_dim,
                         n_actions,
                         hidden_neurons=hidden_neurons,
                         temp=args.tmp,
                         pref_init=pref_init,
                         )#init=partial(torch.nn.init.constant_, val=0.))

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
            log_params=args.log_params)
"""
stormpg(env, policy,
            horizon = args.horizon,
            stepper = stepper,
            init_batchsize = args.batchsize,
            mini_batchsize = args.batchsize // 10,         
            iterations = args.iterations,
            disc = args.disc,
            seed = args.seed,
            logger = logger,
            render = args.render,
            shallow = False,
            estimator = args.estimator,
            baseline = args.baseline,
            test_batchsize=test_batchsize,
            log_params=args.log_params)
#"""