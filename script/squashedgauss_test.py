#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 14:47:33 2019

@author: Matteo Papini
"""
import torch
import gym
import potion.envs
from potion.actors.continuous_policies import ShallowSquashedPolicy, DeepSquashedPolicy
from potion.actors.discrete_policies import ShallowGibbsPolicy
from potion.common.logger import Logger
from potion.algorithms.reinforce import reinforce
import argparse
import re
from potion.meta.steppers import ConstantStepper, RMSprop, Adam
from gym.spaces.discrete import Discrete

# Command line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--name', help='Experiment name', type=str, default='deep_test')
parser.add_argument('--estimator', help='Policy gradient estimator (reinforce/gpomdp)', type=str, default='gpomdp')
parser.add_argument('--baseline', help='baseline for policy gradient estimator (avg/peters/zero)', type=str, default='avg')
parser.add_argument('--seed', help='RNG seed', type=int, default=0)
parser.add_argument('--env', help='Gym environment id', type=str, default='ContCartPole-v0')
parser.add_argument('--horizon', help='Task horizon', type=int, default=200)
parser.add_argument('--batchsize', help='Initial batch size', type=int, default=100)
parser.add_argument('--iterations', help='Iterations', type=int, default=1000)
parser.add_argument('--disc', help='Discount factor', type=float, default=0.99)
parser.add_argument('--std_init', help='Initial policy std', type=float, default=1.)
parser.add_argument('--stepper', help='Step size rule', type=str, default='rmsprop')
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

if type(env.action_space) is Discrete:
    policy = ShallowGibbsPolicy(env, 
                                temp=1.)
else:
    m = sum(env.observation_space.shape)
    d = sum(env.action_space.shape)
    logstd_init = torch.log(torch.zeros(d) + args.std_init)
    """
    policy = ShallowSquashedPolicy(m, d,
                               scale = 10.,
                               mu_init=torch.zeros(m*d),
                               logstd_init=logstd_init, 
                               learn_std=args.learnstd)
    shallow = True
    """
    policy = DeepSquashedPolicy(m, d, [],
                           scale = 10.,
                           logstd_init=logstd_init, 
                           learn_std=args.learnstd)
    shallow = False
    #"""
    
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
            shallow = shallow,
            estimator = args.estimator,
            baseline = args.baseline,
            test_batchsize=test_batchsize,
            log_params=True)