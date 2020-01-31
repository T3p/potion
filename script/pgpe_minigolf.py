#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 14:47:33 2019

@author: Matteo Papini
"""
import torch
import gym
import potion.envs
from potion.actors.continuous_deterministic_policies import ShallowDeterministicPolicy
from potion.actors.hyperpolicies import GaussianHyperpolicy
from potion.common.logger import Logger
from potion.algorithms.pgpe import pgpe
import argparse
import re
from potion.meta.steppers import ConstantStepper, RMSprop, Adam
from gym.spaces.discrete import Discrete
from potion.actors.feature_functions import rbf_fun

# Command line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--name', help='Experiment name', type=str, default='PGPE')
parser.add_argument('--baseline', help='baseline for policy gradient estimator (avg/sugiyama/peters/zero)', type=str, default='peters')
parser.add_argument('--seed', help='RNG seed', type=int, default=0)
parser.add_argument('--env', help='Gym environment id', type=str, default='MiniGolf-v0')
parser.add_argument('--horizon', help='Task horizon', type=int, default=20)
parser.add_argument('--batchsize', help='Initial batch size', type=int, default=500)
parser.add_argument('--iterations', help='Iterations', type=int, default=300)
parser.add_argument('--disc', help='Discount factor', type=float, default=0.99)
parser.add_argument('--std_init', help='Initial policy std', type=float, default=1.)
parser.add_argument('--stepper', help='Step size rule', type=str, default='constant')
parser.add_argument('--step', help='Step size', type=float, default=1e-1)
parser.add_argument("--render", help="Render an episode",
                    action="store_true")
parser.add_argument("--no-render", help="Do not render any episode",
                    action="store_false")
parser.add_argument("--temp", help="Save logs in temp folder",
                    action="store_true")
parser.add_argument("--no-temp", help="Save logs in logs folder",
                    action="store_false")
parser.add_argument("--natural", help="Use natural gradient",
                    action="store_true")
parser.add_argument("--no-natural", help="Use vanilla gradient",
                    action="store_false")
parser.add_argument("--learnstd", help="Learn std",
                    action="store_true")
parser.add_argument("--no-learnstd", help="Don't learn std",
                    action="store_false")
parser.add_argument("--bias", help="Use bias parameter",
                    action="store_true")
parser.add_argument("--no-bias", help="Use bias parameter",
                    action="store_false")
parser.set_defaults(render=False, temp=False, learnstd=True, natural=False, bias=False) 

args = parser.parse_args()

# Prepare

env = gym.make(args.env)
env.seed(args.seed)
env.sigma_noise = 0

m = sum(env.observation_space.shape)
d = sum(env.action_space.shape)
rbf = rbf_fun([torch.tensor(4.), torch.tensor(8.), torch.tensor(12.), torch.tensor(16.)], 
                [torch.tensor(4.)] * 4)
policy = ShallowDeterministicPolicy(m, d, feature_fun=rbf)
mu_init = torch.zeros(policy.num_params())
logstd_init = torch.log(torch.zeros(policy.num_params()) + args.std_init)
hyperpolicy = GaussianHyperpolicy(policy, 
                           learn_std=True,
                           mu_init=mu_init,
                           logstd_init=logstd_init,
                           bias=args.bias)

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
pgpe(env, hyperpolicy,
            horizon = args.horizon,
            stepper = stepper,
            batchsize = args.batchsize,
            iterations = args.iterations,
            disc = args.disc,
            natural = args.natural,
            seed = args.seed,
            logger = logger,
            render = args.render,
            test_batchsize = 100,
            baseline = args.baseline,
            log_params=True)