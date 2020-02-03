#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 14:47:33 2019

@author: Matteo Papini
"""
import torch
import gym
import potion.envs
from potion.actors.continuous_deterministic_policies import ShallowDeterministicPolicy, DeepDeterministicPolicy
from potion.actors.hyperpolicies import GaussianHyperpolicy
from potion.common.logger import Logger
from potion.algorithms.pgpe import pgpe
import argparse
import re
from potion.meta.steppers import ConstantStepper, RMSprop, Adam
from gym.spaces.discrete import Discrete
from potion.actors.feature_functions import rbf_fun
import safety_envs

# Command line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--name', help='Experiment name', type=str, default='PGPE')
parser.add_argument('--baseline', help='baseline for policy gradient estimator (avg/sugiyama/peters/zero)', type=str, default='peters')
parser.add_argument('--seed', help='RNG seed', type=int, default=0)
parser.add_argument('--env', help='Gym environment id', type=str, default='FaultyReach-v0')
parser.add_argument('--horizon', help='Task horizon', type=int, default=1000)
parser.add_argument('--batchsize', help='Initial batch size', type=int, default=100)
parser.add_argument('--iterations', help='Iterations', type=int, default=500)
parser.add_argument('--disc', help='Discount factor', type=float, default=1.)
parser.add_argument('--std_init', help='Initial policy std', type=float, default=1.)
parser.add_argument('--stepper', help='Step size rule', type=str, default='constant')
parser.add_argument('--step', help='Step size', type=float, default=1e-2)
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
parser.add_argument("--tanh", help="Apply tanh to action",
                    action="store_true")
parser.add_argument("--no-tanh", help="Apply tanh to action",
                    action="store_false")
parser.add_argument("--neural", help="Apply tanh to action",
                    action="store_true")
parser.add_argument("--no-neural", help="Apply tanh to action",
                    action="store_false")
parser.set_defaults(render=False, temp=False, learnstd=True, natural=False, bias=False, tanh=False, neural=False) 

args = parser.parse_args()

# Prepare

env = gym.make(args.env)
env.seed(args.seed)
env.sigma_noise = 0

m = sum(env.observation_space.shape)
d = sum(env.action_space.shape)
squash = None
if not args.neural:
    if args.tanh:
        squash = torch.tanh
    policy = ShallowDeterministicPolicy(m, d, squash_fun=squash)
else:
    policy = DeepDeterministicPolicy(m, d, [4])
original_params = torch.tensor([-1.5469e+00, -2.0167e+00, -7.6784e-01, -7.9975e-01, -2.0564e+00,
    -3.2041e-01, -1.9130e-01, -7.9075e-01, -4.7737e-01, -9.9748e-01,
    -4.1719e-01, -6.2246e-01,  9.2992e-01, -9.7951e-01, -1.6872e-01,
    -4.4438e-02, -6.2452e-01, -3.1517e-01,  1.1200e-01,  3.8586e-02,
     4.2222e+00,  1.0748e-01,  6.8834e-03,  1.2165e-01,  1.1850e-02,
    -1.2723e-01, -4.6684e-02,  1.3817e-02,  2.9296e-01,  1.1796e-01,
    -1.4662e+01,  1.1247e+00, -8.5989e-02, -1.7306e-01, -5.6538e-04,
    -3.1667e-01])
mu_init = original_params[18:]#torch.zeros(policy.num_params())
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
            natural = True,
            seed = args.seed,
            logger = logger,
            render = args.render,
            baseline = args.baseline,
            log_params=True)
