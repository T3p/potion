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
import safety_envs
import numpy as np

# Command line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--name', help='Experiment name', type=str, default='PGPE')
parser.add_argument('--baseline', help='baseline for policy gradient estimator (avg/sugiyama/peters/zero)', type=str, default='peters')
parser.add_argument('--seed', help='RNG seed', type=int, default=0)
parser.add_argument('--env', help='Gym environment id', type=str, default='ContCartPole-v0')
parser.add_argument('--horizon', help='Task horizon', type=int, default=100)
parser.add_argument('--batchsize', help='Initial batch size', type=int, default=100)
parser.add_argument('--iterations', help='Iterations', type=int, default=500)
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

optimal = [0.10787985473871231, 0.02179303579032421, 4.300711154937744, 0.10839951038360596,
                  0.017089104279875755, 0.1119314506649971, 0.018646063283085823, -0.17877089977264404,
                  -0.03759196400642395, -0.004248579498380423, 0.48613205552101135, 0.10498402267694473,
                  -12.068914413452148, 1.0702580213546753, -0.04661020636558533, -0.22232159972190857,
                  0.0361342579126358, -0.39843615889549255]

env = gym.make(args.env)
env.seed(args.seed)

m = sum(env.observation_space.shape)
d = sum(env.action_space.shape)

feat = None
if 'Minigolf' in args.env:
    def feat(s):
        sigma = 4.
        centers = [4., 8., 12., 16.]
        res = [np.exp(-1 / (2 * sigma ** 2) * (s - c) ** 2) for c in centers]
        cat_dim = len(s.shape)
        res = torch.cat(res, cat_dim - 1)
        return res
    
policy = ShallowDeterministicPolicy(m, d, feature_fun=feat)
mu_init = torch.zeros(policy.num_params())
if 'Minigolf' in args.env:
    mu_init = torch.ones(policy.num_params())
elif 'DoubleIntegrator' in args.env:
    mu_init = torch.ones(policy.num_params()) * -0.3
elif 'Reach' in args.env:
    mu_init = torch.tensor(optimal[9:])
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
            test_batchsize = False,
            baseline = args.baseline,
            log_params=True)