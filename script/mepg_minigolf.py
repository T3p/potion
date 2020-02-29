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
from potion.algorithms.mepg import mepg
import argparse
import re
from potion.actors.feature_functions import rbf_fun


#Command line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--name', help='Experiment name', type=str, default='MEPG')
parser.add_argument('--seed', help='RNG seed', type=int, default=0)
parser.add_argument('--env', help='Gym environment id', type=str, default='MiniGolf-v0')
parser.add_argument('--alpha', help='Step size', type=float, default=0.01)
parser.add_argument('--eta', help='Meta step size', type=float, default=0.01)
parser.add_argument('--horizon', help='Task horizon', type=int, default=20)
parser.add_argument('--batchsize', help='Batch size', type=int, default=100)
parser.add_argument('--iterations', help='Iterations', type=int, default=5000)
parser.add_argument('--gamma', help='Discount factor', type=float, default=0.95)
parser.add_argument('--std_init', help='Initial policy std', type=float, default=0.1)
parser.add_argument('--ablation', help='What MEPG term to remove (0 means none)', type=int, default=0)
parser.add_argument('--env_noise', help='Environment noise', type=float, default=0.3)
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
parser.set_defaults(render=False, temp=False, test=False) 

args = parser.parse_args()

#Prepare
env = gym.make(args.env)
env.seed(args.seed)
env.sigma_noise = args.env_noise

m = sum(env.observation_space.shape)
d = sum(env.action_space.shape)
logstd_init = torch.log(torch.zeros(d) + args.std_init)

feat = rbf_fun([torch.tensor(4.), torch.tensor(8.), torch.tensor(12.), torch.tensor(16.)], [torch.tensor(4.)] * 4)
num_params = sum(feat(torch.ones(m)).shape)
mu_init = torch.ones(num_params)
policy = ShallowGaussianPolicy(m, d, 
                           mu_init=mu_init,
                           logstd_init=logstd_init, 
                           learn_std=True,
                           feature_fun=feat)

test_batchsize = 100 if args.test else 0

envname = re.sub(r'[^a-zA-Z]', "", args.env)[:-1]
envname = re.sub(r'[^a-zA-Z]', "", args.env)[:-1].lower()
logname = envname + '_' + args.name + '_' + str(args.seed)

if args.temp:
    logger = Logger(directory='../temp', name = logname)
else:
    logger = Logger(directory='../logs', name = logname)
    
#Run
mepg(env, policy,
            horizon = args.horizon,
            batchsize = args.batchsize,
            iterations = args.iterations,
            disc = args.gamma,
            alpha = args.alpha,
            eta = args.eta,
            seed = args.seed,
            logger = logger,
            render = args.render,
            test_batchsize=test_batchsize,
            ablation = args.ablation)
