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
from potion.algorithms.sepg import sepg, naive_sepg
import argparse
import re
from potion.meta.safety_requirements import MonotonicImprovement, Budget, FixedThreshold
from potion.simulation.trajectory_generators import generate_batch
from potion.common.misc_utils import performance
from potion.actors.feature_functions import rbf_fun


#Command line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--name', help='Experiment name', type=str, default='SEPG')
parser.add_argument('--seed', help='RNG seed', type=int, default=0)
parser.add_argument('--env', help='Gym environment id', type=str, default='MiniGolf-v0')
parser.add_argument('--max_rew', help='Max reward', type=float, default=100.)
parser.add_argument('--max_feat', help='Max state feature', type=float, default=1.)
parser.add_argument('--horizon', help='Task horizon', type=int, default=20)
parser.add_argument('--batchsize', help='Batch size', type=int, default=100)
parser.add_argument('--iterations', help='Iterations', type=int, default=5000)
parser.add_argument('--safety', help='Safety creterion', type=str, default='-10')
parser.add_argument('--gamma', help='Discount factor', type=float, default=0.95)
parser.add_argument('--delta', help='Confidence parameter', type=float, default=1.)
parser.add_argument('--std_init', help='Initial policy std', type=float, default=0.1)
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
parser.add_argument("--naive", help="Do not use metagradient",
                    action="store_true")
parser.add_argument("--no-naive", help="Use metagradient",
                    action="store_false")
parser.set_defaults(render=False, temp=False, test=False, naive=False) 

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

if args.safety == 'mi':
    safety_req = MonotonicImprovement(0.)
elif args.safety == 'budget':
    from potion.common.misc_utils import seed_all_agent
    seed_all_agent(args.seed)
    env.seed(args.seed)
    batch = generate_batch(env, policy, args.horizon, args.batchsize)
    perf = performance(batch, args.gamma)
    safety_req = Budget(initial_perf = perf)
else:
    safety_req = FixedThreshold(threshold = float(args.safety))

env.seed(args.seed)

test_batchsize = 100 if args.test else 0

envname = re.sub(r'[^a-zA-Z]', "", args.env)[:-1]
envname = re.sub(r'[^a-zA-Z]', "", args.env)[:-1].lower()
logname = envname + '_' + args.name + '_' + str(args.seed)

if args.temp:
    logger = Logger(directory='../temp', name = logname)
else:
    logger = Logger(directory='../logs', name = logname)
    
#Run
algo = naive_sepg if args.naive else sepg
algo(env, policy,
            horizon = args.horizon,
            batchsize = args.batchsize,
            iterations = args.iterations,
            disc = args.gamma,
            safety_req = safety_req,
            max_feat = args.max_feat,
            max_rew = args.max_rew,
            conf = args.delta,
            seed = args.seed,
            logger = logger,
            render = args.render,
            test_batchsize=test_batchsize)
