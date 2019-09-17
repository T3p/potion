#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 11:46:59 2019

@author: matteo
"""

from baselines.trpo_mpi.trpo_custom import learn as trpo
#from baselines.trpo_mpi.trpo_mpi import learn as trpo
from baselines.common.models import mlp
import potion.envs
import gym
import gym.spaces
import tensorflow as tf
import baselines.logger as logger
import time

#env = gym.make('Drone-v0')
#env = gym.make('MiniGolf-v0')
env = gym.make('DroneCrash-v0')
logger.configure(dir='/home/matteo/policy-optimization/logs', format_strs=['stdout', 'csv', 'tensorboard'], log_suffix='trpo_' + str(int(time.time())))
trpo(#network='mlp',
     env=env,
     total_timesteps=100*100*200,
     timesteps_per_batch=100*100)
