#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 15:00:16 2019

@author: matteo
"""

from gym.envs.registration import register
from gym import spaces

register(
    id='ContCartPole-v0',
    entry_point='potion.envs.cartpole:ContCartPole'
)

register(
    id='LQG1D-v0',
    entry_point='potion.envs.lqg1d:LQG1D'
)

register(
    id='ContAcrobot-v0',
    entry_point='potion.envs.acrobot:ContAcrobot'
)