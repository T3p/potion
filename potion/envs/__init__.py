#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 15:00:16 2019

@author: matteo
"""

from gym.envs.registration import register
from gym import spaces

register(
    id='LQG1D-v0',
    entry_point='potion.envs.lqg1d:LQG1D'
)

register(
    id='lqr1d-v0',
    entry_point='potion.envs.lqr1d:lqr1d'
)

register(
    id='LQ-v0',
    entry_point='potion.envs.lq:LQ'
)

register(
    id='mass-v0',
    entry_point='potion.envs.mass:mass'
)

register(
    id='ContCartPole-v0',
    entry_point='potion.envs.cartpole:ContCartPole'
)

register(
    id='GridWorld-v0',
    entry_point='potion.envs.gridworld:GridWorld'
)

register(
    id='MiniGolf-v0',
    entry_point='potion.envs.minigolf:MiniGolf'
)

register(
    id='ComplexMiniGolf-v0',
    entry_point='potion.envs.minigolf:ComplexMiniGolf'
)
