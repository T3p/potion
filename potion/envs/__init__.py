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
    id='LQGX-v0',
    entry_point='potion.envs.lqgx:LQGX'
)


register(
    id='Hole-v0',
    entry_point='potion.envs.hole:Hole'
)

register(
    id='LQG2D-v0',
    entry_point='potion.envs.lqg2d:LQG2D'
)

register(
    id='Pit-v0',
    entry_point='potion.envs.pit:Pit'
)

register(
    id='Wall-v0',
    entry_point='potion.envs.wall:Wall'
)

register(
    id='Mass-v0',
    entry_point='potion.envs.mass:Mass'
)

register(
    id='Robot-v0',
    entry_point='potion.envs.robot:Robot'
)