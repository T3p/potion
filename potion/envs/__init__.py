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
    id='LQ-v0',
    entry_point='potion.envs.lq:LQ'
)

register(
    id='Drone-v0',
    entry_point='potion.envs.drone:Drone'
)

register(
    id='DroneCrash-v0',
    entry_point='potion.envs.drone_crash:DroneCrash'
)

register(
    id='MiniGolf-v0',
    entry_point='potion.envs.minigolf:MiniGolf'
)

register(
    id='GridWorld-v0',
    entry_point='potion.envs.gridworld:GridWorld'
)

register(
    id='TwoGoals-v0',
    entry_point='potion.envs.twogoals:TwoGoals'
)


register(
    id='PitWorld-v0',
    entry_point='potion.envs.pitworld:PitWorld'
)



register(
    id='LQG1D-v0',
    entry_point='potion.envs.lqg1d:LQG1D'
)

register(
    id='lqr1d-v0',
    entry_point='potion.envs.lqr1d:lqr1d'
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

register(
    id='Corridor-v0',
    entry_point='potion.envs.corridor:Corridor'
)