#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 15:00:16 2019

@author: matteo
"""

from gymnasium.envs.registration import register

register(
    id='LQR-v0',
    entry_point='potion.envs.lqr:LQR'
)

register(
    id='CartPoleContinuous-v0',
    entry_point='potion.envs.cartpole_continuous:CartPoleContinuous'
)

register(
    id='GridWorld-v0',
    entry_point='potion.envs.gridworld:GridWorld'
)

register(
    id='Minigolf-v0',
    entry_point='potion.envs.minigolf:Minigolf'
)
