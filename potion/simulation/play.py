#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Matteo Papini
"""

from potion.simulation.trajectory_generators import generate_batch
from potion.common.misc_utils import clip

def play(env, policy, horizon=100, episodes=100, render=True, action_filter=None):
    if action_filter is None:
        action_filter = clip(env)
    return generate_batch(env, policy, horizon, episodes, render=render, action_filter=action_filter)
