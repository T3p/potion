#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Matteo Papini
"""

from potion.simulation.trajectory_generators import generate_batch

def play(env, policy, horizon=100, episodes=100):
    generate_batch(env, policy, horizon, episodes, render=True)
