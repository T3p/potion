#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 23:12:09 2019

@author: Matteo Papini
"""
import numpy as np

def clip(env):
    def action_filter(a):
        return np.clip(a, env.action_space.low, env.action_space.high)
    return lambda a : action_filter(a)
