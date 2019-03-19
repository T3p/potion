#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 14:21:40 2019

@author: matteo
"""

def reinforce_var_bound(max_rew, disc, smooth_const, horizon):
    var_bound = horizon * smooth_const * max_rew**2 * \
        (1 - disc**horizon)**2 / (1 - disc)**2
    return var_bound

def gpomdp_var_bound(max_rew, disc, smooth_const, horizon=None):
    var_bound = smooth_const * max_rew**2 / (1 - disc)**3
    if horizon is None:
        return var_bound
    else:
        return var_bound * (1 - disc**horizon) * (
                1 - disc**horizon * (1 - disc**horizon) * horizon 
                - disc**horizon)