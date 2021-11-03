#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 14:21:40 2019
"""

import math

def hoeffding_bounded_score(max_rew, score_bound, disc, horizon, dim, 
                            estimator='gpomdp'):
    if estimator=='reinforce':
        time_factor = horizon * (1 - disc**horizon) / (1 - disc)
    elif estimator == 'gpomdp':
        time_factor = (1 - disc**horizon - horizon * (disc**horizon - 
                        disc**(horizon + 1))) / (1 - disc)**2
    
    def _err_bound(fail_prob, batch_size):
        return max_rew * score_bound * time_factor * math.sqrt(2 * dim * 
               math.log(2 * dim / fail_prob))
    
    return _err_bound

def hoeffding_sg_score(max_rew, score_sg, disc, horizon, dim, 
                            estimator='gpomdp'):
    if estimator=='reinforce':
        time_factor = horizon * (1 - disc**horizon) / (1 - disc)
    elif estimator == 'gpomdp':
        time_factor = (1 - disc**horizon - horizon * (disc**horizon - 
                        disc**(horizon + 1))) / (1 - disc)**2
    
    def _err_bound(fail_prob, batch_size):
        return max_rew * score_sg * time_factor * math.sqrt(4 * dim * 
               math.log(2 * dim * batch_size * horizon / fail_prob) * 
               math.log(2 * dim / fail_prob))
    
    return _err_bound


def gauss_gradient_range(max_rew, max_feat, disc, horizon, max_action, std,
                         estimator='gpomdp'):
    if estimator == 'reinforce':
        return (2 * horizon * (1 - disc**horizon) * max_feat * max_action 
                * max_rew / (std**2 * (1 - disc)))
    else:
        return (2 * max_action * max_feat * max_rew 
                * ((horizon * disc**(horizon + 1) 
                - (horizon + 1) * disc**horizon + 1) / (1 - disc)**2) 
                / std**2)
    
