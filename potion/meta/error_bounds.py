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
    
    def _err_bound(fail_prob, dummy):
        return 2 * max_rew * score_bound * time_factor * math.sqrt(2 * dim * 
               math.log(6 / fail_prob))
    
    return _err_bound

def hoeffding_sg_score(max_rew, score_sg, disc, horizon, dim, 
                            estimator='gpomdp'):
    if estimator=='reinforce':
        time_factor = horizon * (1 - disc**horizon) / (1 - disc)
    elif estimator == 'gpomdp':
        time_factor = (1 - disc**horizon - horizon * (disc**horizon - 
                        disc**(horizon + 1))) / (1 - disc)**2
    
    def _err_bound(fail_prob, dummy):
        return 4 * max_rew * score_sg * time_factor * math.sqrt(14 * dim * 
               math.log(6 / fail_prob))
    
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
    
def emp_bernstein(max_rew, score_bound, disc, horizon, dim, 
                            estimator='gpomdp'):
    if estimator=='reinforce':
        time_factor = horizon * (1 - disc**horizon) / (1 - disc)
    elif estimator == 'gpomdp':
        time_factor = (1 - disc**horizon - horizon * (disc**horizon - 
                        disc**(horizon + 1))) / (1 - disc)**2
    
    def _err_bound(fail_prob, sample_var, batchsize):
        return math.sqrt(2. * sample_var * math.log(2. / fail_prob) / batchsize) \
            + 7. * time_factor * math.log(2. / fail_prob) / (3. * (batchsize - 1))
    return _err_bound
    
