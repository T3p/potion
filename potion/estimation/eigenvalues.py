#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 14:01:34 2019

@author: matteo
"""
import torch
import math
from potion.estimation.gradients import gpomdp_estimator
from potion.estimation.importance_sampling import importance_weights
from potion.simulation.trajectory_generators import generate_batch

def power(policy, batch, grad, disc, pow_alpha=0.01, err_tol=0.1, max_it=100, max_attempts=3, estimator=gpomdp_estimator, baseline='peters', shallow=True, clip=0.2, verbose=True):
    params = policy.get_flat()
    
    err = 999
    attempts = 0
    while err > err_tol and attempts < max_attempts:
        pow_it = 0
        psi = torch.rand_like(grad)
        _lip_const = torch.norm(psi).item()
        while err > err_tol and pow_it < max_it:
            params_2 = params + pow_alpha * psi / torch.norm(psi)
            policy.set_from_flat(params_2)
            grad_2_samples = estimator(batch, disc, policy, 
                                      baselinekind=baseline,
                                      shallow=shallow,
                                      result='samples')
            policy.set_from_flat(params)
            iws = importance_weights(batch, policy, params_2)
            clipped_iws = torch.clamp(iws, 1-clip, 1+clip)
            grad_2 = torch.mean(grad_2_samples * iws.unsqueeze(1))
            clipped_grad_2 = torch.mean(grad_2_samples * clipped_iws.unsqueeze(1))
            grad_2 = max(grad_2, clipped_grad_2)
            psi = 1. / pow_alpha * (grad_2 - grad)
            lip_const = torch.norm(psi).item()
            if math.isnan(lip_const):
                err = 999
                break
            err = abs(lip_const - _lip_const) / lip_const
            _lip_const = lip_const
            pow_it += 1
        policy.set_from_flat(params)
        if err <= err_tol and verbose:
            print('Converged in %d iterations after %d failed attempts (alpha = %f, error = %f)' % (pow_it, attempts, pow_alpha, err))
        else:
            pow_alpha /= 10
        attempts += 1
        
    if err > err_tol and verbose:
        print('Failed to converge (%d failed attempts, alpha = %f, error = %f)' % (attempts, pow_alpha, err))
    return lip_const

def power_op(policy, std, disc, horizon, minibatchsize, env, pow_alpha=0.01, err_tol=0.1, max_pow_it=100, max_attempts=3, estimator=gpomdp_estimator, baseline='peters', shallow=True, forget=0.1):
    params = policy.get_flat()
    
    err = 999
    attempts = 0
    while err > err_tol and attempts < max_attempts:
        pow_it = 0
        psi = torch.rand_like(params)
        _lip_const = torch.norm(psi).item()
        while err > err_tol and pow_it < max_pow_it:
            minibatch = generate_batch(env, policy, horizon, minibatchsize)
            grad = estimator(minibatch, disc, policy, 
                                      baselinekind=baseline,
                                      shallow=shallow)
            params_2 = params + pow_alpha * psi / torch.norm(psi)
            policy.set_from_flat(params_2)
            grad_2 = estimator(minibatch, disc, policy, 
                                      baselinekind=baseline,
                                      shallow=shallow)
            policy.set_from_flat(params)
            psi = (1 - forget) * psi + forget * 1. / pow_alpha * (grad_2 - grad)
            lip_const = torch.norm(psi).item()
            if math.isnan(lip_const):
                err = 999
                break
            err = abs(lip_const - _lip_const) / lip_const
            _lip_const = lip_const
            pow_it += 1
        policy.set_from_flat(params)
        if err <= err_tol:
            print('Converged in %d iterations after %d failed attempts (alpha = %f, error = %f)' % (pow_it, attempts, pow_alpha, err))
        else:
            pow_alpha /= 10
        attempts += 1
        
    if err > err_tol:
        print('Failed to converge (%d failed attempts, alpha = %f, error = %f)' % (attempts, pow_alpha, err))
    return lip_const

def power_fo(policy, disc, env, std, horizon=None, pow_alpha=0.01, err_tol=0.1, max_pow_it=10, max_attempts=3):
    params = policy.get_flat()
    grad = env._grad(params, std, disc, horizon=horizon)
    
    err = 999
    attempts = 0
    while err > err_tol and attempts < max_attempts:
        pow_it = 0
        psi = torch.rand_like(grad)
        _lip_const = torch.norm(psi).item()
        while err > err_tol and pow_it < max_pow_it:
            params_2 = params + pow_alpha * psi / torch.norm(psi)
            grad_2 = env._grad(params_2, std, disc, horizon=horizon)
            psi = 1. / pow_alpha * (grad_2 - grad)
            lip_const = torch.norm(psi).item()
            if math.isnan(lip_const):
                err = 999
                break
            err = abs(lip_const - _lip_const) / lip_const
            _lip_const = lip_const
            pow_it += 1
        policy.set_from_flat(params)
        if err <= err_tol:
            print('Converged in %d iterations after %d failed attempts (alpha = %f, error = %f)' % (pow_it, attempts, pow_alpha, err))
        else:
            pow_alpha /= 10
        attempts += 1
        
    if err > err_tol:
        print('Failed to converge (%d failed attempts, alpha = %f, error = %f)' % (attempts, pow_alpha, err))
    return lip_const
