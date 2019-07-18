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

def power(policy, batch, grad, disc, step=0.01, decay=0.1, tol=0.1, max_it=20, max_ep=5, estimator=gpomdp_estimator, baseline='peters', shallow=True, clip=0.2, verbose=True, mask=None, normalize=False):
    params = policy.get_flat()
    if mask is None:
        mask = torch.ones_like(params)
    err = 999
    attempts = 0
    psi = torch.rand_like(grad) * mask
    #must not be orthogonal to gradient!
    while(abs(psi.dot(grad)) < 1e-12):
        psi = torch.rand_like(grad) * mask
    psi /= torch.norm(psi) #normalize
    old_lip_const = torch.norm(psi).item()
    while err > tol and attempts < max_ep:
        pow_it = 0
        while err > tol and pow_it < max_it:
            params_2 = params + step * psi / torch.norm(psi)
            policy.set_from_flat(params_2)
            grad_2_samples = estimator(batch, disc, policy, 
                                      baselinekind=baseline,
                                      shallow=shallow,
                                      result='samples')
            policy.set_from_flat(params)
            iws = importance_weights(batch, policy, params_2)
            grad_2 = torch.mean(grad_2_samples * iws.unsqueeze(1), 0)
            if clip is not None:
                clipped_iws = torch.clamp(iws, 1-clip, 1+clip)
                clipped_grad_2 = torch.mean(grad_2_samples * clipped_iws.unsqueeze(1), 0)
                grad_2 = torch.max(grad_2, clipped_grad_2)
            elif normalize:
                grad_2 = torch.sum(grad_2_samples * iws.unsqueeze(1), 0) / torch.sum(iws)
            psi = (1 - decay) * psi + decay / step * (grad_2 - grad) * mask
            lip_const = torch.norm(psi).item()
            if math.isnan(lip_const):
                err = 999
                break
            err = abs(lip_const - old_lip_const) / lip_const
            old_lip_const = lip_const
            pow_it += 1
        policy.set_from_flat(params)
        if err <= tol and verbose:
            print('Power Method: converged in %d iterations after %d failed attempts (decay = %f, error = %f)' % (pow_it, attempts, decay, err))
        elif verbose > 1:
            print('Power Method: failed attempt (%d iterations, step = %f, error = %f)' % (pow_it, decay, err))
        decay /= 10
        attempts += 1
        
    if err > tol and verbose:
        print('Power Method: failed to converge (%d failed attempts, decay = %f, error = %f)' % (attempts, decay, err))
    return lip_const