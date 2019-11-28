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
import random

def oja(policy, batch, disc, horizon= None, env=None, iterations=None, step1=None, step2=1e-2, estimator=gpomdp_estimator, shallow=True, verbose=False):
    params = policy.get_flat()
    eigvec = torch.randn_like(params)
    if iterations == None:
        iterations = len(batch)
    if step1 == None:
        step1 = 1. / math.sqrt(iterations)
    for _ in range(iterations):    
        random.shuffle(batch)
        grad = estimator(batch[1:len(batch)//2], disc, policy, baselinekind='peters', shallow=shallow)
        pert_params = params + step2 * eigvec
        policy.set_from_flat(pert_params)
        iw = importance_weights(batch[len(batch)//2:], policy, pert_params)
        pert_grad = torch.mean(iw.unsqueeze(-1) * estimator(batch[len(batch)//2:], disc, policy, baselinekind='peters', shallow=shallow, result='samples'), 0)
        policy.set_from_flat(params)
        hvp = (pert_grad - grad) / step2
        eigvec = eigvec + step1 * hvp
        eigvec = eigvec / torch.norm(eigvec)
    return torch.abs(torch.dot(eigvec, hvp))