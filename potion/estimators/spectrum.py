#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 17:23:58 2020

@author: matteo
"""
import torch
from potion.estimators.gradients import gpomdp_estimator
from potion.estimators.offpolicy_gradients import off_gpomdp_estimator

"""
Largest (smallest if reversed) eigenvalue of Hessian from an SFO
"""
def oja(x, sfo, iterations=1000, step=1e-2, perturbation=1e-2, base_grad=None, reverse=False):
    with torch.no_grad():
        #Input vector
        assert len(x.shape) == 1
        d = x.shape[0]
        
        #Random unit vector
        w = torch.randn(d) 
        w = w / torch.norm(w) 
    
        #Initialize largest eigenvalue
        sigma = torch.zeros(1)
        #Pre-compute unperturbed gradient for efficiency
        if base_grad is None:
            base_grad = sfo(x)
        
        for i in range(iterations):
            #Matrix-vector product
            mvp = (sfo(x + perturbation * w) - base_grad) / perturbation
            #Update largest eigenvalue
            sigma_new = (1 - step) * sigma + step * torch.dot(w, mvp)
            err = torch.abs(sigma_new - sigma) / torch.abs(sigma)
            sigma = sigma_new
            #Update corresponding eigenvector
            if reverse:
                w = w - step * mvp
            else:
                w = w + step * mvp
            w = w / torch.norm(w)
        
        print(err.item())
        return sigma, w

"""
Hessian spectral norm from an SFO
"""
def spectral_norm(x, sfo, iterations=1000, step=1e-2, perturbation=1e-2, base_grad=None):
    sigma_max, _ = oja(x, sfo, iterations, step, perturbation, base_grad)
    sigma_min, _ = oja(x, sfo, iterations, step, perturbation, base_grad, reverse=True)
    return torch.max(torch.abs(sigma_max), torch.abs(sigma_min))


def ojapg(policy, batch, disc, 
                     iterations=1000, 
                     step=1e-2, 
                     perturbation=1e-2, 
                     estimator='gpomdp', 
                     baseline='peters', 
                     shallow=True, 
                     verbose=False):
    
    if estimator != 'gpomdp':
        raise NotImplementedError
    
    base_grad = gpomdp_estimator(batch, disc, policy, 
                          baselinekind=baseline, 
                          shallow=shallow)
    
    
    def sfo(pert_params):
        return off_gpomdp_estimator(batch, disc, policy, 
                               target_params = pert_params,
                               baselinekind=baseline, 
                               shallow=shallow)
    
    return spectral_norm(policy.get_flat(), sfo, iterations, step, perturbation,
                         base_grad=base_grad)


if __name__ == '__main__':
    #Simple example with quadratic matrix f(x) = 1/2 x^TAx
    A = torch.tensor([[2., 0.], [0., -3.]])
    sigma_noise = 1e-3

    #An SFO is simulated with Gaussian noise    
    def sfo(x):
        return torch.mv(A, x) + torch.randn(1).item() * sigma_noise
    
    #Hessian is constant in this case
    x = torch.ones(2) 

    #Maximum eigenvalue    
    sigma, w = oja(x, sfo)
    print('Max eig:', sigma, w)
    
    #Minimum eigenvalue
    sigma, w = oja(x, sfo, reverse=True)
    print('Min eig:', sigma, w)
    
    #Maximum singular value (only for symmetric matrices):
    sigma = spectral_norm(x, sfo)
    print('Max sv:', sigma)