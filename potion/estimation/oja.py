#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 17:23:58 2020

@author: matteo
"""
import torch

"""
Oja for Hessian from an SFO
"""
def oja(x, sfo, iterations=1000, step=1e-2, q=1e-2, reverse=False, base_grad=None):
    with torch.no_grad():
        #Input vector
        assert len(x.shape) == 1
        d = x.shape[0]
        
        #Random unit vector
        w = torch.randn(d) 
        w = w / torch.norm(w) 
    
        #Initialize largest eigenvalue
        sigma = 0
        #Pre-compute unperturbed gradient for efficiency
        if base_grad is None:
            base_grad = sfo(x)
        
        for i in range(iterations):
            #Matrix-vector product
            mvp = (sfo(x + q * w) - base_grad) / q
            #Update largest eigenvalue
            sigma = (1 - step) * sigma + step * torch.dot(w, mvp)
            #Update corresponding eigenvector
            if reverse:
                w = w - step * mvp
            else:
                w = w + step * mvp
            w = w / torch.norm(w)
        
        return sigma, w


def abs_oja(x, sfo, iterations=1000, step=1e-2, q=1e-2):
    sigma_max, _ = oja(x, sfo, iterations, step, q)
    sigma_min, _ = oja(x, sfo, iterations, step, q, reverse=True)
    return torch.max(torch.abs(sigma_max), torch.abs(sigma_min))


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
    sigma = abs_oja(x, sfo)
    print('Max sv:', sigma)