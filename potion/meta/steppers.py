#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 00:40:38 2019

@author: matteo
"""
import torch
import math

class ConstantStepper:
    def __init__(self, alpha):
        self.alpha = torch.tensor(alpha)
    
    def next(self, *args):
        return self.alpha

    def reset(self):
        pass
    
    def __str__(self):
        return str(self.alpha)
    
class AlphaEta:
    def __init__(self, alpha, eta):
        self.alpha = alpha
        self.eta = eta
    def next(self, grad):
        step = torch.ones_like(grad) * self.alpha
        step[0] = self.eta
        return step
    
    def __str__(self):
        return 'alpha = %f, eta = %f' % (str(self.alpha), str(self.eta))

class SqrtDecay:
    def __init__(self, alpha):
        self.alpha = torch.tensor(alpha)
        self.t = 1
    
    def next(self, *args):
        return self.alpha / math.sqrt(self.t)
        self.t += 1    
    
    def reset(self):
        self.t = 1
    
    def __str__(self):
        return str(self.alpha) + ' / sqrt(t)'

class RMSprop:
    def __init__(self, alpha=1e-3, beta=0.9, epsilon=1e-8):
        self.m2 = 0
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        
    def next(self, grad, *args):
        self.m2 = self.beta * self.m2 + (1 - self.beta) * grad ** 2
        return self.alpha / torch.sqrt(self.m2 + self.epsilon)
    
    def reset(self):
        self.m2 = 0

    def __str__(self):
        return 'RMSprop (alpha = %f, beta = %f, epsilon = %f)' % (
                str(self.alpha), str(self.beta), str(self.epsilon))
