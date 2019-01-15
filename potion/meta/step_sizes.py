#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 00:40:38 2019

@author: matteo
"""
import torch

class RMSprop:
    def __init__(self, alpha=1e-3, beta=0.9, epsilon=1e-8):
        self.m2 = 0
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        
    def next(self, grad):
        self.m2 = self.beta * self.m2 + (1 - self.beta) * grad ** 2
        print(self.m2)
        return self.alpha / torch.sqrt(self.m2 + self.epsilon)
    
    def reset(self):
        self.m2 = 0