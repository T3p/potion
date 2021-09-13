#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 16:05:38 2021
"""
import torch

def jackknife(base_estimator, data):
    size = len(data)
    full = base_estimator(data)
    sum_of_parts = 0.
    sum_of_squares = 0.
    for i in range(size):
        part = base_estimator(torch.cat((data[:i-1,:], data[i:,:]), 0))
        corrected = size * full - (size - 1) * part
        sum_of_parts += corrected
        sum_of_squares += (corrected - full)**2
    
    return sum_of_parts / size, sum_of_squares / (size * (size - 1))