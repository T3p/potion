#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 10:58:20 2019

@author: matteo
"""

def incr_mean(old_mean, new_sample, tot_samples):
    return old_mean + (new_sample - old_mean) / tot_samples

def incr_var(old_var, old_mean, new_mean, new_sample, tot_samples):
    old_ssd = old_var * (tot_samples - 1)
    new_ssd = old_ssd + (new_sample - old_mean) * (new_sample - new_mean)
    return new_ssd / tot_samples