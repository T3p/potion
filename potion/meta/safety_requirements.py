#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 20:32:36 2019

@author: matteo
"""

class MonotonicImprovement:
    def __init__(self, *args):
        pass
    
    def next(self, *args):
        return 0.
    
class Budget:
    def __init__(self, initial_perf, *args):
        self.initial_perf = initial_perf
    
    def next(self, curr_perf, *args):
        budget = curr_perf - self.initial_perf
        return - budget
    
class FixedThreshold:
    def __init__(self, threshold, *args):
        self.threshold = threshold
    def next(self, curr_perf, *args):
        budget = curr_perf - self.threshold
        return -budget