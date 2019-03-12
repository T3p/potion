#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 15:34:15 2019
"""
import math

class InfiniteHorizonConfidence:
    def __init__(self, delta):
        self.delta = delta
    def next(self, t):
        return 6 * self.delta / (math.pi ** 2 + t ** 2)