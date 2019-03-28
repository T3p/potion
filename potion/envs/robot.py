#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 17:25:59 2019

@author: matteo
"""

from gym import spaces
import gym
import numpy as np

class Robot(gym.Env):

    def __init__(self):
        self.horizon = 100
        self.max_pos = 10.
        self.max_action = 1.
        self.speed = 1.
       
        self.init = np.array([.1 * self.max_pos, .1 * self.max_pos, 1., 0.])
        self.goal = np.array([.9 * self.max_pos, .9 * self.max_pos])
        self.eps = 0.05 * self.max_pos
        self.angle = 0.

        # gym attributes
        self.viewer = None
        high = np.array([self.max_pos, self.max_pos, 1., 1.])
        self.action_space = spaces.Box(low=-self.max_action,
                                            high=self.max_action,
                                            shape=(1,))
        self.observation_space = spaces.Box(low=-high,
                                            high=high)

        # initialize state
        self.seed()
        self.reset()

    def step(self, action, render=False):
        action = np.clip(action, -self.max_action, self.max_action)
        self.angle += action
        cos = np.cos(self.angle)
        sin = np.sin(self.angle)
        self.state[0] = self.speed * cos
        self.state[1] = self.speed * sin
        self.state[2] = cos
        self.state[3] = sin
        reward = -np.linalg.norm(self.state[:2] - self.goal)
        done = np.linalg.norm(self.state[:2] - self.goal) < self.eps
        return self.state, reward, done, {}

    def reset(self, state=None):
        self.state = self.init
        return self.state
