"""
GridWorld
"""

import logging
import gym
import gym.spaces
import random
import time
import math

logger = logging.getLogger(__name__)

class TwoGoals(gym.Env):
    def __init__(self):
        self.height = 5
        self.width = 5
        self.start = [(0,0)]
        self.absorbing = {(0, self.width - 1) : 1., (self.height - 1, self.width - 1) : 4.}
        self.goals = {(0, self.width - 1) : 1., (self.height - 1, self.width - 1) : 2.}
        
        self.n_actions = 4
        self.n_states = self.height * self.width
        self.action_space = gym.spaces.Discrete(self.n_actions)
        self.observation_space = gym.spaces.Discrete(self.n_states)
        
        self.reset()
        
    def seed(self, seed=None):
        random.seed(seed)
        return [seed]
        
    def step(self, action):
        #Move
        r = self.state[0]
        c = self.state[1]
        if action == 0:
            s = [r + 1, c]
        elif action == 1:
            s = [r, c + 1]
        elif action == 2:
            s = [r - 1, c]
        else:
            s = [r, c - 1]
        
        #Borders
        if s[0] < 0:
            s[0] = 0
        if s[0] >= self.height:
            s[0] = self.height - 1
        if s[1] < 0:
            s[1] = 0
        if s[1] >= self.width:
            s[1] = self.width - 1
        
        done = False
        reward = 0.
        self.state = tuple(s)
        if self.state in self.absorbing:
            done = True
        if self.state in self.goals:
            reward = self.goals[self.state]
        
        return self._get_state(), reward, done, {}

    def reset(self,initial=None):
        self.state = random.choice(self.start)
        return self._get_state()
    
    def render(self, mode='human', close=False):
        for i in range(self.height):
            for j in range(self.width):
                if self.state == (i,j):
                    print('Y', end='')
                elif (i,j) in self.goals:
                    print('*', end='')
                else:
                    print('_', end='')
            print()
        print()
        time.sleep(.1)
    
    def _get_state(self):
        return self.state[0] * self.width + self.state[1]
