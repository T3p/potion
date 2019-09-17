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

class Corridor(gym.Env):
    def __init__(self):
        self.length = 5
        self.start = 0
        self.reward = [0.] * self.length #list(range(self.length))
        self.absorbing = [0] * self.length
        self.absorbing[-1] = 1
        self.reward[1] = 1
        self.reward[-2] = -1
        self.reward[-1] = 10
        
        
        self.n_actions = 2
        self.n_states = self.length
        self.action_space = gym.spaces.Discrete(self.n_actions)
        self.observation_space = gym.spaces.Discrete(self.n_states)
        
        self.reset()
        
    def seed(self, seed=None):
        random.seed(seed)
        return [seed]
        
    def step(self, action):
        #Move
        if action == 0:
            s = self.state + 1
        elif action == 1:
            s = self.state - 1
        
        #Borders
        if s < 0:
            s = 0
        if s >= self.length:
            s = self.length - 1
        
        done = False
        self.state = s
        if self.absorbing[s]:
            done = True
        reward = self.reward[s]
        
        return self._get_state(), reward, done, {}

    def reset(self,initial=None):
        self.state = 0
        return self._get_state()
    
    def render(self, mode='human', close=False):
        for i in range(self.length):
            if self.state == i:
                print('Y', end='')
            else:
                print('_', end='')
        print()
        time.sleep(.1)
    
    def _get_state(self):
        return self.state
