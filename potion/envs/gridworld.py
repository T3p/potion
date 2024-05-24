"""
GridWorld
"""

import logging
import gymnasium as gym
from gymnasium.utils import seeding
import random
import time

logger = logging.getLogger(__name__)


class GridWorld(gym.Env):
    def __init__(self):
        self.active_goals = None
        self.state = None
        self.height = 5
        self.width = 5
        self.start = [(0, 0)]
        self.absorbing = {(self.height - 1, self.width - 1),
                          (self.height - 3, self.width - 3)}
        self.goals = {(self.height - 1, self.width - 1): 1.,
                      (self.height - 3, self.width - 3): -1.}

        self.n_actions = 4
        self.n_states = self.height * self.width
        self.action_space = gym.spaces.Discrete(self.n_actions)
        self.observation_space = gym.spaces.Discrete(self.n_states)
        self.noise = 0.05
        self.np_random = None

        self.reset()

    def step(self, action):
        heuristic = 0.1 * (self.state[0] + self.state[1])

        # Noise
        if random.uniform(0, 1) < self.noise:
            action = random.choice(range(self.n_actions))

        # Move
        r = self.state[0]
        c = self.state[1]
        s = [r, c]
        if action == 0:
            s = [r + 1, c]
        elif action == 1:
            s = [r, c + 1]
        elif action == 2:
            s = [r - 1, c]
        elif action == 3:
            s = [r, c - 1]
        elif action == 4:
            s = [r + 1, c + 1]
        elif action == 5:
            s = [r - 1, c + 1]
        elif action == 6:
            s = [r - 1, c - 1]
        elif action == 7:
            s = [r + 1, c - 1]

        # Borders
        if s[0] < 0:
            s[0] = 0
        if s[0] >= self.height:
            s[0] = self.height - 1
        if s[1] < 0:
            s[1] = 0
        if s[1] >= self.width:
            s[1] = self.width - 1

        terminated = False
        truncated = False
        reward = 0.
        self.state = tuple[int, int](s)
        if self.state in self.absorbing:
            terminated = True
        if self.state in self.active_goals:
            reward = self.active_goals[self.state]
            self.active_goals[self.state] = 0

        reward = reward + heuristic

        return self._get_state(), reward, terminated, truncated, dict()

    def reset(self, seed=None, options=None):
        self.np_random, _ = seeding.np_random(seed)
        if options is not None and "state" in options:
            self.state = options["state"]
        else:
            self.state = self.start[self.np_random.choice(len(self.start))]
        self.active_goals = dict(self.goals)
        return self._get_state(), dict()

    def render(self, mode='human', close=False):
        for i in range(self.height):
            for j in range(self.width):
                if self.state == (i, j):
                    print('Y', end='')
                elif (i, j) in self.active_goals and self.active_goals[(i, j)] > 0:
                    print('*', end='')
                elif (i, j) in self.active_goals:
                    print('x', end='')
                else:
                    print('_', end='')
            print()
        print()
        time.sleep(.1)

    def _get_state(self):
        return self.state[0] * self.width + self.state[1]
