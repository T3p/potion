from potion.envs.lqg1d import LQG1D
from gym import spaces
import numpy as np

class Mass(LQG1D):

    def __init__(self, discrete_reward=False):
        self.discrete_reward = discrete_reward
        self.horizon = 100
        self.gamma = 0.9
        self.max_pos = 100.
        self.max_action = 1.
        self.sigma_noise = 0.
        tau = self.tau = 1.
        mass = self.mass = 1.
        pos_penalty = 0.9
        vel_penalty = 9.
        force_penalty = 9.
        self.A = np.array([[1, tau],
                           [0, 1]])
        self.B = np.array([[0, 0],
                          [tau/mass, 0]])
        self.Q = np.array([[pos_penalty, 0],
                           [0, vel_penalty]])
        self.R = force_penalty*np.eye(2)
        self.initial_x = -0.75 * self.max_pos

        # gym attributes
        self.viewer = None
        high = self.max_pos
        self.action_space = spaces.Box(low=-self.max_action,
                                            high=self.max_action,
                                            shape=(1,))
        self.observation_space = spaces.Box(low=-high,
                                            high=high,
                                            shape=(2,))

        # initialize state
        self.seed()
        self.reset()

    def computeOptimalParam(self):
        return super(Mass, self).computeOptimalK()[0]

    def step(self, action, render=False):
        u = np.hstack((action, 0.))
        return super(Mass, self).step(u, render)

    def reset(self, state=None):
        x = self.initial_x #super(Mass, self).reset(state)
        self.state = np.hstack((x, 0.))
        return self.state
