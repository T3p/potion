"""classic Linear Quadratic Gaussian Regulator task"""
from numbers import Number

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

"""
Linear quadratic regulator task (1-dimensional).

References
----------
"""

class lqr1d(gym.Env):

    def __init__(self, discrete_reward=False):
        self.max_pos = 2.
        self.max_action = 2.
        self.A = 1.
        self.B = 1.
        self.Q = 0.5
        self.R = 0.5
        
        # gym attributes
        self.viewer = None
        self.action_space = spaces.Box(low=-self.max_action,
                                       high=self.max_action,
                                       shape=(1,))
        self.observation_space = spaces.Box(low=-self.max_pos,
                                       high=self.max_pos,
                                       shape=(1,))

        self.initial_states = np.linspace(-self.max_pos, self.max_pos, 100)
        #self.initial_states = np.array([-0.9 * self.max_pos, 0.9 * self.max_pos])

        # initialize state
        self.seed()
        self.reset()

    def step(self, action, render=False):
        u = np.clip(action, -self.max_action, self.max_action)
        x_1 = np.clip(self.A * self.state + self.B * u, -self.max_pos, self.max_pos)
        cost = self.state ** 2 * self.Q + u ** 2 * self.R

        self.state = x_1
        return self.state, -np.asscalar(cost), False, {}

    def reset(self, state=None):
        if state is None:
            self.state = np.random.choice(self.initial_states)
        else:
            self.state = np.array(state)

        return self.state
    
    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 600

        world_width = (self.max_pos * 2)
        scale = screen_width / world_width
        bally = screen_height / 2
        ballradius = 3

        if self.viewer is None:
            clearance = 0  # y-offset
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            mass = rendering.make_circle(ballradius * 2)
            mass.set_color(.8, .3, .3)
            mass.add_attr(rendering.Transform(translation=(0, clearance)))
            self.masstrans = rendering.Transform()
            mass.add_attr(self.masstrans)
            self.viewer.add_geom(mass)
            self.track = rendering.Line((0, bally), (screen_width, bally))
            self.track.set_color(0.5, 0.5, 0.5)
            self.viewer.add_geom(self.track)
            zero_line = rendering.Line((screen_width / 2, 0),
                                       (screen_width / 2, screen_height))
            zero_line.set_color(0.5, 0.5, 0.5)
            self.viewer.add_geom(zero_line)

        x = self.state[0]
        ballx = x * scale + screen_width / 2.0
        self.masstrans.set_translation(ballx, bally)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
    
    def _get_initial_state(self):
        return np.array(np.random.choice(self.initial_states))

    def _riccati(self, param, disc, horizon=None):
        p = (self.Q + self.R * param**2) \
                    / (1 - disc * (self.A + self.B * param)**2)
        if horizon is not None:
            p *= 1 - disc**horizon * (self.A + self.B * param)**(2 * horizon)
        return p

    def _optimal_param(self, disc, iterations=100, horizon=None):
        p = 1.
        for _ in range(iterations):
            param = - disc * (self.B * p * self.A) \
                        / (self.R + disc * self.B**2 * p)
            p = self._riccati(param, disc, horizon)
        return param

    def _performance(self, param, std, disc, initial_state=None, horizon=None):
        p = self._riccati(param, disc)
        if horizon is not None:
            p *= 1 - disc**horizon * (self.A + self.B * param) ** (2 * horizon)
        if initial_state is not None:
            x2 = initial_state**2
        else:
            x2 = np.mean([x**2 for x in self.initial_states])
        return - p * x2 \
                - std**2 * (self.R / (1 - disc) + disc * self.B**2 * p)

    def _riccati_grad(self, param, disc, p=None, horizon=None):
        if p is None:
            p = self._riccati(param, disc)
        step = self.A + self.B * param
        dp = 2 * (self.R * param + disc * self.B * p * step) \
                / (1 - disc * step**2)
        if horizon is not None:
            dp = dp * (1 - disc**horizon * step**(2 * horizon)) - \
                2 * horizon * disc**horizon * step**(2 * horizon - 1) * self.B * p
        return dp
        
    def _grad(self, param, std, disc, initial_state=None, horizon=None):
        dp = self._riccati_grad(param, disc, horizon=horizon)
        if initial_state is not None:
            x2 = initial_state**2
        else:
            x2 = np.mean([x**2 for x in self.initial_states])
        return - dp * x2 - std**2 * disc * self.B**2 * dp
    
    def _riccati_hess(self, param, disc, horizon=None):
        p = self._riccati(param, disc)
        step = self.A + self.B * param
        dp = self._riccati_grad(param, disc, p)
        ddp = 2 * (self.R + disc * self.B * (self.B * p + 2 * step * dp)) \
                / (1 - disc * step**2)
        if horizon is not None:
            ddp = ddp * (1 - disc**horizon * step**(2 * horizon)) - \
                4 * horizon * disc**horizon * self.B * step**(2*horizon - 1) * dp -\
                2 * horizon * (horizon - 1) * disc**horizon * self.B**2 * step**(2*horizon - 2) * p
        return ddp
    
    def _hess(self, param, std, disc, initial_state=None, horizon=None):
        ddp = self._riccati_hess(param, disc, horizon=horizon)
        if initial_state is not None:
            x2 = initial_state**2
        else:
            x2 = np.mean([x**2 for x in self.initial_states])
        return - ddp * x2 \
                - std**2 * disc * self.B**2 * ddp
        
"""
    def computeQFunction(self, x, u, K, Sigma, n_random_xn=100):
        if isinstance(x, Number):
            x = np.array([x])
        if isinstance(u, Number):
            u = np.array([u])
        if isinstance(K, Number):
            K = np.array([K]).reshape(1, 1)
        if isinstance(Sigma, Number):
            Sigma = np.array([Sigma]).reshape(1, 1)

        P = self._computeP2(K)
        Qfun = 0
        for i in range(n_random_xn):
            noise = self.np_random.randn() * self.sigma_noise
            action_noise = self.np_random.multivariate_normal(
                np.zeros(Sigma.shape[0]), Sigma, 1)
            nextstate = np.dot(self.A, x) + np.dot(self.B,
                                                   u + action_noise) + noise
            Qfun -= np.dot(x.T, np.dot(self.Q, x)) + \
                np.dot(u.T, np.dot(self.R, u)) + \
                self.gamma * np.dot(nextstate.T, np.dot(P, nextstate)) + \
                (self.gamma / (1 - self.gamma)) * \
                np.trace(np.dot(Sigma,
                                self.R + self.gamma *
                                np.dot(self.B.T, np.dot(P, self.B))))
        Qfun = np.asscalar(Qfun) / n_random_xn
        return Qfun
"""

if __name__ == '__main__':

    env = lqr1d()
    gamma = 0.9
    horizon = 20
    std = 0.1
    theta_star = env._optimal_param(gamma)
    print('theta^* = ', theta_star)
    print('J^* = ', env._performance(theta_star,std, gamma, horizon=horizon))
