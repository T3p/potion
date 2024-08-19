#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
from numbers import Number


class LQR(gym.Env):
    """
    Gym environment implementing an LQR problem

    s_{t+1} = A s_t + B a_t + noise
    r_{t+1} = - s_t^T Q s_t - a_t^T R a_t

    Run script to compute optimal policy parameters
    """
    metadata = {
        'render_modes': ['human']
    }

    def __init__(self):
        self.state = None
        self.ds = 1  # state dimension
        self.da = 1  # action dimension
        self.horizon = 10  # task horizon (reset is not automatic!)
        self.gamma = 0.9  # discount factor
        self.max_pos = 100 * np.ones(self.ds)  # max state for clipping
        self.max_action = 200 * np.ones(self.da)  # max action for clipping
        self.sigma_noise = 0 * np.eye(self.ds)  # std dev of environment noise
        self.A = 1. * np.eye(self.ds)
        self.B = 1. * np.eye(self.ds, self.da)
        self.Q = 1. * np.eye(self.ds)
        self.R = 1. * np.eye(self.da)
        self.timestep = 0
        self.np_random = None

        # Gym attributes
        self.viewer = None
        self.action_space = spaces.Box(low=-self.max_action,
                                       high=self.max_action,
                                       dtype=float)
        self.observation_space = spaces.Box(low=-self.max_pos,
                                            high=self.max_pos,
                                            dtype=float)

    def step(self, action, render=False):
        u = np.clip(np.ravel(np.atleast_1d(action)), -self.max_action, self.max_action)
        noise = np.dot(self.sigma_noise, self.np_random.standard_normal(self.ds))
        xn = np.clip(np.dot(self.A, self.state.T) + np.dot(self.B, u) + noise, -self.max_pos, self.max_pos)
        cost = np.dot(self.state,
                      np.dot(self.Q, self.state)) + \
               np.dot(u, np.dot(self.R, u))

        self.state = xn.ravel()
        self.timestep += 1

        terminated = False
        truncated = (self.timestep >= self.horizon)

        return self.get_state(), -cost.item(), terminated, truncated, dict()

    def reset(self, seed=None, options=None):
        """
        By default, uniform initialization
        if options is not None, options["state"] can be used to reset to a specific state
        """
        self.timestep = 0
        self.np_random, _ = seeding.np_random(seed)

        if options is not None and "state" in options:
            self.state = np.array(options["state"])
        else:
            self.state = np.array(self.np_random.uniform(low=-1. * np.ones(self.ds),
                                                         high=1. * np.ones(self.ds)))

        return self.get_state(), dict()

    def get_state(self):
        return np.array(self.state)

    def render(self, mode='human', close=False):
        print(np.array2string(self.get_state()))

    def _computeP2(self, K):
        """
        This function computes the Riccati equation associated to the LQG
        problem.
        Args:
            K (matrix): the matrix associated to the linear controller a = K s

        Returns:
            P (matrix): the Riccati Matrix

        """
        I = np.eye(self.Q.shape[0], self.Q.shape[1])
        if np.array_equal(self.A, I) and np.array_equal(self.B, I):
            P = (self.Q + np.dot(K.T, np.dot(self.R, K))) / (I - self.gamma *
                                                             (I + 2 * K + K **
                                                              2))
        else:
            tolerance = 0.0001
            converged = False
            P = np.eye(self.Q.shape[0], self.Q.shape[1])
            while not converged:
                Pnew = self.Q + self.gamma * np.dot(self.A.T,
                                                    np.dot(P, self.A)) + \
                       self.gamma * np.dot(K.T, np.dot(self.B.T,
                                                       np.dot(P, self.A))) + \
                       self.gamma * np.dot(self.A.T,
                                           np.dot(P, np.dot(self.B, K))) + \
                       self.gamma * np.dot(K.T,
                                           np.dot(self.B.T,
                                                  np.dot(P, np.dot(self.B,
                                                                   K)))) + \
                       np.dot(K.T, np.dot(self.R, K))
                converged = np.max(np.abs(P - Pnew)) < tolerance
                P = Pnew
        return P

    def computeOptimalK(self):
        """
        This function computes the optimal linear controller associated to the
        LQG problem (a = K * s).

        Returns:
            K (matrix): the optimal controller

        """
        P = np.eye(self.Q.shape[0], self.Q.shape[1])
        for i in range(100):
            K = -self.gamma * np.dot(np.linalg.inv(
                self.R + self.gamma * (np.dot(self.B.T, np.dot(P, self.B)))),
                np.dot(self.B.T, np.dot(P, self.A)))
            P = self._computeP2(K)
        K = -self.gamma * np.dot(np.linalg.inv(self.R + self.gamma *
                                               (np.dot(self.B.T,
                                                       np.dot(P, self.B)))),
                                 np.dot(self.B.T, np.dot(P, self.A)))
        return K

    def computeJ(self, K, Sigma=1., n_random_x0=10000):
        """
        This function computes the discounted reward associated to the provided
        linear controller (a = K s + epsilon, epsilon sim N(0,Sigma)).
        Args:
            K (matrix): the controller matrix
            Sigma (matrix): covariance matrix of the zero-mean noise added to
                            the controller action
            n_random_x0: the number of samples to draw in order to average over
                         the initial state

        Returns:
            J (float): The discounted reward

        """
        if isinstance(K, Number):
            K = np.array([K]).reshape(1, 1)
        if isinstance(Sigma, Number):
            Sigma = np.array([Sigma]).reshape(1, 1)

        P = self._computeP2(K)
        temp = np.dot(
            Sigma, (self.R + self.gamma * np.dot(self.B.T,
                                                 np.dot(P, self.B))))
        temp = np.trace(temp) if np.ndim(temp) > 1 else temp
        W = (1 / (1 - self.gamma)) * temp

        # Closed-form expectation in the scalar case:
        if np.size(K) == 1:
            return min(0, np.array(-self.max_pos ** 2 * P / 3 - W).item())

        # Monte Carlo estimators for higher dimensions
        J = 0.0
        for i in range(n_random_x0):
            self.reset()
            x0 = self.get_state()
            J -= np.dot(x0.T, np.dot(P, x0)) \
                 + W
        J /= n_random_x0
        return min(0., J)

    def grad_K(self, K, Sigma):
        """
        Policy gradient (wrt K) of Gaussian linear policy with mean K s
        and covariance Sigma.
        Scalar case only
        """
        I = np.eye(self.Q.shape[0], self.Q.shape[1])
        if not np.array_equal(self.A, I) or not np.array_equal(self.B, I):
            raise NotImplementedError
        if not isinstance(K, Number) or not isinstance(Sigma, Number):
            raise NotImplementedError
        theta = np.array(K).item()
        sigma = np.array(Sigma).item()

        den = 1 - self.gamma * (1 + 2 * theta + theta ** 2)
        dePdeK = 2 * (theta * self.R / den + self.gamma * (self.Q + theta ** 2 * self.R) * (1 + theta) / den ** 2)
        return (- dePdeK * (self.max_pos ** 2 / 3 + self.gamma * sigma / (1 - self.gamma))).item()

    def grad_Sigma(self, K, Sigma=None):
        """
        Policy gradient wrt (adaptive) covariance Sigma
        """
        I = np.eye(self.Q.shape[0], self.Q.shape[1])
        if not np.array_equal(self.A, I) or not np.array_equal(self.B, I):
            raise NotImplementedError
        if not isinstance(K, Number) or not isinstance(Sigma, Number):
            raise NotImplementedError

        K = np.array(K)
        P = self._computeP2(K)
        return (-(self.R + self.gamma * P) / (1 - self.gamma)).item()

    def grad_mixed(self, K, Sigma=None):
        """
        Mixed-derivative policy gradient for K and Sigma
        """
        I = np.eye(self.Q.shape[0], self.Q.shape[1])
        if not np.array_equal(self.A, I) or not np.array_equal(self.B, I):
            raise NotImplementedError
        if not isinstance(K, Number) or not isinstance(Sigma, Number):
            raise NotImplementedError
        theta = np.array(K).item()

        den = 1 - self.gamma * (1 + 2 * theta + theta ** 2)
        dePdeK = 2 * (theta * self.R / den + self.gamma * (self.Q + theta ** 2 * self.R) * (1 + theta) / den ** 2)

        return (-dePdeK * self.gamma / (1 - self.gamma)).item()

    def computeQFunction(self, x, u, K, Sigma, n_random_xn=100):
        """
        This function computes the Q-value of a pair (x,u) given the linear
        controller Kx + epsilon where epsilon sim N(0, Sigma).
        Args:
            x (int, array): the state
            u (int, array): the action
            K (matrix): the controller matrix
            Sigma (matrix): covariance matrix of the zero-mean noise added to
            the controller action
            n_random_xn: the number of samples to draw in order to average over
            the next state

        Returns:
            Qfun (float): The Q-value in the given pair (x,u) under the given
            controller

        """
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
            noise = self.np_random.standard_normal() * self.sigma_noise
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
        Qfun = np.array(Qfun).item() / n_random_xn
        return Qfun


if __name__ == '__main__':
    """
    Compute optimal parameters K for Gaussian policy with mean Ks
    and covariance matrix sigma_controller (1 by default)
    """
    env = LQR()
    sigma_controller = 1 * np.ones(env.da)
    theta_star = env.computeOptimalK()
    print('theta^* = ', theta_star)
    print('J^* = ', env.computeJ(theta_star, sigma_controller))
