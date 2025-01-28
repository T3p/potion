import gymnasium as gym
from gymnasium import spaces
import numpy as np
import warnings
from potion.policies.gaussian_policies import LinearGaussianPolicy
from potion.simulation.trajectory_generators import estimate_average_return
from potion.policies.wrappers import Staged

class LQR(gym.Env):
    metadata = {
        'render_modes': ['human']
    }

    def __init__(self,
                 A=1.,
                 B=1.,
                 Q=1.,
                 R=1.,
                 Q_final=0.,  # cost at terminal state
                 M_mix=0.,  # keep it zero for now
                 init_mean=0.,
                 init_std=1.,
                 state_bounds=(-100., 100.),
                 action_bounds=(-100, 100.),
                 noise_std=0.,
                 horizon=None):

        # A matrix
        if np.isscalar(A):
            A = A * np.eye(1)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("A must be a square matrix")
        self.A = A
        self.ds = A.shape[0]  # state dimension

        # B matrix
        if np.isscalar(B):
            B = B * np.ones(self.ds)[..., None]
        if B.ndim == 1 and B.shape[0] == self.ds:
            B = B[..., None]
        if B.ndim != 2 or B.shape[0] != self.ds:
            raise ValueError("Bad shape: B should be a {}xk matrix".format(self.ds))
        self.B = B
        self.da = B.shape[1]

        # M_mix matrix
        if np.isscalar(M_mix):
            if self.da == self.ds:
                M_mix = M_mix * np.ones((self.da, self.ds))
            elif np.isclose(M_mix, 0.):
                M_mix = np.zeros((self.da, self.ds))
        if M_mix.ndim != 2 or M_mix.shape[0] != self.da or M_mix.shape[1] != self.ds:
            raise ValueError("Bad shape: B should be a {}x{} matrix".format(self.da, self.ds))
        self.M_mix = M_mix

        # Controllability
        powers = [B]
        for _ in range(self.ds - 1):
            powers.append(A @ powers[-1])
        C = np.concatenate(powers, axis=-1)
        if np.linalg.matrix_rank(C) < self.ds:
            warnings.warn("The system is not controllable!", UserWarning)

        # Q matrix
        if np.isscalar(Q):
            Q = Q * np.eye(self.ds)
        if Q.ndim != 2 or Q.shape[0] != self.ds or Q.shape[1] != self.ds:
            raise ValueError("Bad shape: Q should be a {}x{} matrix".format(self.ds, self.ds))
        if not (np.allclose(Q, Q.T) and np.all(np.linalg.eigvals(Q)) > 0):
            raise ValueError("Q matrix should symmetric positive definite")
        self.Q = Q

        # R matrix
        if np.isscalar(R):
            R = R * np.eye(self.da)
        if R.ndim != 2 or R.shape[0] != self.da or R.shape[1] != self.da:
            raise ValueError("Bad shape: R should be a {}x{} matrix".format(self.da, self.da))
        if not (np.allclose(R, R.T) and np.all(np.linalg.eigvals(R)) > 0):
            raise ValueError("R matrix should symmetric positive definite")
        self.R = R

        # Q final matrix
        if np.isscalar(Q_final):
            Q_final = Q_final * np.eye(self.ds)
        if Q_final.ndim != 2 or Q_final.shape[0] != self.ds or Q_final.shape[1] != self.ds:
            raise ValueError("Bad shape: Q_final should be a {}x{} matrix".format(self.ds, self.ds))
        if not (np.allclose(Q_final, Q_final.T) and np.all(np.linalg.eigvals(Q_final)) >= 0):
            raise ValueError("Q_final matrix should symmetric positive semi-definite")
        self.Q_final = Q_final

        # State space
        if state_bounds is None:
            min_state = -np.inf
            max_state = np.inf
        elif not isinstance(state_bounds, tuple):
            min_state = -state_bounds
            max_state = state_bounds
        else:
            min_state = state_bounds[0]
            max_state = state_bounds[1]

        if min_state is None:
            min_state = -np.inf
        elif np.isscalar(min_state):
            min_state = min_state * np.ones(self.ds)
        if min_state.ndim != 1 or min_state.shape[0] != self.ds:
            raise ValueError("Bad shape: min state should be a {}-dimensional vector".format(self.ds))
        self.min_s = min_state

        if max_state is None:
            max_state = np.inf
        elif np.isscalar(max_state):
            max_state = max_state * np.ones(self.ds)
        if max_state.ndim != 1 or max_state.shape[0] != self.ds:
            raise ValueError("Bad shape: max state should be a {}-dimensional vector".format(self.ds))
        if np.any(max_state < min_state):
            raise ValueError("Given state upper bound is below given state lower bound")
        self.max_s = max_state

        self.observation_space = spaces.Box(low=min_state,
                                            high=max_state,
                                            dtype=float)

        # Initial State
        if np.isscalar(init_mean):
            init_mean = init_mean * np.ones(self.ds)
        if isinstance(init_mean, list):
            for i in range(len(init_mean)):
                if np.isscalar(init_mean[i]):
                    init_mean[i] = init_mean[i] * np.ones(self.ds)
        self.init_mean = init_mean
        if np.isscalar(init_std):
            init_std = init_std * np.ones(self.ds)
        self.init_std = init_std

        # Action space
        if action_bounds is None:
            min_action = -np.inf
            max_action = np.inf
        elif not isinstance(action_bounds, tuple):
            min_action = -action_bounds
            max_action = action_bounds
        else:
            min_action = action_bounds[0]
            max_action = action_bounds[1]

        if min_action is None:
            min_action = -np.inf
        elif np.isscalar(min_action):
            min_action = min_action * np.ones(self.da)
        if min_action.ndim != 1 or min_action.shape[0] != self.da:
            raise ValueError("Bad shape: min action should be a {}-dimensional vector".format(self.da))
        self.min_a = min_action

        if max_action is None:
            max_action = np.inf
        elif np.isscalar(max_action):
            max_action = max_action * np.ones(self.da)
        if max_action.ndim != 1 or max_action.shape[0] != self.da:
            raise ValueError("Bad shape: max action should be a {}-dimensional vector".format(self.da))
        if np.any(max_action < min_action):
            raise ValueError("Given action upper bound is below given action lower bound")
        self.max_a = max_action

        self.action_space = spaces.Box(low=min_action,
                                       high=max_action,
                                       dtype=float)

        # Noise
        if np.isscalar(noise_std):
            noise_std = noise_std * np.ones(self.ds)
        self.noise_std = noise_std

        # Horizon
        self.horizon = horizon

        # Initializations
        self.state = None
        self.rng = None
        self.viewer = None
        self.t = 0

    def step(self, action, render=False):
        a = np.clip(action, self.min_a, self.max_a)
        cost = np.dot(self.state, self.Q @ self.state) + np.dot(a, self.R @ a) + 2 * np.dot(a, self.M_mix @ self.state)
        noise = self.noise_std * self.rng.normal(size=self.ds)
        s = self.A @ self.state + self.B @ action + noise
        s = np.clip(s, self.min_s, self.max_s)

        self.t += 1
        self.state = s

        terminated = False
        truncated = self.horizon is not None and self.t >= self.horizon
        if self.t == self.horizon:
            cost += np.dot(self.state, self.Q_final @ self.state)

        return self.state, -cost, terminated, truncated, dict()

    def reset(self, seed=None, options=None):
        self.t = 0
        self.rng = np.random.default_rng(seed)

        if options is not None and "state" in options:
            self.state = np.array(options["state"])
        else:
            if isinstance(self.init_mean, list):
                mean_state = self.rng.choice(self.init_mean)
            else:
                mean_state = self.init_mean
            self.state = mean_state + self.rng.normal(size=self.ds) * self.init_std

        return self.state, dict()

    def render(self, mode='human', close=False):
        print(np.array2string(self.state))

    def discounted_P_matrix(self, discount, max_iterations=100):
        P = self.Q  # dxd
        for it in range(max_iterations):
            inverse = np.linalg.inv(self.R + discount * self.B.T @ P @ self.B)
            M = self.M_mix + discount * self.B.T @ P @ self.A
            P_next = self.Q + discount * self.A.T @ P @ self.A - M.T @ inverse @ M
            if np.allclose(P_next, P):
                return P_next
            P = P_next

        warnings.warn("Computation of P matrix did not converge")
        return P

    def discounted_optimal_gain(self, discount, max_iterations=100):
        P = self.discounted_P_matrix(discount, max_iterations)
        inverse = np.linalg.inv(self.R + discount * self.B.T @ P @ self.B)
        return - inverse @ (self.M_mix + discount * self.B.T @ P @ self.A)

    def discounted_optimal_return(self, discount, policy_std=0., max_iterations=100):
        # only works with scalar stds
        P = self.discounted_P_matrix(discount, max_iterations)

        if isinstance(self.init_mean, list):
            init_term = - np.mean([self._init_term(P, s, self.init_std) for s in self.init_mean])
        else:
            init_term = - self._init_term(P, self.init_mean, self.init_std)

        state_noise_term = - discount * np.trace(np.diag(self.noise_std ** 2) @ P) / (1. - discount)
        action_noise_term = - np.trace((self.R + discount * self.B.T @ P @ self.B) * policy_std ** 2) / (1. - discount)
        noise_term = state_noise_term + action_noise_term

        return init_term + noise_term

    def _discounted_P_K_matrix(self, K, discount, max_iterations):
        if not np.allclose(self.M_mix, 0.):
            raise NotImplementedError  # TODO
        P = self.Q  # dxd
        for it in range(max_iterations):
            pi_term = (discount * K @ self.B.T @ P @ self.A
                       + discount * self.A.T @ P @ self.B @ K.T
                       + discount * K @ self.B.T @ P @ self.B @ K.T
                       + K @ self.R @ K.T)
            P_next = self.Q + discount * self.A.T @ P @ self.A + pi_term
            if np.allclose(P_next, P):
                return P_next
            P = P_next

        warnings.warn("Computation of P matrix did not converge")
        return P

    def discounted_v(self, state, policy_param, discount, policy_std=0., max_iterations=100):
        # policy_param must be a ds x da matrix (TODO reshaping)
        P = self._discounted_P_K_matrix(policy_param, discount, max_iterations)
        state_term = - np.dot(state, P @ state)

        state_noise_term = - discount * np.trace(np.diag(self.noise_std ** 2) @ P) / (1. - discount)
        action_noise_term = - np.trace((self.R + discount * self.B.T @ P @ self.B) * policy_std ** 2) / (1. - discount)
        noise_term = state_noise_term + action_noise_term

        return state_term + noise_term

    def discounted_q(self, state, action, policy_param, discount, policy_std=0., max_iterations=100):
        # policy_param must be a ds x da matrix (TODO reshaping)
        P = self._discounted_P_K_matrix(policy_param, discount, max_iterations)
        Q_11 = self.Q + discount * self.A.T @ P @ self.A
        Q_12 = discount * self.A.T @ P @ self.B
        Q_21 = discount * self.B.T @ P @ self.A
        Q_22 = self.R + discount * self.B.T @ P @ self.B
        state_action_term = - (np.dot(state, Q_11 @ state)
                                + np.dot(state, Q_12 @ action)
                                + np.dot(action, Q_21 @ state)
                                + np.dot(action, Q_22 @ action))

        state_noise_term = - discount * np.trace(np.diag(self.noise_std ** 2) @ P) / (1. - discount)
        action_noise_term = - discount * np.trace((self.R + discount * self.B.T @ P @ self.B) * policy_std ** 2) / (1. - discount)
        noise_term = state_noise_term + action_noise_term

        return state_action_term + noise_term

    def q_representation(self, state, action):
        if state.shape != (self.ds,) or action.shape != (self.da,):
            raise ValueError("Invalid state or action shape")
        x = np.concatenate((state, action))
        A = np.outer(x, x)
        triu = A[np.triu_indices(self.ds + self.da)]
        return np.concatenate((np.ones(1), triu))

    def P_matrices(self, horizon, discount=1.):
        if horizon == 0:
            return [self.Q_final]
        next_Ps = self.P_matrices(horizon - 1, discount)
        P_next = next_Ps[0]
        M = self.M_mix + discount * self.B.T @ P_next @ self.A
        inverse = np.linalg.inv(self.R + discount * self.B.T @ P_next @ self.B)
        P = self.Q + discount * (self.A.T @ P_next @ self.A) - (M.T @ inverse @ M)
        return [P] + next_Ps

    def optimal_gains(self, horizon, discount=1.):
        Ps = self.P_matrices(horizon, discount)
        inverses = [np.linalg.inv(self.R + discount * self.B.T @ Ps[h + 1] @ self.B) for h in range(horizon)]
        return [- inverses[h] @ (self.M_mix + discount * self.B.T @ Ps[h + 1] @ self.A) for h in range(horizon)]


    @staticmethod
    def _init_term(P, mean, std):
        return np.dot(mean, P @ mean) + np.matrix.trace(P @ np.diag(std ** 2))

    def optimal_return(self, horizon, policy_std=0., discount=1.):
        # only works with scalar stds
        Ps = self.P_matrices(horizon, discount)

        if isinstance(self.init_mean, list):
            init_term = np.mean([self._init_term(Ps[0], s, self.init_std) for s in self.init_mean])
        else:
            init_term = self._init_term(Ps[0], self.init_mean, self.init_std)
        state_noise_term = sum(np.trace(discount * np.diag(self.noise_std**2) @ Ps[h+1]) for h in range(horizon))
        action_noise_term = sum(np.trace((self.R + discount * self.B.T @ Ps[h+1] @ self.B) * policy_std**2)
                                for h in range(horizon))
        noise_term = state_noise_term + action_noise_term

        return - init_term - noise_term


if __name__ == "__main__":
    horizon = 100
    env = LQR(A = np.eye(2),
              B = np.eye(2),
              init_mean=1., init_std=0.)
    discount = 0.9
    policy_std = 0.2
    seed = 42
    optimal_gain = env.discounted_optimal_gain(discount)
    print(optimal_gain)
    print(env.discounted_optimal_return(discount, policy_std))

    optimal_param = optimal_gain.ravel()
    pol = LinearGaussianPolicy.make(env, std_init=policy_std)
    pol.set_params(optimal_param)
    ret2 = estimate_average_return(env, pol,
                                   n_episodes=1000,
                                   horizon=None,
                                   discount=discount,
                                   rng=np.random.default_rng(seed))
    print("Simulated optimal:")
    print(ret2)

    # Value functions
    s = 1 * np.ones(2)
    a = s @ optimal_gain
    K = optimal_gain

    print("V:")
    v = env.discounted_v(s, K, discount, policy_std)
    print(v)

    print("Q:")
    q = env.discounted_q(s, a, K, discount, policy_std)
    print(q)

    print("phi(s,a)")
    print(env.q_representation(s, a))
