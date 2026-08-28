import numpy as np


class Adam:
    def __init__(self, alpha=1e-3, beta_1=0.9, beta_2=0.999, eps=1e-8):
        self._alpha = alpha
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._eps = eps
        self._m = 0.
        self._v = 0.
        self._t = 0

    def _reset(self):
        self._m = 0.
        self._v = 0.
        self._t = 0

    def __call__(self, gradient, reset=False):
        if reset:
            self._reset()

        self._t += 1
        self._m = self._beta_1 * self._m + (1 - self._beta_1) * gradient
        self._v = self._beta_2 * self._v + (1 - self._beta_2) * gradient**2
        m_hat = self._m / (1 - self._beta_1**self._t)
        v_hat = self._v / (1 - self._beta_2**self._t)

        return self._alpha * m_hat / (np.sqrt(v_hat) + self._eps)


class RMS:
    def __init__(self, alpha=1e-3, beta=0.9, eps=1e-8):
        self._alpha = alpha
        self._beta = beta
        self._eps = eps
        self._q = 0.

    def _reset(self):
        self._q = 0.

    def __call__(self, gradient, reset=False):
        if reset:
            self._reset()

        second_moment = gradient**2
        self._q = self._beta * self._q + (1 - self._beta) * second_moment

        return self._alpha * gradient / (np.sqrt(self._q) + self._eps)
