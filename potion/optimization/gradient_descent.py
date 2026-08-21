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

    def __call__(self, grad):
        self._t += 1
        self.m = self._beta_1 * self._m + (1 - self._beta_1) * grad
        self.v = self._beta_2 * self._v + (1 - self._beta_2) * grad**2
        m_hat = self.m / (1 - self._beta_1**self._t)
        v_hat = self.v / (1 - self._beta_2**self._t)
        return self._alpha * m_hat / (np.sqrt(v_hat) + self._eps)

