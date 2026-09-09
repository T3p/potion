import gymnasium as gym
import numpy as np


class TanhActionWrapper(gym.Wrapper):
    """Expose latent actions and squash them before stepping a bounded env.

    The wrapped environment advertises an unbounded, float64 action space so
    trajectory generators retain the Gaussian action sampled before the tanh
    transform.  The underlying environment receives only controls within its
    original finite bounds.
    """

    def __init__(self, env, invalid_action_penalty=-1_000_000.0):
        super().__init__(env)
        physical_space = env.action_space
        if not isinstance(physical_space, gym.spaces.Box):
            raise TypeError("TanhActionWrapper requires a Box action space")

        self._physical_low = np.asarray(physical_space.low, dtype=float)
        self._physical_high = np.asarray(physical_space.high, dtype=float)
        if not (
            np.isfinite(self._physical_low).all()
            and np.isfinite(self._physical_high).all()
            and np.all(self._physical_low < self._physical_high)
        ):
            raise ValueError("The wrapped action space must have finite bounds")

        self._physical_midpoint = (
            self._physical_low + self._physical_high
        ) / 2.0
        self._physical_scale = (
            self._physical_high - self._physical_low
        ) / 2.0
        self._physical_dtype = physical_space.dtype
        self.invalid_action_penalty = float(invalid_action_penalty)
        if not np.isfinite(self.invalid_action_penalty):
            raise ValueError("invalid_action_penalty must be finite")

        self.action_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=physical_space.shape,
            dtype=np.float64,
        )
        self._last_observation = None

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._last_observation = observation
        return observation, info

    def step(self, action):
        latent_action = np.asarray(action, dtype=float)
        invalid = (
            latent_action.shape != self.action_space.shape
            or not np.isfinite(latent_action).all()
        )
        if invalid:
            if self._last_observation is None:
                raise RuntimeError("The environment must be reset before step")
            return (
                self._last_observation,
                self.invalid_action_penalty,
                True,
                False,
                {"invalid_action": True},
            )

        physical_action = (
            self._physical_midpoint
            + self._physical_scale * np.tanh(latent_action)
        ).astype(self._physical_dtype, copy=False)
        transition = self.env.step(physical_action)
        self._last_observation = transition[0]
        return transition
