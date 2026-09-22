import operator

import gymnasium as gym


class VectorizedBatchEnv(gym.Wrapper):
    """Scalar environment with an opt-in vector environment for collection.

    The wrapped scalar environment remains available to loggers and other code
    that expects the ordinary Gymnasium API. ``vector_env`` is reserved for
    trajectory batch collection.
    """

    def __init__(self, env_fn, num_envs):
        if not callable(env_fn):
            raise TypeError("env_fn should be callable")
        try:
            num_envs = operator.index(num_envs)
        except TypeError as exception:
            raise TypeError("num_envs should be an integer") from exception
        if num_envs < 1:
            raise ValueError("num_envs should be positive")

        env = env_fn()
        try:
            vector_env = gym.vector.SyncVectorEnv(
                [env_fn for _ in range(num_envs)]
            )
        except Exception:
            env.close()
            raise

        super().__init__(env)
        self.vector_env = vector_env
        self.num_envs = num_envs
        self._closed = False

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.vector_env.close()
        finally:
            super().close()
