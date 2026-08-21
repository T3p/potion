from abc import ABC, abstractmethod


class Logger(ABC):
    @abstractmethod
    def initialize(self, env, policy, max_trajectory_len, discount, rng):
        pass

    @abstractmethod
    def submit(self, trajectories, policy):
        pass

    @abstractmethod
    def close(self):
        pass
