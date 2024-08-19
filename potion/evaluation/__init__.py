from abc import ABC, abstractmethod


class Logger(ABC):
    @abstractmethod
    def initialize(self, env, policy, max_trajectory_len, discount, rng):
        pass

    @abstractmethod
    def submit_policy(self, policy):
        pass

    @abstractmethod
    def submit_trajectories(self, trajectories):
        pass

    @abstractmethod
    def close(self):
        pass
