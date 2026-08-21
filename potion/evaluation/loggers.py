from potion.simulation.trajectory_generators import (apply_discount, apply_mask,
                                                     blackbox_simulate_episode,
                                                     blackbox_simulate_infinite_trajectory)
import numpy as np
import pandas as pd
import os
import warnings
from potion.evaluation import Logger


class SilentLogger(Logger):
    def initialize(self, env, policy, max_trajectory_len, discount, rng):
        pass

    def submit(self, trajectories, policy):
        pass

    def close(self):
        pass


class EpisodicPerformanceLogger(Logger):
    def __init__(self, log_every=1, save_every=1000, verbose=True,
                 override_discount=False,
                 log_params=False,
                 path="tmp_log.csv"):
        self.log_every = log_every
        self.save_every = save_every
        self.verbose = verbose
        self.log_params = log_params
        self.discount = override_discount
        self.policy = None
        self.path = path

        self.tot_traj = 0
        self.average_return = 0
        self.buffer = []
        self.blank = True

    def initialize(self, env, policy, horizon, discount, rng):
        self.policy = policy
        if not self.discount:
            self.discount = discount
        seed = rng.bit_generator.seed_seq.generate_state(1)[0]
        if horizon is not None:
            ret, _ = blackbox_simulate_episode(env, policy, horizon, seed, self.discount)
        else:
            ret, _ = blackbox_simulate_infinite_trajectory(env, policy, self.discount, seed)
        record = {"tot_trajectories": 0, "return": ret}
        self.buffer.append(record)
        if self.verbose:
            print(">> Episodic Performance Logger ***")
            print(">> Initial policy obtained return {}".format(ret))
            if self.log_params:
                print(">> Policy parameters: ", self.policy.parameters)
            print()

    def submit(self, trajectories, policy):
        for traj in trajectories:
            self.tot_traj += 1
            if self.tot_traj % self.log_every == 0:
                _, _, rewards, alive = traj
                rewards = apply_mask(rewards, alive)
                discounted_rewards = apply_discount(rewards, self.discount)
                ret = np.sum(discounted_rewards)
                record = {"tot_trajectories": self.tot_traj, "return": ret}
                self.buffer.append(record)
                if self.verbose:
                    print(">> Episodic Performance Logger")
                    print(">> Policy after {} trajectories obtained return {}".format(self.tot_traj, ret))
                    if self.log_params:
                        print(">> Policy parameters: ", policy.parameters)

            if self.tot_traj % self.save_every == 0:
                self.save()

    def save(self):
        df = pd.DataFrame.from_records(self.buffer)
        self.buffer = []

        if self.path is None:
            return

        try:
            if self.blank:
                if os.path.exists(self.path):
                    warnings.warn("Logger is overriding file {}".format(self.path), UserWarning)
                with open(self.path, "w") as file:
                    df.to_csv(file, index=False, header=True)
                self.blank = False
            else:
                with open(self.path, "a") as file:
                    df.to_csv(file, index=False, header=False)
        except Exception as e:
            warnings.warn("Could not save log due to the following error: {}".format(repr(e)), UserWarning)

    def close(self):
        self.save()
