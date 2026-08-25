from potion.simulation.trajectory_generators import (apply_discount, apply_mask,
                                                     estimate_average_return)
import numpy as np
import pandas as pd
import os
import warnings
from potion.evaluation import Logger


_ANSI_COLORS = {
    "black": "\033[30m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
}
_ANSI_RESET = "\033[0m"


class SilentLogger(Logger):
    def initialize(self, env, policy, max_trajectory_len, discount, rng):
        pass

    def submit(self, trajectories, policy):
        pass

    def close(self):
        pass


class _EpisodicLogger(Logger):
    def __init__(self, save_every, path, color):
        if color is not None and color not in _ANSI_COLORS:
            raise ValueError("Unknown logger color: {}".format(color))
        self.save_every = save_every
        self.path = path
        self.color = color
        self._reset_run_state()

    def _reset_run_state(self):
        """Clear all state that belongs to one algorithm run."""
        self.tot_traj = 0
        self.buffer = []
        self.blank = True
        self._auc = 0.
        self._auc_start_trajectory = None
        self._auc_last_trajectory = None
        self._auc_last_return = None
        self._normalized_auc = None

    @property
    def normalized_auc(self):
        return self._normalized_auc

    def _print(self, *values):
        if self.color is None:
            print(*values)
            return
        print(_ANSI_COLORS[self.color], end="")
        print(*values, end="")
        print(_ANSI_RESET)

    def _update_auc(self, tot_trajectories, ret):
        if self._auc_start_trajectory is None:
            self._auc_start_trajectory = tot_trajectories
            normalized_auc = ret
        else:
            interval = tot_trajectories - self._auc_last_trajectory
            self._auc += interval * (self._auc_last_return + ret) / 2.
            span = tot_trajectories - self._auc_start_trajectory
            normalized_auc = self._auc / span

        self._auc_last_trajectory = tot_trajectories
        self._auc_last_return = ret
        self._normalized_auc = normalized_auc
        return normalized_auc

    def _append_return_record(self, tot_trajectories, ret, normalized_auc):
        record = {
            "tot_trajectories": tot_trajectories,
            "return": ret,
            "normalized_auc": normalized_auc,
        }
        self.buffer.append(record)

    def _record_return(self, tot_trajectories, ret):
        normalized_auc = self._update_auc(tot_trajectories, ret)
        self._append_return_record(tot_trajectories, ret, normalized_auc)
        return normalized_auc

    def save(self):
        if not self.buffer:
            return
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


class EpisodicOnlineLogger(_EpisodicLogger):
    def __init__(self, log_every=1, save_every=1000, verbose=True,
                 override_discount=None,
                 log_params=False,
                 path="tmp_log.csv",
                 color="cyan"):
        super().__init__(save_every, path, color)
        self.log_every = log_every
        self.verbose = verbose
        self.log_params = log_params
        self.override_discount = override_discount
        self.discount = None
        self.policy = None

    def initialize(self, env, policy, horizon, discount, rng):
        self._reset_run_state()
        self.policy = policy
        self.discount = (
            discount if self.override_discount is None else self.override_discount
        )
        if self.verbose:
            self._print(">> Episodic Online Logger ***")
            if self.log_params:
                self._print(">> Policy parameters: ", self.policy.parameters)
            self._print()

    def submit(self, trajectories, policy):
        for traj in trajectories:
            self.tot_traj += 1
            _, _, rewards, alive, _ = traj
            rewards = apply_mask(rewards, alive)
            discounted_rewards = apply_discount(rewards, self.discount)
            ret = np.sum(discounted_rewards)
            normalized_auc = self._update_auc(self.tot_traj, ret)
            if (self.tot_traj - 1) % self.log_every == 0:
                self._append_return_record(self.tot_traj, ret, normalized_auc)
                if self.verbose:
                    self._print(">> Episodic Online Logger")
                    self._print(">> Online trajectory {} obtained return {}".format(self.tot_traj, ret))
                    self._print(">> Normalized area under learning curve: {}".format(normalized_auc))
                    if self.log_params:
                        self._print(">> Policy parameters: ", policy.parameters)

            if self.tot_traj % self.save_every == 0:
                self.save()


class EpisodicTestLogger(_EpisodicLogger):
    def __init__(self, log_every=1, save_every=1000, verbose=True,
                 override_discount=None,
                 log_params=False,
                 path="tmp_log.csv",
                 n_test=100,
                 color="cyan",
                 keep_records=False):
        if not isinstance(n_test, (int, np.integer)) or n_test < 0:
            raise ValueError("n_test must be a non-negative integer")
        super().__init__(save_every, path, color)
        self.log_every = log_every
        self.verbose = verbose
        self.log_params = log_params
        self.n_test = int(n_test)
        self.override_discount = override_discount
        self.discount = None
        self.env = None
        self.horizon = None
        self.rng = None
        self.keep_records = keep_records
        self.records = []

    def _append_return_record(self, tot_trajectories, ret, normalized_auc):
        record = {
            "tot_trajectories": tot_trajectories,
            "return": ret,
            "normalized_auc": normalized_auc,
        }
        if self.keep_records:
            self.records.append(record.copy())
        self.buffer.append(record)

    def initialize(self, env, policy, horizon, discount, rng):
        self._reset_run_state()
        self.records = []
        self.env = env
        self.horizon = horizon
        self.rng = rng
        self.discount = (
            discount if self.override_discount is None else self.override_discount
        )

        if self.verbose:
            self._print(">> Episodic Test Logger ***")
        if self.n_test > 0:
            ret = self._evaluate_policy(policy)
            normalized_auc = self._record_return(0, ret)
            if self.verbose:
                self._print(">> Initial policy obtained average return {} over {} test trajectories".format(
                    ret, self.n_test))
                self._print(">> Normalized area under learning curve: {}".format(normalized_auc))
                if self.log_params:
                    self._print(">> Policy parameters: ", policy.parameters)
        if self.verbose:
            self._print()

    def submit(self, trajectories, policy):
        previous_tot_traj = self.tot_traj
        for _ in trajectories:
            self.tot_traj += 1
            if self.tot_traj % self.log_every == 0 and self.n_test > 0:
                ret = self._evaluate_policy(policy)
                normalized_auc = self._record_return(self.tot_traj, ret)
                if self.verbose:
                    self._print(">> Episodic Test Logger")
                    self._print(">> Policy after {} online trajectories obtained average return {} "
                                "over {} test trajectories".format(self.tot_traj, ret, self.n_test))
                    self._print(">> Normalized area under learning curve: {}".format(normalized_auc))
                    if self.log_params:
                        self._print(">> Policy parameters: ", policy.parameters)

        crossed_save_interval = (self.tot_traj // self.save_every
                                 > previous_tot_traj // self.save_every)
        if crossed_save_interval:
            self.save()

    def _evaluate_policy(self, policy):
        return estimate_average_return(
            self.env,
            policy,
            self.n_test,
            self.horizon,
            self.rng,
            discount=self.discount,
        )
