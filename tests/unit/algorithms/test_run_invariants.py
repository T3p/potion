import inspect

import pytest

from potion.algorithms import (
    def_pagepg,
    def_srvrpg,
    def_stormpg,
    def_svrpg,
    pagepg,
    reinforce,
    srvrpg,
    stormpg,
    svrpg,
)


ALGORITHMS = (
    reinforce,
    svrpg,
    def_svrpg,
    srvrpg,
    def_srvrpg,
    stormpg,
    def_stormpg,
    pagepg,
    def_pagepg,
)


class CountingLogger:
    def initialize(self, env, policy, horizon, discount, rng):
        self.tot_traj = 0

    def submit(self, trajectories, policy):
        self.tot_traj += len(trajectories)

    def close(self):
        pass


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_algorithm_logger_default_is_not_mutable(algorithm):
    assert inspect.signature(algorithm).parameters["logger"].default is None


@pytest.mark.parametrize("budget", [1, 3, 103, 137])
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_algorithms_honor_exact_awkward_trajectory_budgets(
        env, policy, algorithm, budget):
    logger = CountingLogger()
    arguments = {
        "horizon": 1,
        "discount": 0.9999,
        "step_size": 1e-4,
        "batch_size": 100,
        "max_iterations": None,
        "max_trajectories": budget,
        "baseline": "zero",
        "seed": 42,
        "logger": logger,
        "n_jobs": 1,
        "verbose": False,
    }
    if algorithm in (svrpg, def_svrpg, srvrpg, def_srvrpg):
        arguments.update(mini_batch_size=5, epoch_length=10)
    elif algorithm in (stormpg, def_stormpg):
        arguments.update(mini_batch_size=5, momentum_parameter=0.99)
    elif algorithm in (pagepg, def_pagepg):
        arguments.update(mini_batch_size=5, refresh_probability=0.8)

    algorithm(env, policy, **arguments)

    assert logger.tot_traj == budget
