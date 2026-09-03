"""Normalized policy gradient with implicit gradient transport."""

import warnings

import numpy as np

from potion.algorithms._common import capped_batch_size, initialize_run
from potion.estimators.gradients import (
    gpomdp_estimator,
    nonstationary_pg_estimator,
    reinforce_estimator,
)
from potion.simulation.trajectory_generators import generate_batch


def npg_igt(env, policy, *,
            horizon=100,
            discount=1.,
            step_size=1e-4,
            batch_size=100,
            momentum_parameter=0.9,
            max_iterations=1000,
            max_trajectories=None,
            estimator='gpomdp',
            baseline='average',
            seed=None,
            logger=None,
            n_jobs=1,
            verbose=True):
    """Run normalized PG with implicit gradient transport.

    This implements Algorithm 1 (N-PG-IGT) from Fatkhullin et al.,
    "Stochastic Policy Gradient Methods: Improved Sample Complexity for
    Fisher-non-degenerate Policies". ``momentum_parameter`` is the constant
    value of eta in the paper and ``step_size`` is gamma. As in
    :func:`reinforce`, a batch may be used to estimate each stochastic
    gradient; set ``batch_size=1`` for the paper's single-trajectory update.
    """
    if max_iterations is None and max_trajectories is None:
        raise ValueError("max_iterations and max_trajectories cannot both be None")
    if not 0. < momentum_parameter <= 1.:
        raise ValueError(
            "momentum parameter should be greater than zero and at most one"
        )

    rng, evaluation_rng, logger = initialize_run(seed, logger)

    if verbose:
        print("\n*** N-PG-IGT ***\n")

    logger.initialize(env, policy, horizon, discount, evaluation_rng)

    if estimator not in ["reinforce", "gpomdp", "nonstationary"]:
        warnings.warn(
            "Unknown gradient estimator: will default to gpomdp", UserWarning
        )
    if estimator == "reinforce":
        gradient_estimator = reinforce_estimator
    elif estimator == "nonstationary":
        gradient_estimator = nonstationary_pg_estimator
    else:
        gradient_estimator = gpomdp_estimator

    estimator_discount = discount if horizon is not None else 1.
    previous_params = policy.parameters.copy()
    direction = None
    it = 1
    total_trajectories = 0

    while ((max_iterations is None or it <= max_iterations)
           and (max_trajectories is None
                or total_trajectories < max_trajectories)):
        if verbose:
            iteration = (
                "{} of {}".format(it, max_iterations)
                if max_iterations is not None else str(it)
            )
            print("\nIteration {} running...".format(iteration))

        current_params = policy.parameters.copy()
        lookahead_scale = (1. - momentum_parameter) / momentum_parameter
        lookahead_params = current_params + lookahead_scale * (
            current_params - previous_params
        )

        # The trajectory and its score-function gradient must both use the
        # extrapolated policy. Restore the actual iterate before logging and
        # applying the normalized update.
        try:
            policy.set_params(lookahead_params)
            actual_batch_size = capped_batch_size(
                batch_size, total_trajectories, max_trajectories
            )
            batch = generate_batch(
                env,
                policy,
                actual_batch_size,
                horizon,
                rng=rng,
                discount=discount,
                parallel=(n_jobs > 1),
                n_jobs=n_jobs,
            )
            gradient = gradient_estimator(
                batch, estimator_discount, policy, baseline
            )
        finally:
            policy.set_params(current_params)

        total_trajectories += len(batch)
        logger.submit(batch, policy)

        # The first stochastic gradient initializes d_0. Subsequent estimates
        # follow d_t = (1 - eta) d_{t-1} + eta g(theta_tilde_t).
        if direction is None:
            direction = gradient
        else:
            direction = (
                (1. - momentum_parameter) * direction
                + momentum_parameter * gradient
            )

        direction_norm = np.linalg.norm(direction)
        if direction_norm == 0.:
            normalized_direction = np.zeros_like(direction)
        else:
            normalized_direction = direction / direction_norm

        if callable(step_size):
            delta = step_size(normalized_direction)
        else:
            delta = step_size * normalized_direction

        previous_params = current_params
        policy.set_params(current_params + delta)

        if verbose:
            print("Iteration {} completed!".format(iteration))
            print("Gradient norm = {}".format(np.linalg.norm(gradient)))
            print("Momentum norm = {}".format(direction_norm))
            print("Parameter delta norm = {}".format(np.linalg.norm(delta)))

        it += 1

    logger.close()
