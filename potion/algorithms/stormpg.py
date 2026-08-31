import warnings

import numpy as np
from tqdm.auto import tqdm

from potion.algorithms._common import capped_batch_size, initialize_run
from potion.estimators.gradients import (
    gpomdp_estimator,
    nonstationary_pg_estimator,
    reinforce_estimator,
)
from potion.simulation.trajectory_generators import generate_batch


def stormpg(
    env,
    policy,
    *,
    horizon=100,
    discount=1.0,
    step_size=1e-4,
    batch_size=100,
    mini_batch_size=10,
    momentum_parameter=0.9,
    max_iterations=1000,
    max_trajectories=None,
    estimator="gpomdp",
    baseline="average",
    seed=None,
    logger=None,
    n_jobs=1,
    verbose=True,
):
    """Run STORM-PG training until an iteration or trajectory limit is met."""
    if max_iterations is None and max_trajectories is None:
        raise ValueError(
            "max_iterations and max_trajectories cannot both be None"
        )
    if not 0.0 < momentum_parameter < 1.0:
        raise ValueError(
            "momentum parameter should be strictly between zero and one"
        )

    rng, evaluation_rng, logger = initialize_run(seed, logger)

    if verbose:
        print("\n*** STORM-PG ***\n")

    # Initialize logger
    logger.initialize(env, policy, horizon, discount, evaluation_rng)

    if (max_iterations is not None and max_iterations < 1) or (
        max_trajectories is not None and max_trajectories < 1
    ):
        logger.close()
        return

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

    estimator_discount = discount if horizon is not None else 1.0

    # Progress bar initialization
    total_budget = (
        max_trajectories if max_trajectories is not None else max_iterations
    )
    unit_name = "traj" if max_trajectories is not None else "it"
    pbar = tqdm(
        total=total_budget,
        desc="STORM-PG",
        unit=unit_name,
        leave=True,
    )

    # Estimate the initial gradient using a large batch.
    initial_batch_size = capped_batch_size(batch_size, 0, max_trajectories)
    batch = generate_batch(
        env,
        policy,
        initial_batch_size,
        horizon,
        rng=rng,
        discount=discount,
        parallel=(n_jobs > 1),
        n_jobs=n_jobs,
    )
    total_trajectories = len(batch)

    #  Update progress bar
    if max_trajectories is not None:
        pbar.update(len(batch))

    logger.submit(batch, policy)
    gradient = gradient_estimator(batch, estimator_discount, policy, baseline)

    # Learning loop
    it = 1
    while max_iterations is None or it <= max_iterations:
        if verbose:
            iteration = (
                f"{it} of {max_iterations}"
                if max_iterations is not None
                else str(it)
            )
            print(f"\nIteration {iteration} running...")

        if callable(step_size):
            delta = step_size(gradient)
        else:
            delta = step_size * gradient

        previous_params = policy.parameters.copy()
        policy.set_params(previous_params + delta)

        if verbose:
            print(f"Iteration {iteration} completed!")
            print(f"Gradient norm = {np.linalg.norm(gradient)}")
            print(f"Parameter delta norm = {np.linalg.norm(delta)}")

        if max_trajectories is None:
            pbar.update(1)

        it += 1
        if (max_iterations is not None and it > max_iterations) or (
            max_trajectories is not None
            and total_trajectories >= max_trajectories
        ):
            break

        # Sample with the updated policy and form the momentum-corrected
        # gradient estimate for the following update.
        actual_mini_batch_size = capped_batch_size(
            mini_batch_size, total_trajectories, max_trajectories
        )
        batch = generate_batch(
            env,
            policy,
            actual_mini_batch_size,
            horizon,
            rng=rng,
            discount=discount,
            parallel=(n_jobs > 1),
            n_jobs=n_jobs,
        )
        total_trajectories += len(batch)

        # Update progress bar
        if max_trajectories is not None:
            pbar.update(len(batch))

        logger.submit(batch, policy)

        current_gradient = gradient_estimator(
            batch, estimator_discount, policy, baseline
        )
        current_params = policy.parameters.copy()
        try:
            policy.set_params(previous_params)
            previous_batch_gradient = gradient_estimator(
                batch, estimator_discount, policy, baseline, off_policy=True
            )
        finally:
            policy.set_params(current_params)

        decay = 1.0 - momentum_parameter
        gradient = current_gradient + decay * (
            gradient - previous_batch_gradient
        )

    # Cleanup
    pbar.close()
    logger.close()
