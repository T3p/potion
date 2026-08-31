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


def reinforce(
    env,
    policy,
    *,
    horizon=100,
    discount=1.0,
    step_size=1e-4,
    batch_size=100,
    max_iterations=1000,
    max_trajectories=None,
    estimator="gpomdp",
    baseline="average",
    seed=None,
    logger=None,
    n_jobs=1,
    verbose=True,
):
    """Run policy-gradient training until an iteration or trajectory limit is met."""
    if max_iterations is None and max_trajectories is None:
        raise ValueError(
            "max_iterations and max_trajectories cannot both be None"
        )

    rng, evaluation_rng, logger = initialize_run(seed, logger)

    if verbose:
        print("\n*** REINFORCE ***\n")

    # Initialize logger
    logger.initialize(env, policy, horizon, discount, evaluation_rng)

    # Initialize progress bar
    total_budget = (
        max_trajectories if max_trajectories is not None else max_iterations
    )
    unit_name = "traj" if max_trajectories is not None else "it"
    pbar = tqdm(
        total=total_budget,
        desc="GPOMDP",
        unit=unit_name,
        leave=True,
    )

    # Learning loop
    it = 1
    total_trajectories = 0
    while (max_iterations is None or it <= max_iterations) and (
        max_trajectories is None or total_trajectories < max_trajectories
    ):
        if verbose:
            iteration = (
                f"{it} of {max_iterations}"
                if max_iterations is not None
                else str(it)
            )
            print(f"\nIteration {iteration} running...")

        # Collect trajectories
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
        total_trajectories += len(batch)

        # Update progress bar
        if max_trajectories is not None:
            pbar.update(len(batch))
        else:
            pbar.update(1)

        # Log
        logger.submit(batch, policy)

        # Estimate policy gradient
        estimator_discount = discount if horizon is not None else 1.0
        if estimator not in ["reinforce", "gpomdp", "nonstationary"]:
            warnings.warn(
                "Unknown gradient estimator: will default to gpomdp",
                UserWarning,
            )
        if estimator == "reinforce":
            gradient = reinforce_estimator(
                batch, estimator_discount, policy, baseline
            )
        elif estimator == "nonstationary":
            gradient = nonstationary_pg_estimator(
                batch, estimator_discount, policy, baseline
            )
        else:
            gradient = gpomdp_estimator(
                batch, estimator_discount, policy, baseline
            )

        # Compute update vector
        if callable(step_size):
            delta = step_size(gradient)
        else:
            delta = step_size * gradient

        # Update policy parameters
        params = policy.parameters
        new_params = params + delta
        policy.set_params(new_params)

        if verbose:
            print(f"Iteration {iteration} completed!")
            print(f"Gradient norm = {np.linalg.norm(gradient)}")
            print(f"Parameter delta norm = {np.linalg.norm(delta)}")
        # Next iteration
        it += 1

    # Cleanup
    pbar.close()
    logger.close()
