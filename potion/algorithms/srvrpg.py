import warnings

import numpy as np

from potion.algorithms._common import (
    capped_batch_size,
    initialize_progress_bar,
    initialize_run,
    update_progress_bar,
)
from potion.estimators.gradients import (
    gpomdp_estimator,
    nonstationary_pg_estimator,
    reinforce_estimator,
)
from potion.simulation.trajectory_generators import generate_batch


def srvrpg(
    env,
    policy,
    *,
    horizon=100,
    discount=1.0,
    step_size=1e-4,
    batch_size=100,
    mini_batch_size=10,
    epoch_length=10,
    max_iterations=1000,
    max_trajectories=None,
    estimator="gpomdp",
    baseline="average",
    seed=None,
    logger=None,
    n_jobs=1,
    verbose=True,
):
    """Run SRVR-PG training until an iteration or trajectory limit is met."""
    if max_iterations is None and max_trajectories is None:
        raise ValueError(
            "max_iterations and max_trajectories cannot both be None"
        )

    rng, evaluation_rng, logger = initialize_run(seed, logger)

    if verbose:
        print("\n*** SRVR-PG ***\n")

    # Initialize logger
    logger.initialize(env, policy, horizon, discount, evaluation_rng)

    progress_bar = initialize_progress_bar(
        max_iterations, max_trajectories, "SRVRPG"
    )

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

        # Start the epoch with a large-batch gradient estimate and immediately
        # update the policy, as in the first SRVR-PG recursion step.
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
        if max_trajectories is not None:
            update_progress_bar(progress_bar, max_trajectories, len(batch))
        logger.submit(batch, policy)
        gradient = gradient_estimator(
            batch, estimator_discount, policy, baseline
        )

        if callable(step_size):
            delta = step_size(gradient, reset=True)
        else:
            delta = step_size * gradient

        previous_params = policy.parameters.copy()
        policy.set_params(previous_params + delta)

        if verbose:
            print("GRADIENT = ", gradient)
            print(f"Epoch 1 of {epoch_length} completed!")
            print(f"Gradient norm = {np.linalg.norm(gradient)}")
            print(f"Parameter delta norm = {np.linalg.norm(delta)}")

        epoch = 2
        while epoch <= epoch_length and (
            max_trajectories is None or total_trajectories < max_trajectories
        ):
            if verbose:
                print(f"Epoch {epoch} of {epoch_length} running...")

            # Sample with the current policy and recursively correct the
            # preceding gradient estimate using the preceding policy.
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
            if max_trajectories is not None:
                update_progress_bar(progress_bar, max_trajectories, len(batch))
            logger.submit(batch, policy)

            current_gradient = gradient_estimator(
                batch, estimator_discount, policy, baseline
            )
            current_params = policy.parameters.copy()
            try:
                policy.set_params(previous_params)
                previous_batch_gradient = gradient_estimator(
                    batch,
                    estimator_discount,
                    policy,
                    baseline,
                    off_policy=True,
                )
            finally:
                policy.set_params(current_params)

            gradient = gradient + current_gradient - previous_batch_gradient

            if callable(step_size):
                delta = step_size(gradient, reset=False)
            else:
                delta = step_size * gradient

            previous_params = current_params
            policy.set_params(current_params + delta)

            if verbose:
                print("GRADIENT = ", gradient)
                print(f"Epoch {epoch} of {epoch_length} completed!")
                print(f"Gradient norm = {np.linalg.norm(gradient)}")
                print(f"Parameter delta norm = {np.linalg.norm(delta)}")
            epoch += 1

        if verbose:
            print(f"Iteration {iteration} completed!")
        it += 1
        if max_trajectories is None:
            update_progress_bar(progress_bar, max_trajectories)

    # Cleanup
    progress_bar.close()
    logger.close()
