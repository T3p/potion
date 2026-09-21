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


def svrpg(
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
    """Run SVRPG training until an iteration or trajectory limit is met."""
    if max_iterations is None and max_trajectories is None:
        raise ValueError(
            "max_iterations and max_trajectories cannot both be None"
        )

    rng, evaluation_rng, logger = initialize_run(seed, logger)

    if verbose:
        print("\n*** SVRPG ***\n")

    # Initialize logger
    logger.initialize(env, policy, horizon, discount, evaluation_rng)

    progress_bar = initialize_progress_bar(
        max_iterations, max_trajectories, "SVRPG"
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

        snapshot_params = policy.parameters.copy()

        # Estimate the gradient at the snapshot policy using a large batch
        actual_batch_size = capped_batch_size(
            batch_size, total_trajectories, max_trajectories
        )
        snapshot_batch = generate_batch(
            env,
            policy,
            actual_batch_size,
            horizon,
            rng=rng,
            discount=discount,
            parallel=(n_jobs > 1),
            n_jobs=n_jobs,
        )
        total_trajectories += len(snapshot_batch)
        if max_trajectories is not None:
            update_progress_bar(
                progress_bar, max_trajectories, len(snapshot_batch)
            )
        logger.submit(snapshot_batch, policy)
        snapshot_gradient = gradient_estimator(
            snapshot_batch, estimator_discount, policy, baseline
        )

        epoch = 1
        while epoch <= epoch_length and (
            max_trajectories is None or total_trajectories < max_trajectories
        ):
            if verbose:
                print(f"Epoch {epoch} of {epoch_length} running...")

            # Collect trajectories with the current policy
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

            # Compute the SVRPG correction on the same trajectories. The
            # off-policy snapshot samples include p_snapshot / p_current.
            current_gradient = gradient_estimator(
                batch, estimator_discount, policy, baseline
            )
            current_params = policy.parameters.copy()
            try:
                policy.set_params(snapshot_params)
                snapshot_batch_gradient = gradient_estimator(
                    batch,
                    estimator_discount,
                    policy,
                    baseline,
                    off_policy=True,
                )
            finally:
                policy.set_params(current_params)

            gradient = (
                snapshot_gradient + current_gradient - snapshot_batch_gradient
            )

            # Compute update vector
            if callable(step_size):
                delta = step_size(gradient, reset=(epoch == 1))
            else:
                delta = step_size * gradient

            # Update policy parameters
            params = policy.parameters
            new_params = params + delta
            policy.set_params(new_params)

            if verbose:
                print(f"Epoch {epoch} of {epoch_length} completed!")
                print(f"Gradient norm = {np.linalg.norm(gradient)}")
                print(f"Parameter delta norm = {np.linalg.norm(delta)}")
            # Next epoch
            epoch += 1

        if verbose:
            print(f"Iteration {iteration} completed!")
        # Next iteration
        it += 1

        if max_trajectories is None:
            update_progress_bar(progress_bar, max_trajectories)

    # Cleanup
    progress_bar.close()
    logger.close()
