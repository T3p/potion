from potion.simulation.trajectory_generators import generate_batch
from potion.estimators.gradients import gpomdp_estimator, reinforce_estimator, nonstationary_pg_estimator
from potion.algorithms._common import capped_batch_size, initialize_run
import numpy as np
import warnings


def srvrpg(env, policy, *,
           horizon=100,
           discount=1.,
           step_size=1e-4,
           batch_size=100,
           mini_batch_size=10,
           epoch_length=10,
           max_iterations=1000,
           max_trajectories=None,
           estimator='gpomdp',
           baseline='average',
           seed=None,
           logger=None,
           n_jobs=1,
           verbose=True):
    """Run SRVR-PG training until an iteration or trajectory limit is met."""
    if max_iterations is None and max_trajectories is None:
        raise ValueError("max_iterations and max_trajectories cannot both be None")

    rng, evaluation_rng, logger = initialize_run(seed, logger)

    if verbose:
        print("\n*** SRVR-PG ***\n")

    # Initialize logger
    logger.initialize(env, policy, horizon, discount, evaluation_rng)

    if estimator not in ["reinforce", "gpomdp", "nonstationary"]:
        warnings.warn("Unknown gradient estimator: will default to gpomdp", UserWarning)
    if estimator == "reinforce":
        gradient_estimator = reinforce_estimator
    elif estimator == "nonstationary":
        gradient_estimator = nonstationary_pg_estimator
    else:
        gradient_estimator = gpomdp_estimator

    estimator_discount = discount if horizon is not None else 1.

    # Learning loop
    it = 1
    total_trajectories = 0
    while ((max_iterations is None or it <= max_iterations)
           and (max_trajectories is None or total_trajectories < max_trajectories)):
        if verbose:
            iteration = "{} of {}".format(it, max_iterations) if max_iterations is not None else str(it)
            print("\nIteration {} running...".format(iteration))

        # Start the epoch with a large-batch gradient estimate and immediately
        # update the policy, as in the first SRVR-PG recursion step.
        actual_batch_size = capped_batch_size(
            batch_size, total_trajectories, max_trajectories
        )
        batch = generate_batch(env, policy, actual_batch_size, horizon,
                               rng=rng,
                               discount=discount,
                               parallel=(n_jobs > 1),
                               n_jobs=n_jobs)
        total_trajectories += len(batch)
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
            print("Epoch 1 of {} completed!".format(epoch_length))
            print("Gradient norm = {}".format(np.linalg.norm(gradient)))
            print("Parameter delta norm = {}".format(np.linalg.norm(delta)))

        epoch = 2
        while (epoch <= epoch_length
               and (max_trajectories is None or total_trajectories < max_trajectories)):
            if verbose:
                print("Epoch {} of {} running...".format(epoch, epoch_length))

            # Sample with the current policy and recursively correct the
            # preceding gradient estimate using the preceding policy.
            actual_mini_batch_size = capped_batch_size(
                mini_batch_size, total_trajectories, max_trajectories
            )
            batch = generate_batch(env, policy, actual_mini_batch_size, horizon,
                                   rng=rng,
                                   discount=discount,
                                   parallel=(n_jobs > 1),
                                   n_jobs=n_jobs)
            total_trajectories += len(batch)
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

            gradient = gradient + current_gradient - previous_batch_gradient

            if callable(step_size):
                delta = step_size(gradient, reset=False)
            else:
                delta = step_size * gradient

            previous_params = current_params
            policy.set_params(current_params + delta)

            if verbose:
                print("GRADIENT = ", gradient)
                print("Epoch {} of {} completed!".format(epoch, epoch_length))
                print("Gradient norm = {}".format(np.linalg.norm(gradient)))
                print("Parameter delta norm = {}".format(np.linalg.norm(delta)))
            epoch += 1

        if verbose:
            print("Iteration {} completed!".format(iteration))
        it += 1

    # Cleanup
    logger.close()
