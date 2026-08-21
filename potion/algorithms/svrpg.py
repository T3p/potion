from potion.simulation.trajectory_generators import generate_batch
from potion.estimators.gradients import gpomdp_estimator, reinforce_estimator, nonstationary_pg_estimator
from potion.evaluation.loggers import EpisodicPerformanceLogger
import numpy as np
import warnings


def svrpg(env, policy, *,
          horizon=100,
          discount=1.,
          step_size=1e-4,
          batch_size=100,
          mini_batch_size=22,
          epoch_length=10,
          max_iterations=1000,
          estimator='gpomdp',
          baseline='average',
          seed=None,
          logger=EpisodicPerformanceLogger(),
          n_jobs=1,
          verbose=True):
    rng = np.random.default_rng(seed)

    if verbose:
        print("\n*** SVRPG ***\n")

    # Initialize logger
    logger.initialize(env, policy, horizon, discount, rng)

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
    while it <= max_iterations:
        if verbose:
            print("\nIteration {} of {} running...".format(it, max_iterations))

        snapshot_params = policy.parameters.copy()

        # Estimate the gradient at the snapshot policy using a large batch
        snapshot_batch = generate_batch(env, policy, batch_size, horizon,
                                        rng=rng,
                                        discount=discount,
                                        parallel=(n_jobs > 1),
                                        n_jobs=n_jobs)
        logger.submit(snapshot_batch, policy)
        snapshot_gradient = gradient_estimator(snapshot_batch, estimator_discount, policy, baseline)

        epoch = 1
        while epoch <= epoch_length:
            if verbose:
                print("Epoch {} of {} running...".format(epoch, epoch_length))

            # Collect trajectories with the current policy
            batch = generate_batch(env, policy, mini_batch_size, horizon,
                                   rng=rng,
                                   discount=discount,
                                   parallel=(n_jobs > 1),
                                   n_jobs=n_jobs)
            logger.submit(batch, policy)

            # Compute the SVRPG correction on the same trajectories. The
            # off-policy snapshot samples include p_snapshot / p_current.
            current_gradient_samples = gradient_estimator(
                batch, estimator_discount, policy, baseline, average=False
            )
            current_params = policy.parameters.copy()
            try:
                policy.set_params(snapshot_params)
                snapshot_gradient_samples = gradient_estimator(
                    batch, estimator_discount, policy, baseline,
                    average=False, off_policy=True
                )
            finally:
                policy.set_params(current_params)

            correction = np.mean(current_gradient_samples - snapshot_gradient_samples, axis=0)
            gradient = snapshot_gradient + correction

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
                print("GRADIENT = ", gradient)
                print("Epoch {} of {} completed!".format(epoch, epoch_length))
                print("Gradient norm = {}".format(np.linalg.norm(gradient)))
                print("Parameter delta norm = {}".format(np.linalg.norm(delta)))
            # Next epoch
            epoch += 1

        if verbose:
            print("Iteration {} of {} completed!".format(it, max_iterations))
        # Next iteration
        it += 1

    # Cleanup
    logger.close()
