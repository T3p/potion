from potion.simulation.trajectory_generators import generate_batch
from potion.estimators.gradients import gpomdp_estimator, reinforce_estimator
from potion.evaluation.loggers import EpisodicPerformanceLogger
import numpy as np
import warnings


def reinforce(env, policy, *,
              horizon=100,
              discount=1.,
              step_size=1e-4,
              batch_size=100,
              max_iterations=1000,
              estimator='gpomdp',
              baseline='average',
              seed=None,
              logger=EpisodicPerformanceLogger(),
              n_jobs=1,
              verbose=True):
    rng = np.random.default_rng(seed)

    if verbose:
        print("\n*** REINFORCE ***\n")

    # Initialize logger
    logger.initialize(env, policy, horizon, discount, rng)

    # Learning loop
    it = 1
    while it <= max_iterations:
        if verbose:
            print("\nIteration {} of {} running...".format(it, max_iterations))
        # Collect trajectories
        batch = generate_batch(env, policy, batch_size, horizon, rng,
                               parallel=(n_jobs > 1),
                               n_jobs=n_jobs)
        # Log
        logger.submit_trajectories(batch)
        logger.submit_policy(policy)

        # Estimate policy gradient
        if estimator not in ["reinforce", "gpomdp"]:
            warnings.warn("Unknown gradient estimator: will default to gpomdp", UserWarning)
        if estimator == "reinforce":
            gradient = reinforce_estimator(batch, discount, policy, baseline)
        else:
            gradient = gpomdp_estimator(batch, discount, policy, baseline)

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
            print("Iteration {} of {} completed!".format(it, max_iterations))
            print("Gradient norm = {}".format(np.linalg.norm(gradient)))
            print("Parameter delta norm = {}".format(np.linalg.norm(delta)))
        # Next iteration
        it += 1

    # Cleanup
    logger.close()
    
