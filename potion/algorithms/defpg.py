"""Defensive policy-gradient algorithms.

``defensive_parameter`` is the snapshot-policy mixture mass. In the paper's
notation, alpha is the current-policy mass, so alpha equals
``1 - defensive_parameter`` (and both are one half in the reproduction).
"""

from potion.simulation.trajectory_generators import generate_batch, unpack, apply_mask
from potion.estimators.gradients import gpomdp_estimator, reinforce_estimator, nonstationary_pg_estimator
from potion.algorithms._common import capped_batch_size, initialize_run
import numpy as np
import warnings


def _trajectory_log_probabilities(batch, policy):
    states, actions, _, alive, _ = unpack(batch)
    logps = np.zeros_like(alive, dtype=float)
    for t in range(states.shape[1]):
        active = alive[:, t]
        if np.any(active):
            logps[active, t] = policy.log_prob(states[active, t], actions[active, t], t)
    return np.sum(apply_mask(logps, alive), axis=1)


def _defensive_importance_weights(current_logps, snapshot_logps, defensive_parameter):
    """Return component/mixture weights.

    ``defensive_parameter`` is the snapshot-policy mixture mass; the paper's
    alpha is the current-policy mass, i.e. ``alpha = 1 - defensive_parameter``.
    """
    if not 0. <= defensive_parameter < 1.:
        raise ValueError(
            "defensive parameter should be greater than or equal to zero and less than one"
        )
    if current_logps.shape != snapshot_logps.shape:
        raise ValueError("current and snapshot log probabilities should have the same shape")

    if defensive_parameter == 0.:
        log_mixture = current_logps
    else:
        log_mixture = np.logaddexp(
            np.log1p(-defensive_parameter) + current_logps,
            np.log(defensive_parameter) + snapshot_logps,
        )
    current_weights = np.exp(current_logps - log_mixture)
    snapshot_weights = np.exp(snapshot_logps - log_mixture)
    return current_weights, snapshot_weights


def _defensive_gradient_samples(batch, policy, reference_params,
                                defensive_parameter, gradient_estimator,
                                estimator_discount, baseline):
    """Evaluate both mixture-corrected gradient samples on one batch."""
    current_params = policy.parameters.copy()
    current_logps = _trajectory_log_probabilities(batch, policy)
    try:
        policy.set_params(reference_params)
        reference_logps = _trajectory_log_probabilities(batch, policy)
    finally:
        policy.set_params(current_params)

    current_weights, reference_weights = _defensive_importance_weights(
        current_logps, reference_logps, defensive_parameter
    )
    current_samples = gradient_estimator(
        batch, estimator_discount, policy, baseline, average=False,
        importance_weights=current_weights
    )
    try:
        policy.set_params(reference_params)
        reference_samples = gradient_estimator(
            batch, estimator_discount, policy, baseline, average=False,
            importance_weights=reference_weights
        )
    finally:
        policy.set_params(current_params)
    return current_samples, reference_samples


def _generate_defensive_batch(env, policy, snapshot_params, defensive_parameter,
                              n_episodes, horizon, discount, rng, n_jobs):
    # At zero the mixture is exactly the current policy. Preserve SVRPG's RNG
    # consumption as well as its sampling distribution.
    if defensive_parameter == 0.:
        return generate_batch(env, policy, n_episodes, horizon,
                              rng=rng,
                              discount=discount,
                              parallel=(n_jobs > 1),
                              n_jobs=n_jobs)

    current_params = policy.parameters.copy()
    draw_snapshot = rng.random(n_episodes) < defensive_parameter
    n_snapshot = np.count_nonzero(draw_snapshot)
    n_current = n_episodes - n_snapshot

    # Draw component RNG seeds from the advancing training stream so repeated
    # calls cannot reuse the same episode seeds.
    current_seed, snapshot_seed = rng.integers(
        0, np.iinfo(np.uint32).max, size=2, dtype=np.uint32
    )
    current_rng = np.random.default_rng(current_seed)
    snapshot_rng = np.random.default_rng(snapshot_seed)

    batch = []
    if n_current:
        batch.extend(generate_batch(env, policy, n_current, horizon,
                                    rng=current_rng,
                                    discount=discount,
                                    parallel=(n_jobs > 1),
                                    n_jobs=n_jobs))

    if n_snapshot:
        try:
            policy.set_params(snapshot_params)
            batch.extend(generate_batch(env, policy, n_snapshot, horizon,
                                        rng=snapshot_rng,
                                        discount=discount,
                                        parallel=(n_jobs > 1),
                                        n_jobs=n_jobs))
        finally:
            policy.set_params(current_params)

    permutation = rng.permutation(n_episodes)
    return [batch[i] for i in permutation]


def def_svrpg(env, policy, *,
              horizon=100,
              discount=1.,
              step_size=1e-4,
              batch_size=100,
              mini_batch_size=10,
              epoch_length=10,
              max_iterations=1000,
              max_trajectories=None,
              defensive_parameter=0.5,
              estimator='gpomdp',
              baseline='average',
              seed=None,
              logger=None,
              n_jobs=1,
              verbose=True):
    """Run defensive SVRPG until an iteration or trajectory limit is met."""
    if max_iterations is None and max_trajectories is None:
        raise ValueError("max_iterations and max_trajectories cannot both be None")
    if not 0. <= defensive_parameter < 1.:
        raise ValueError(
            "defensive parameter should be greater than or equal to zero and less than one"
        )

    rng, evaluation_rng, logger = initialize_run(seed, logger)

    if verbose:
        print("\n*** DEF-SVRPG ***\n")

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

        snapshot_params = policy.parameters.copy()

        # Estimate the gradient at the snapshot policy using a large batch
        actual_batch_size = capped_batch_size(
            batch_size, total_trajectories, max_trajectories
        )
        snapshot_batch = generate_batch(env, policy, actual_batch_size, horizon,
                                        rng=rng,
                                        discount=discount,
                                        parallel=(n_jobs > 1),
                                        n_jobs=n_jobs)
        total_trajectories += len(snapshot_batch)
        logger.submit(snapshot_batch, policy)
        snapshot_gradient = gradient_estimator(
            snapshot_batch, estimator_discount, policy, baseline
        )

        epoch = 1
        while (epoch <= epoch_length
               and (max_trajectories is None or total_trajectories < max_trajectories)):
            if verbose:
                print("Epoch {} of {} running...".format(epoch, epoch_length))

            # Collect trajectories from the trajectory-level mixture of the
            # current and snapshot policies.
            actual_mini_batch_size = capped_batch_size(
                mini_batch_size, total_trajectories, max_trajectories
            )
            batch = _generate_defensive_batch(
                env, policy, snapshot_params, defensive_parameter,
                actual_mini_batch_size, horizon, discount, rng, n_jobs
            )
            total_trajectories += len(batch)
            logger.submit(batch, policy)

            current_params = policy.parameters.copy()
            if defensive_parameter == 0.:
                current_gradient = gradient_estimator(
                    batch, estimator_discount, policy, baseline
                )
            else:
                (current_gradient_samples,
                 snapshot_gradient_samples) = _defensive_gradient_samples(
                    batch, policy, snapshot_params, defensive_parameter,
                    gradient_estimator, estimator_discount, baseline
                )

            if defensive_parameter == 0.:
                try:
                    policy.set_params(snapshot_params)
                    snapshot_batch_gradient = gradient_estimator(
                        batch, estimator_discount, policy, baseline, off_policy=True
                    )
                finally:
                    policy.set_params(current_params)

            if defensive_parameter == 0.:
                gradient = snapshot_gradient + current_gradient - snapshot_batch_gradient
            else:
                correction = np.mean(
                    current_gradient_samples - snapshot_gradient_samples,
                    axis=0,
                )
                gradient = snapshot_gradient + correction

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
                print("Epoch {} of {} completed!".format(epoch, epoch_length))
                print("Gradient norm = {}".format(np.linalg.norm(gradient)))
                print("Parameter delta norm = {}".format(np.linalg.norm(delta)))
            # Next epoch
            epoch += 1

        if verbose:
            print("Iteration {} completed!".format(iteration))
        # Next iteration
        it += 1

    # Cleanup
    logger.close()


def def_srvrpg(env, policy, *,
               horizon=100,
               discount=1.,
               step_size=1e-4,
               batch_size=100,
               mini_batch_size=10,
               epoch_length=10,
               max_iterations=1000,
               max_trajectories=None,
               defensive_parameter=0.5,
               estimator='gpomdp',
               baseline='average',
               seed=None,
               logger=None,
               n_jobs=1,
               verbose=True):
    """Run defensive SRVR-PG until an iteration or trajectory limit is met."""
    if max_iterations is None and max_trajectories is None:
        raise ValueError("max_iterations and max_trajectories cannot both be None")
    if not 0. <= defensive_parameter < 1.:
        raise ValueError(
            "defensive parameter should be greater than or equal to zero and less than one"
        )

    rng, evaluation_rng, logger = initialize_run(seed, logger)

    if verbose:
        print("\n*** DEF-SRVR-PG ***\n")

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

        # Start the epoch with an on-policy large-batch estimate and update.
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

            # Sample from the trajectory-level mixture of the current and
            # preceding policies used by the recursive correction.
            actual_mini_batch_size = capped_batch_size(
                mini_batch_size, total_trajectories, max_trajectories
            )
            batch = _generate_defensive_batch(
                env, policy, previous_params, defensive_parameter,
                actual_mini_batch_size, horizon, discount, rng, n_jobs
            )
            total_trajectories += len(batch)
            logger.submit(batch, policy)

            current_params = policy.parameters.copy()
            if defensive_parameter == 0.:
                current_gradient = gradient_estimator(
                    batch, estimator_discount, policy, baseline
                )
            else:
                (current_gradient_samples,
                 previous_gradient_samples) = _defensive_gradient_samples(
                    batch, policy, previous_params, defensive_parameter,
                    gradient_estimator, estimator_discount, baseline
                )
            if defensive_parameter == 0.:
                try:
                    policy.set_params(previous_params)
                    previous_batch_gradient = gradient_estimator(
                        batch, estimator_discount, policy, baseline, off_policy=True
                    )
                finally:
                    policy.set_params(current_params)

            if defensive_parameter == 0.:
                gradient = gradient + current_gradient - previous_batch_gradient
            else:
                correction = np.mean(
                    current_gradient_samples - previous_gradient_samples,
                    axis=0,
                )
                gradient = gradient + correction

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


def def_stormpg(env, policy, *,
                horizon=100,
                discount=1.,
                step_size=1e-4,
                batch_size=100,
                mini_batch_size=10,
                momentum_parameter=0.9,
                max_iterations=1000,
                max_trajectories=None,
                defensive_parameter=0.5,
                estimator='gpomdp',
                baseline='average',
                seed=None,
                logger=None,
                n_jobs=1,
                verbose=True):
    """Run defensive STORM-PG until an iteration or trajectory limit is met."""
    if max_iterations is None and max_trajectories is None:
        raise ValueError("max_iterations and max_trajectories cannot both be None")
    if not 0. < momentum_parameter < 1.:
        raise ValueError("momentum parameter should be strictly between zero and one")
    if not 0. <= defensive_parameter < 1.:
        raise ValueError(
            "defensive parameter should be greater than or equal to zero and less than one"
        )

    rng, evaluation_rng, logger = initialize_run(seed, logger)

    if verbose:
        print("\n*** DEF-STORM-PG ***\n")

    # Initialize logger
    logger.initialize(env, policy, horizon, discount, evaluation_rng)

    if ((max_iterations is not None and max_iterations < 1)
            or (max_trajectories is not None and max_trajectories < 1)):
        logger.close()
        return

    if estimator not in ["reinforce", "gpomdp", "nonstationary"]:
        warnings.warn("Unknown gradient estimator: will default to gpomdp", UserWarning)
    if estimator == "reinforce":
        gradient_estimator = reinforce_estimator
    elif estimator == "nonstationary":
        gradient_estimator = nonstationary_pg_estimator
    else:
        gradient_estimator = gpomdp_estimator

    estimator_discount = discount if horizon is not None else 1.

    # Estimate the initial gradient using an on-policy large batch.
    initial_batch_size = capped_batch_size(batch_size, 0, max_trajectories)
    batch = generate_batch(env, policy, initial_batch_size, horizon,
                           rng=rng,
                           discount=discount,
                           parallel=(n_jobs > 1),
                           n_jobs=n_jobs)
    total_trajectories = len(batch)
    logger.submit(batch, policy)
    gradient = gradient_estimator(
        batch, estimator_discount, policy, baseline
    )

    # Learning loop
    it = 1
    while max_iterations is None or it <= max_iterations:
        if verbose:
            iteration = "{} of {}".format(it, max_iterations) if max_iterations is not None else str(it)
            print("\nIteration {} running...".format(iteration))

        if callable(step_size):
            delta = step_size(gradient)
        else:
            delta = step_size * gradient

        previous_params = policy.parameters.copy()
        policy.set_params(previous_params + delta)

        if verbose:
            print("Iteration {} completed!".format(iteration))
            print("Gradient norm = {}".format(np.linalg.norm(gradient)))
            print("Parameter delta norm = {}".format(np.linalg.norm(delta)))

        it += 1
        if ((max_iterations is not None and it > max_iterations)
                or (max_trajectories is not None and total_trajectories >= max_trajectories)):
            break

        # Sample from the trajectory-level mixture of the updated and
        # preceding policies used by the momentum correction.
        actual_mini_batch_size = capped_batch_size(
            mini_batch_size, total_trajectories, max_trajectories
        )
        batch = _generate_defensive_batch(
            env, policy, previous_params, defensive_parameter,
            actual_mini_batch_size, horizon, discount, rng, n_jobs
        )
        total_trajectories += len(batch)
        logger.submit(batch, policy)

        current_params = policy.parameters.copy()
        if defensive_parameter == 0.:
            current_gradient = gradient_estimator(
                batch, estimator_discount, policy, baseline
            )
        else:
            (current_gradient_samples,
             previous_gradient_samples) = _defensive_gradient_samples(
                batch, policy, previous_params, defensive_parameter,
                gradient_estimator, estimator_discount, baseline
            )
        if defensive_parameter == 0.:
            try:
                policy.set_params(previous_params)
                previous_batch_gradient = gradient_estimator(
                    batch, estimator_discount, policy, baseline, off_policy=True
                )
            finally:
                policy.set_params(current_params)

        decay = 1. - momentum_parameter
        if defensive_parameter == 0.:
            gradient = current_gradient + decay * (
                gradient - previous_batch_gradient
            )
        else:
            gradient = decay * gradient + np.mean(
                current_gradient_samples - decay * previous_gradient_samples,
                axis=0,
            )

    # Cleanup
    logger.close()


def def_pagepg(env, policy, *,
               horizon=100,
               discount=1.,
               step_size=1e-4,
               batch_size=100,
               mini_batch_size=10,
               refresh_probability=0.8,
               max_iterations=1000,
               max_trajectories=None,
               defensive_parameter=0.5,
               estimator='gpomdp',
               baseline='average',
               seed=None,
               logger=None,
               n_jobs=1,
               verbose=True):
    """Run defensive PAGE-PG until an iteration or trajectory limit is met."""
    if max_iterations is None and max_trajectories is None:
        raise ValueError("max_iterations and max_trajectories cannot both be None")
    if not 0. < refresh_probability <= 1.:
        raise ValueError("refresh probability should be greater than zero and at most one")
    if not 0. <= defensive_parameter < 1.:
        raise ValueError(
            "defensive parameter should be greater than or equal to zero and less than one"
        )

    rng, evaluation_rng, logger = initialize_run(seed, logger)

    if verbose:
        print("\n*** DEF-PAGE-PG ***\n")

    # Initialize logger
    logger.initialize(env, policy, horizon, discount, evaluation_rng)

    if ((max_iterations is not None and max_iterations < 1)
            or (max_trajectories is not None and max_trajectories < 1)):
        logger.close()
        return

    if estimator not in ["reinforce", "gpomdp", "nonstationary"]:
        warnings.warn("Unknown gradient estimator: will default to gpomdp", UserWarning)
    if estimator == "reinforce":
        gradient_estimator = reinforce_estimator
    elif estimator == "nonstationary":
        gradient_estimator = nonstationary_pg_estimator
    else:
        gradient_estimator = gpomdp_estimator

    estimator_discount = discount if horizon is not None else 1.

    # Estimate the initial gradient using an on-policy large batch.
    initial_batch_size = capped_batch_size(batch_size, 0, max_trajectories)
    batch = generate_batch(env, policy, initial_batch_size, horizon,
                           rng=rng,
                           discount=discount,
                           parallel=(n_jobs > 1),
                           n_jobs=n_jobs)
    total_trajectories = len(batch)
    logger.submit(batch, policy)
    gradient = gradient_estimator(
        batch, estimator_discount, policy, baseline
    )
    reset_step_size = True

    # Learning loop
    it = 1
    while max_iterations is None or it <= max_iterations:
        if verbose:
            iteration = "{} of {}".format(it, max_iterations) if max_iterations is not None else str(it)
            print("\nIteration {} running...".format(iteration))

        if callable(step_size):
            delta = step_size(gradient, reset=reset_step_size)
        else:
            delta = step_size * gradient

        previous_params = policy.parameters.copy()
        policy.set_params(previous_params + delta)

        if verbose:
            print("Iteration {} completed!".format(iteration))
            print("Gradient norm = {}".format(np.linalg.norm(gradient)))
            print("Parameter delta norm = {}".format(np.linalg.norm(delta)))

        it += 1
        if ((max_iterations is not None and it > max_iterations)
                or (max_trajectories is not None and total_trajectories >= max_trajectories)):
            break

        if (refresh_probability == 1.
                or rng.random() < refresh_probability):
            # Large-batch refresh at the updated policy.
            next_batch_size = capped_batch_size(
                batch_size, total_trajectories, max_trajectories
            )
            batch = generate_batch(env, policy, next_batch_size, horizon,
                                   rng=rng,
                                   discount=discount,
                                   parallel=(n_jobs > 1),
                                   n_jobs=n_jobs)
            total_trajectories += len(batch)
            logger.submit(batch, policy)
            gradient = gradient_estimator(
                batch, estimator_discount, policy, baseline
            )
            reset_step_size = True
        else:
            # Defensive recursive correction between the updated and
            # preceding policies.
            next_batch_size = capped_batch_size(
                mini_batch_size, total_trajectories, max_trajectories
            )
            batch = _generate_defensive_batch(
                env, policy, previous_params, defensive_parameter,
                next_batch_size, horizon, discount, rng, n_jobs
            )
            total_trajectories += len(batch)
            logger.submit(batch, policy)

            current_params = policy.parameters.copy()
            if defensive_parameter == 0.:
                current_gradient = gradient_estimator(
                    batch, estimator_discount, policy, baseline
                )
            else:
                (current_gradient_samples,
                 previous_gradient_samples) = _defensive_gradient_samples(
                    batch, policy, previous_params, defensive_parameter,
                    gradient_estimator, estimator_discount, baseline
                )
            if defensive_parameter == 0.:
                try:
                    policy.set_params(previous_params)
                    previous_batch_gradient = gradient_estimator(
                        batch, estimator_discount, policy, baseline, off_policy=True
                    )
                finally:
                    policy.set_params(current_params)

            if defensive_parameter == 0.:
                gradient = gradient + current_gradient - previous_batch_gradient
            else:
                correction = np.mean(
                    current_gradient_samples - previous_gradient_samples,
                    axis=0,
                )
                gradient = gradient + correction
            reset_step_size = False

    # Cleanup
    logger.close()


# Short public name for defensive PAGE-PG.
def_pg = def_pagepg
