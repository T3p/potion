from potion.simulation.trajectory_generators import generate_batch, unpack, apply_mask
from potion.estimators.gradients import gpomdp_estimator, reinforce_estimator, nonstationary_pg_estimator
from potion.evaluation.loggers import EpisodicOnlineLogger
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

    # generate_batch derives episode seeds from its RNG's seed sequence. Use
    # independent child seed sequences so the two components do not reuse seeds.
    current_seed_seq, snapshot_seed_seq = rng.bit_generator.seed_seq.spawn(2)
    current_rng = np.random.default_rng(current_seed_seq)
    snapshot_rng = np.random.default_rng(snapshot_seed_seq)

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
              logger=EpisodicOnlineLogger(),
              n_jobs=1,
              verbose=True):
    """Run defensive SVRPG until an iteration or trajectory limit is met."""
    if max_iterations is None and max_trajectories is None:
        raise ValueError("max_iterations and max_trajectories cannot both be None")
    if not 0. <= defensive_parameter < 1.:
        raise ValueError(
            "defensive parameter should be greater than or equal to zero and less than one"
        )

    rng = np.random.default_rng(seed)

    if verbose:
        print("\n*** DEF-SVRPG ***\n")

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
    total_trajectories = 0
    while ((max_iterations is None or it <= max_iterations)
           and (max_trajectories is None or total_trajectories < max_trajectories)):
        if verbose:
            iteration = "{} of {}".format(it, max_iterations) if max_iterations is not None else str(it)
            print("\nIteration {} running...".format(iteration))

        snapshot_params = policy.parameters.copy()

        # Estimate the gradient at the snapshot policy using a large batch
        snapshot_batch = generate_batch(env, policy, batch_size, horizon,
                                        rng=rng,
                                        discount=discount,
                                        parallel=(n_jobs > 1),
                                        n_jobs=n_jobs)
        total_trajectories += len(snapshot_batch)
        logger.submit(snapshot_batch, policy)
        snapshot_gradient = gradient_estimator(snapshot_batch, estimator_discount, policy, baseline)

        epoch = 1
        while (epoch <= epoch_length
               and (max_trajectories is None or total_trajectories < max_trajectories)):
            if verbose:
                print("Epoch {} of {} running...".format(epoch, epoch_length))

            # Collect trajectories from the trajectory-level mixture of the
            # current and snapshot policies.
            batch = _generate_defensive_batch(
                env, policy, snapshot_params, defensive_parameter,
                mini_batch_size, horizon, discount, rng, n_jobs
            )
            total_trajectories += len(batch)
            logger.submit(batch, policy)

            current_params = policy.parameters.copy()
            current_gradient_samples = gradient_estimator(
                batch, estimator_discount, policy, baseline, average=False
            )

            if defensive_parameter > 0.:
                # Evaluate each trajectory under both policies. The component
                # that generated it is irrelevant: both weights use the
                # mixture density.
                current_logps = _trajectory_log_probabilities(batch, policy)

            try:
                policy.set_params(snapshot_params)
                if defensive_parameter == 0.:
                    snapshot_gradient_samples = gradient_estimator(
                        batch, estimator_discount, policy, baseline,
                        average=False, off_policy=True
                    )
                else:
                    snapshot_logps = _trajectory_log_probabilities(batch, policy)
                    snapshot_gradient_samples = gradient_estimator(
                        batch, estimator_discount, policy, baseline, average=False
                    )
            finally:
                policy.set_params(current_params)

            if defensive_parameter == 0.:
                correction = np.mean(
                    current_gradient_samples - snapshot_gradient_samples, axis=0
                )
            else:
                current_weights, snapshot_weights = _defensive_importance_weights(
                    current_logps, snapshot_logps, defensive_parameter
                )
                correction = np.mean(
                    current_weights[..., None] * current_gradient_samples
                    - snapshot_weights[..., None] * snapshot_gradient_samples,
                    axis=0,
                )
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
               logger=EpisodicOnlineLogger(),
               n_jobs=1,
               verbose=True):
    """Run defensive SRVR-PG until an iteration or trajectory limit is met."""
    if max_iterations is None and max_trajectories is None:
        raise ValueError("max_iterations and max_trajectories cannot both be None")
    if not 0. <= defensive_parameter < 1.:
        raise ValueError(
            "defensive parameter should be greater than or equal to zero and less than one"
        )

    rng = np.random.default_rng(seed)

    if verbose:
        print("\n*** DEF-SRVR-PG ***\n")

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
    total_trajectories = 0
    while ((max_iterations is None or it <= max_iterations)
           and (max_trajectories is None or total_trajectories < max_trajectories)):
        if verbose:
            iteration = "{} of {}".format(it, max_iterations) if max_iterations is not None else str(it)
            print("\nIteration {} running...".format(iteration))

        # Start the epoch with an on-policy large-batch estimate and update.
        batch = generate_batch(env, policy, batch_size, horizon,
                               rng=rng,
                               discount=discount,
                               parallel=(n_jobs > 1),
                               n_jobs=n_jobs)
        total_trajectories += len(batch)
        logger.submit(batch, policy)
        gradient = gradient_estimator(batch, estimator_discount, policy, baseline)

        if callable(step_size):
            delta = step_size(gradient)
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
            batch = _generate_defensive_batch(
                env, policy, previous_params, defensive_parameter,
                mini_batch_size, horizon, discount, rng, n_jobs
            )
            total_trajectories += len(batch)
            logger.submit(batch, policy)

            current_params = policy.parameters.copy()
            current_gradient_samples = gradient_estimator(
                batch, estimator_discount, policy, baseline, average=False
            )
            if defensive_parameter > 0.:
                current_logps = _trajectory_log_probabilities(batch, policy)
            try:
                policy.set_params(previous_params)
                if defensive_parameter == 0.:
                    previous_gradient_samples = gradient_estimator(
                        batch, estimator_discount, policy, baseline,
                        average=False, off_policy=True
                    )
                else:
                    previous_logps = _trajectory_log_probabilities(batch, policy)
                    previous_gradient_samples = gradient_estimator(
                        batch, estimator_discount, policy, baseline, average=False
                    )
            finally:
                policy.set_params(current_params)

            if defensive_parameter == 0.:
                gradient = gradient + np.mean(
                    current_gradient_samples - previous_gradient_samples, axis=0
                )
            else:
                current_weights, previous_weights = _defensive_importance_weights(
                    current_logps, previous_logps, defensive_parameter
                )
                gradient = gradient + np.mean(
                    current_weights[..., None] * current_gradient_samples
                    - previous_weights[..., None] * previous_gradient_samples,
                    axis=0,
                )

            if callable(step_size):
                delta = step_size(gradient)
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
                logger=EpisodicOnlineLogger(),
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

    rng = np.random.default_rng(seed)

    if verbose:
        print("\n*** DEF-STORM-PG ***\n")

    # Initialize logger
    logger.initialize(env, policy, horizon, discount, rng)

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
    batch = generate_batch(env, policy, batch_size, horizon,
                           rng=rng,
                           discount=discount,
                           parallel=(n_jobs > 1),
                           n_jobs=n_jobs)
    total_trajectories = len(batch)
    logger.submit(batch, policy)
    gradient = gradient_estimator(batch, estimator_discount, policy, baseline)

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
        batch = _generate_defensive_batch(
            env, policy, previous_params, defensive_parameter,
            mini_batch_size, horizon, discount, rng, n_jobs
        )
        total_trajectories += len(batch)
        logger.submit(batch, policy)

        current_params = policy.parameters.copy()
        current_gradient_samples = gradient_estimator(
            batch, estimator_discount, policy, baseline, average=False
        )
        if defensive_parameter > 0.:
            current_logps = _trajectory_log_probabilities(batch, policy)
        try:
            policy.set_params(previous_params)
            if defensive_parameter == 0.:
                previous_gradient_samples = gradient_estimator(
                    batch, estimator_discount, policy, baseline,
                    average=False, off_policy=True
                )
            else:
                previous_logps = _trajectory_log_probabilities(batch, policy)
                previous_gradient_samples = gradient_estimator(
                    batch, estimator_discount, policy, baseline, average=False
                )
        finally:
            policy.set_params(current_params)

        decay = 1. - momentum_parameter
        if defensive_parameter == 0.:
            gradient = np.mean(current_gradient_samples, axis=0) + decay * (
                gradient - np.mean(previous_gradient_samples, axis=0)
            )
        else:
            current_weights, previous_weights = _defensive_importance_weights(
                current_logps, previous_logps, defensive_parameter
            )
            gradient = decay * gradient + np.mean(
                current_weights[..., None] * current_gradient_samples
                - decay * previous_weights[..., None] * previous_gradient_samples,
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
               logger=EpisodicOnlineLogger(),
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

    rng = np.random.default_rng(seed)

    if verbose:
        print("\n*** DEF-PAGE-PG ***\n")

    # Initialize logger
    logger.initialize(env, policy, horizon, discount, rng)

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
    initial_batch_size = (
        batch_size if max_trajectories is None
        else min(batch_size, max_trajectories)
    )
    batch = generate_batch(env, policy, initial_batch_size, horizon,
                           rng=rng,
                           discount=discount,
                           parallel=(n_jobs > 1),
                           n_jobs=n_jobs)
    total_trajectories = len(batch)
    logger.submit(batch, policy)
    gradient = gradient_estimator(batch, estimator_discount, policy, baseline)

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

        if rng.random() < refresh_probability:
            # Large-batch refresh at the updated policy.
            next_batch_size = (
                batch_size if max_trajectories is None
                else min(batch_size, max_trajectories - total_trajectories)
            )
            batch = generate_batch(env, policy, next_batch_size, horizon,
                                   rng=rng,
                                   discount=discount,
                                   parallel=(n_jobs > 1),
                                   n_jobs=n_jobs)
            total_trajectories += len(batch)
            logger.submit(batch, policy)
            gradient = gradient_estimator(batch, estimator_discount, policy, baseline)
        else:
            # Defensive recursive correction between the updated and
            # preceding policies.
            next_batch_size = (
                mini_batch_size if max_trajectories is None
                else min(mini_batch_size, max_trajectories - total_trajectories)
            )
            batch = _generate_defensive_batch(
                env, policy, previous_params, defensive_parameter,
                next_batch_size, horizon, discount, rng, n_jobs
            )
            total_trajectories += len(batch)
            logger.submit(batch, policy)

            current_params = policy.parameters.copy()
            current_gradient_samples = gradient_estimator(
                batch, estimator_discount, policy, baseline, average=False
            )
            if defensive_parameter > 0.:
                current_logps = _trajectory_log_probabilities(batch, policy)
            try:
                policy.set_params(previous_params)
                if defensive_parameter == 0.:
                    previous_gradient_samples = gradient_estimator(
                        batch, estimator_discount, policy, baseline,
                        average=False, off_policy=True
                    )
                else:
                    previous_logps = _trajectory_log_probabilities(batch, policy)
                    previous_gradient_samples = gradient_estimator(
                        batch, estimator_discount, policy, baseline, average=False
                    )
            finally:
                policy.set_params(current_params)

            if defensive_parameter == 0.:
                gradient = gradient + np.mean(
                    current_gradient_samples - previous_gradient_samples, axis=0
                )
            else:
                current_weights, previous_weights = _defensive_importance_weights(
                    current_logps, previous_logps, defensive_parameter
                )
                gradient = gradient + np.mean(
                    current_weights[..., None] * current_gradient_samples
                    - previous_weights[..., None] * previous_gradient_samples,
                    axis=0,
                )

    # Cleanup
    logger.close()


# Short public name for defensive PAGE-PG.
def_pg = def_pagepg
