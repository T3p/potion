from potion.simulation.trajectory_generators import generate_batch, unpack, apply_mask
from potion.estimators.gradients import gpomdp_estimator, reinforce_estimator, nonstationary_pg_estimator
from potion.evaluation.loggers import EpisodicPerformanceLogger
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
    if not 0. < defensive_parameter < 1.:
        raise ValueError("defensive parameter should be strictly between zero and one")
    if current_logps.shape != snapshot_logps.shape:
        raise ValueError("current and snapshot log probabilities should have the same shape")

    log_mixture = np.logaddexp(
        np.log(defensive_parameter) + current_logps,
        np.log1p(-defensive_parameter) + snapshot_logps,
    )
    current_weights = np.exp(current_logps - log_mixture)
    snapshot_weights = np.exp(snapshot_logps - log_mixture)
    return current_weights, snapshot_weights


def _generate_defensive_batch(env, policy, snapshot_params, defensive_parameter,
                              n_episodes, horizon, discount, rng, n_jobs):
    current_params = policy.parameters.copy()
    draw_current = rng.random(n_episodes) < defensive_parameter
    n_current = np.count_nonzero(draw_current)
    n_snapshot = n_episodes - n_current

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
              mini_batch_size=22,
              epoch_length=10,
              max_iterations=1000,
              defensive_parameter=0.5,
              estimator='gpomdp',
              baseline='average',
              seed=None,
              logger=EpisodicPerformanceLogger(),
              n_jobs=1,
              verbose=True):
    """Defensive-importance-sampling variant of SVRPG."""
    if not 0. < defensive_parameter < 1.:
        raise ValueError("defensive parameter should be strictly between zero and one")

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

            # Collect trajectories from the trajectory-level mixture of the
            # current and snapshot policies.
            batch = _generate_defensive_batch(
                env, policy, snapshot_params, defensive_parameter,
                mini_batch_size, horizon, discount, rng, n_jobs
            )
            logger.submit(batch, policy)

            # Evaluate each trajectory under both policies. The component that
            # generated it is irrelevant: both weights use the mixture density.
            current_params = policy.parameters.copy()
            current_logps = _trajectory_log_probabilities(batch, policy)
            current_gradient_samples = gradient_estimator(
                batch, estimator_discount, policy, baseline, average=False
            )
            try:
                policy.set_params(snapshot_params)
                snapshot_logps = _trajectory_log_probabilities(batch, policy)
                snapshot_gradient_samples = gradient_estimator(
                    batch, estimator_discount, policy, baseline, average=False
                )
            finally:
                policy.set_params(current_params)

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
