import numpy as np
from potion.simulation.trajectory_generators import unpack, apply_mask, apply_discount
import warnings


def _leave_one_out_average(values, alive):
    """Average ``values`` over the other active trajectories at each time."""
    values = apply_mask(values, alive)
    totals = np.sum(values, axis=0, keepdims=True)
    counts = np.sum(alive, axis=0, keepdims=True)
    denominators = counts - alive
    return np.divide(
        totals - values,
        denominators,
        out=np.zeros_like(values, dtype=float),
        where=denominators > 0,
    )


def _leave_one_out_weighted_average(values, weights, alive):
    """Peters-style baseline formed without each sample's own trajectory."""
    masked_weights = apply_mask(weights, alive)
    weighted_values = masked_weights * values[..., None]
    weight_totals = np.sum(masked_weights, axis=0, keepdims=True)
    value_totals = np.sum(weighted_values, axis=0, keepdims=True)
    denominators = weight_totals - masked_weights
    return np.divide(
        value_totals - weighted_values,
        denominators,
        out=np.zeros_like(weighted_values, dtype=float),
        where=~np.isclose(denominators, 0.),
    )


def _leave_one_out_importance_average(values, weights, alive):
    """Importance-weighted average over the other active trajectories."""
    expanded_weights = np.broadcast_to(
        weights[:, None, None], values.shape + (1,)
    )
    return _leave_one_out_weighted_average(
        values, expanded_weights, alive
    )[..., 0]


def _mastrangelo_weights(score_weights, importance_weights):
    """Add squared trajectory ratios to Peters' squared-score weights."""
    if importance_weights is None:
        return score_weights
    weight_shape = (len(importance_weights),) + (1,) * (score_weights.ndim - 1)
    return score_weights * importance_weights.reshape(weight_shape) ** 2


def _importance_weights(states, actions, alive, behavior_logps, policy):
    if behavior_logps.shape != alive.shape:
        raise ValueError("Bad shape: behavior log probabilities should match alive flags")

    target_logps = np.zeros_like(behavior_logps, dtype=float)
    for t in range(states.shape[1]):
        active = alive[:, t]
        if np.any(active):
            target_logps[active, t] = policy.log_prob(states[active, t], actions[active, t], t)

    log_ratios = apply_mask(target_logps - behavior_logps, alive)
    return np.exp(np.sum(log_ratios, axis=1))


def _resolve_importance_weights(states, actions, alive, behavior_logps, policy,
                                off_policy, importance_weights):
    if off_policy and importance_weights is not None:
        raise ValueError(
            "off_policy and precomputed importance weights are mutually exclusive"
        )
    if importance_weights is None:
        if off_policy:
            return _importance_weights(
                states, actions, alive, behavior_logps, policy
            )
        return None

    importance_weights = np.asarray(importance_weights, dtype=float)
    if importance_weights.shape != (len(states),):
        raise ValueError(
            "Bad shape: importance weights should have one value per trajectory"
        )
    if not np.all(np.isfinite(importance_weights)):
        raise ValueError("Importance weights should be finite")
    if np.any(importance_weights < 0.):
        raise ValueError("Importance weights should be nonnegative")
    return importance_weights


def reinforce_estimator(batch, discount, policy, baseline="average", average=True,
                        off_policy=False, importance_weights=None):
    if baseline not in ["average", "weighted-average", "peters", "mastrangelo", "zero", None]:
        warnings.warn("Unknown baseline type, will default to zero baseline", UserWarning)

    states, actions, rewards, alive, logps = unpack(batch)  # NxHxS, NxHxA, NxH, NxH, NxH

    if not states.shape[-1] == policy.state_dim:
        raise ValueError("Bad shape: state dimension does not match that of given policy")
    if not actions.shape[-1] == policy.action_dim:
        raise ValueError("Bad shape: action dimension does not match that of given policy")

    importance_weights = _resolve_importance_weights(
        states, actions, alive, logps, policy, off_policy, importance_weights
    )

    scores = policy.score(states, actions)  # NxHxd
    scores = apply_mask(scores, alive)
    cum_scores = np.sum(scores, 1)  # Nxm
    rewards = apply_mask(rewards, alive)
    disc_rewards = apply_discount(rewards, discount)  # NxH
    returns = np.sum(disc_rewards, 1)  # N

    trajectory_alive = np.ones((len(returns), 1), dtype=bool)
    if baseline == 'average':
        baseline = _leave_one_out_average(
            returns[..., None], trajectory_alive
        )
    elif baseline == 'weighted-average':
        if importance_weights is not None:
            baseline = _leave_one_out_importance_average(
                returns[..., None], importance_weights, trajectory_alive
            )
        else:
            baseline = _leave_one_out_average(
                returns[..., None], trajectory_alive
            )
    elif baseline in ['peters', 'mastrangelo']:
        baseline_weights = cum_scores[:, None, :] ** 2
        if baseline == 'mastrangelo':
            baseline_weights = _mastrangelo_weights(
                baseline_weights, importance_weights
            )
        baseline = _leave_one_out_weighted_average(
            returns[..., None], baseline_weights, trajectory_alive
        )[:, 0, :]
    else:
        baseline = np.zeros((1, 1))  # 1x1
    baseline[baseline != baseline] = 0.  # replaces nan with zero
    values = returns[..., None] - baseline  # Nxd or Nx1

    grad_samples = cum_scores * values  # Nxd
    if importance_weights is not None:
        grad_samples = importance_weights[..., None] * grad_samples
    if average:
        return np.mean(grad_samples, axis=0)  # d
    return grad_samples  # Nxd


def gpomdp_estimator(batch, discount, policy, baseline='average', average=True,
                     off_policy=False, importance_weights=None):
    """Estimate GPOMDP as action scores multiplied by discounted returns-to-go.

    Sample-derived baselines exclude the trajectory to which they are applied.
    The average baseline is the leave-one-out return-to-go mean, while Peters'
    baseline is its squared-score-weighted leave-one-out counterpart. The
    Mastrangelo baseline additionally uses squared full-trajectory importance
    ratios off-policy and is identical to Peters on-policy.
    """
    if baseline not in ["average", "weighted-average", "peters", "mastrangelo", "zero", None]:
        warnings.warn("Unknown baseline type, will default to zero baseline", UserWarning)

    states, actions, rewards, alive, logps = unpack(batch)  # NxHxS, NxHxA, NxH, NxH, NxH

    if not states.shape[-1] == policy.state_dim:
        raise ValueError("Bad shape: state dimension does not match that of given policy")
    if not actions.shape[-1] == policy.action_dim:
        raise ValueError("Bad shape: action dimension does not match that of given policy")

    importance_weights = _resolve_importance_weights(
        states, actions, alive, logps, policy, off_policy, importance_weights
    )

    scores = apply_mask(policy.score(states, actions), alive)  # NxHxd
    rewards = apply_mask(rewards, alive)
    disc_rewards = apply_discount(rewards, discount)  # NxH
    returns_to_go = np.cumsum(disc_rewards[:, ::-1], axis=1)[:, ::-1]

    if baseline == 'average':
        baseline = _leave_one_out_average(returns_to_go, alive)[..., None]
    elif baseline == 'weighted-average':
        if importance_weights is not None:
            baseline = _leave_one_out_importance_average(
                returns_to_go, importance_weights, alive
            )[..., None]
        else:
            baseline = _leave_one_out_average(returns_to_go, alive)[..., None]
    elif baseline in ['peters', 'mastrangelo']:
        baseline_weights = scores ** 2
        if baseline == 'mastrangelo':
            baseline_weights = _mastrangelo_weights(
                baseline_weights, importance_weights
            )
        baseline = _leave_one_out_weighted_average(
            returns_to_go, baseline_weights, alive
        )
    else:
        baseline = np.zeros((1, 1, 1))  # 1x1x1
    values = returns_to_go[..., None] - baseline  # NxHxd or NxHx1

    grad_samples = np.sum(scores * values, axis=1)  # Nxd
    if importance_weights is not None:
        grad_samples = importance_weights[..., None] * grad_samples
    if average:
        return np.mean(grad_samples, axis=0)  # d
    return grad_samples  # Nxd


def nonstationary_pg_estimator(batch, discount, policy, baseline="average", average=True,
                               off_policy=False, importance_weights=None):
    if baseline not in ["average", "weighted-average", "peters", "mastrangelo", "zero", None]:
        warnings.warn("Unknown baseline type, will default to zero baseline", UserWarning)

    states, actions, rewards, alive, logps = unpack(batch)  # NxHxS, NxHxA, NxH, NxH, NxH

    if not states.shape[-1] == policy.state_dim:
        raise ValueError("Bad shape: state dimension does not match that of given policy")
    if not actions.shape[-1] == policy.action_dim:
        raise ValueError("Bad shape: action dimension does not match that of given policy")

    importance_weights = _resolve_importance_weights(
        states, actions, alive, logps, policy, off_policy, importance_weights
    )

    scores = policy.score(states, actions)  # NxHxd
    scores = apply_mask(scores, alive)  # NxHxd
    rewards = apply_mask(rewards, alive)
    disc_rewards = apply_discount(rewards, discount)  # NxH
    returns_to_go = np.cumsum(disc_rewards[:, ::-1], 1)[:, ::-1]  # NxH

    if baseline == 'average':
        baseline = _leave_one_out_average(returns_to_go, alive)[..., None]
    elif baseline == 'weighted-average':
        if importance_weights is not None:
            baseline = _leave_one_out_importance_average(
                returns_to_go, importance_weights, alive
            )[..., None]
        else:
            baseline = _leave_one_out_average(returns_to_go, alive)[..., None]
    elif baseline in ['peters', 'mastrangelo']:
        baseline_weights = scores ** 2
        if baseline == 'mastrangelo':
            baseline_weights = _mastrangelo_weights(
                baseline_weights, importance_weights
            )
        baseline = _leave_one_out_weighted_average(
            returns_to_go, baseline_weights, alive
        )
    else:
        baseline = np.zeros((1, 1, 1))  # 1x1x1
    baseline[baseline != baseline] = 0.  # replaces nan with zero
    values = returns_to_go[..., None] - baseline  # NxHxd or NxHx1

    grad_samples = scores * values
    grad_samples = np.reshape(grad_samples, (grad_samples.shape[0], -1))
    if importance_weights is not None:
        grad_samples = importance_weights[..., None] * grad_samples

    if average:
        return np.mean(grad_samples, axis=0)  # Hd
    return grad_samples  # NxHd
