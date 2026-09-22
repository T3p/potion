import numpy as np
from potion.simulation.trajectory_generators import unpack, apply_mask, apply_discount
import warnings
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PreparedGradientBatch:
    """Stacked trajectory data and reward terms shared by gradient estimates."""

    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    alive: np.ndarray
    logps: np.ndarray
    discount: float
    discounted_rewards: np.ndarray
    returns: np.ndarray
    returns_to_go: np.ndarray
    average_return_baseline: Optional[np.ndarray]
    average_returns_to_go_baseline: Optional[np.ndarray]


def prepare_gradient_batch(batch, discount, baseline=None):
    """Stack a trajectory batch once and precompute policy-independent terms."""
    states, actions, rewards, alive, logps = unpack(batch)
    rewards = apply_mask(rewards, alive)
    discounted_rewards = apply_discount(rewards, discount)
    returns = np.sum(discounted_rewards, axis=1)
    returns_to_go = np.cumsum(discounted_rewards[:, ::-1], axis=1)[:, ::-1]

    average_return_baseline = None
    average_returns_to_go_baseline = None
    if baseline == "average":
        trajectory_alive = np.ones((len(returns), 1), dtype=bool)
        average_return_baseline = _leave_one_out_average(
            returns[..., None], trajectory_alive
        )
        average_returns_to_go_baseline = _leave_one_out_average(
            returns_to_go, alive
        )[..., None]

    return PreparedGradientBatch(
        states=states,
        actions=actions,
        rewards=rewards,
        alive=alive,
        logps=logps,
        discount=discount,
        discounted_rewards=discounted_rewards,
        returns=returns,
        returns_to_go=returns_to_go,
        average_return_baseline=average_return_baseline,
        average_returns_to_go_baseline=average_returns_to_go_baseline,
    )


def _resolve_batch_context(batch, discount, baseline, batch_context):
    if batch_context is None:
        return prepare_gradient_batch(batch, discount, baseline)
    if not isinstance(batch_context, PreparedGradientBatch):
        raise TypeError("batch_context should be a PreparedGradientBatch")
    if batch_context.discount != discount:
        raise ValueError("batch_context discount does not match estimator discount")
    return batch_context


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
    if weights.ndim == 1:
        weight_shape = (len(weights),) + (1,) * values.ndim
        weights = weights.reshape(weight_shape)
    else:
        weights = weights[..., None]
    expanded_weights = np.broadcast_to(weights, values.shape + (1,))
    return _leave_one_out_weighted_average(
        values, expanded_weights, alive
    )[..., 0]


def _mastrangelo_weights(score_weights, importance_weights):
    """Add squared importance ratios to Peters' squared-score weights."""
    if importance_weights is None:
        return score_weights
    if importance_weights.ndim == 1:
        weight_shape = ((len(importance_weights),)
                        + (1,) * (score_weights.ndim - 1))
        importance_weights = importance_weights.reshape(weight_shape)
    else:
        importance_weights = importance_weights[..., None]
    return score_weights * importance_weights ** 2


def _importance_weights(states, actions, alive, behavior_logps, policy,
                        per_decision=False):
    if behavior_logps.shape != alive.shape:
        raise ValueError("Bad shape: behavior log probabilities should match alive flags")

    if getattr(policy, "is_stationary", False):
        target_logps = apply_mask(
            np.asarray(policy.log_prob(states, actions), dtype=np.float32), alive
        )
    else:
        target_logps = np.zeros_like(behavior_logps, dtype=np.float32)
        for t in range(states.shape[1]):
            active = alive[:, t]
            if np.any(active):
                target_logps[active, t] = policy.log_prob(
                    states[active, t], actions[active, t], t
                )

    return _importance_weights_from_logps(
        target_logps, behavior_logps, alive, per_decision
    )


def _importance_weights_from_logps(target_logps, behavior_logps, alive,
                                   per_decision=False):
    """Form stable importance weights from already evaluated log probabilities."""
    if target_logps.shape != alive.shape:
        raise ValueError("Bad shape: target log probabilities should match alive flags")
    # Rollouts keep ordinary log probabilities in float32, but their sums can
    # span long horizons. Promote before subtracting and accumulating them.
    log_ratios = apply_mask(
        np.asarray(target_logps, dtype=np.float64)
        - np.asarray(behavior_logps, dtype=np.float64),
        alive,
    )
    log_weights = (
        np.cumsum(log_ratios, axis=1)
        if per_decision
        else np.sum(log_ratios, axis=1)
    )
    with np.errstate(over="ignore", invalid="ignore"):
        importance_weights = np.exp(log_weights)
    if not np.all(np.isfinite(importance_weights)):
        raise FloatingPointError("Importance weights are not finite")
    return importance_weights


def _gpomdp_scalar_baseline(baseline, context, returns_to_go, alive,
                            importance_weights):
    if baseline == 'average':
        value = context.average_returns_to_go_baseline
        if value is None:
            value = _leave_one_out_average(returns_to_go, alive)[..., None]
        return value
    if baseline == 'weighted-average':
        if importance_weights is not None:
            return _leave_one_out_importance_average(
                returns_to_go, importance_weights, alive
            )[..., None]
        return _leave_one_out_average(returns_to_go, alive)[..., None]
    return np.zeros((1, 1, 1))


def _gpomdp_vjp_coefficients(returns_to_go, discounted_rewards, baseline,
                              importance_weights, alive):
    scalar_baseline = baseline[..., 0]
    if importance_weights is None:
        coefficients = returns_to_go - scalar_baseline
    else:
        weighted_returns_to_go = np.cumsum(
            (importance_weights * discounted_rewards)[:, ::-1],
            axis=1,
        )[:, ::-1]
        coefficients = (
            weighted_returns_to_go
            - importance_weights * scalar_baseline
        )
    return apply_mask(coefficients, alive).astype(np.float32, copy=False)


def _resolve_importance_weights(states, actions, alive, behavior_logps, policy,
                                off_policy, importance_weights,
                                per_decision=False):
    if off_policy and importance_weights is not None:
        raise ValueError(
            "off_policy and precomputed importance weights are mutually exclusive"
        )
    if importance_weights is None:
        if off_policy:
            return _importance_weights(
                states, actions, alive, behavior_logps, policy,
                per_decision=per_decision,
            )
        return None

    importance_weights = np.asarray(importance_weights, dtype=float)
    if per_decision:
        if importance_weights.shape == (len(states),):
            importance_weights = np.broadcast_to(
                importance_weights[:, None], alive.shape
            )
        elif importance_weights.shape != alive.shape:
            raise ValueError(
                "Bad shape: per-decision importance weights should match alive "
                "flags or have one value per trajectory"
            )
    elif importance_weights.shape != (len(states),):
        raise ValueError(
            "Bad shape: importance weights should have one value per trajectory"
        )
    if not np.all(np.isfinite(importance_weights)):
        raise ValueError("Importance weights should be finite")
    if np.any(importance_weights < 0.):
        raise ValueError("Importance weights should be nonnegative")
    return importance_weights


def reinforce_estimator(batch, discount, policy, baseline="average", average=True,
                        off_policy=False, importance_weights=None,
                        batch_context=None):
    if baseline not in ["average", "weighted-average", "peters", "mastrangelo", "zero", None]:
        warnings.warn("Unknown baseline type, will default to zero baseline", UserWarning)

    context = _resolve_batch_context(
        batch, discount, baseline, batch_context
    )
    states = context.states
    actions = context.actions
    alive = context.alive
    logps = context.logps

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
    returns = context.returns

    trajectory_alive = np.ones((len(returns), 1), dtype=bool)
    if baseline == 'average':
        baseline = context.average_return_baseline
        if baseline is None:
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
                     off_policy=False, importance_weights=None,
                     batch_context=None):
    """Estimate GPOMDP as action scores multiplied by discounted returns-to-go.

    Sample-derived baselines exclude the trajectory to which they are applied.
    The average baseline is the leave-one-out return-to-go mean, while Peters'
    baseline is its squared-score-weighted leave-one-out counterpart. The
    Mastrangelo baseline additionally uses squared per-decision importance
    ratios off-policy and is identical to Peters on-policy.
    """
    baseline_type = baseline
    if baseline not in ["average", "weighted-average", "peters", "mastrangelo", "zero", None]:
        warnings.warn("Unknown baseline type, will default to zero baseline", UserWarning)

    context = _resolve_batch_context(
        batch, discount, baseline, batch_context
    )
    states = context.states
    actions = context.actions
    alive = context.alive
    logps = context.logps

    if not states.shape[-1] == policy.state_dim:
        raise ValueError("Bad shape: state dimension does not match that of given policy")
    if not actions.shape[-1] == policy.action_dim:
        raise ValueError("Bad shape: action dimension does not match that of given policy")

    disc_rewards = context.discounted_rewards
    returns_to_go = context.returns_to_go

    fused_weighted_score_sum = getattr(
        policy, "fused_weighted_score_sum", None
    )
    if (
        average
        and off_policy
        and importance_weights is None
        and baseline_type not in ['peters', 'mastrangelo']
        and callable(fused_weighted_score_sum)
    ):
        def coefficient_builder(target_logps):
            fused_importance_weights = _importance_weights_from_logps(
                target_logps, logps, alive, per_decision=True
            )
            fused_baseline = _gpomdp_scalar_baseline(
                baseline, context, returns_to_go, alive,
                fused_importance_weights,
            )
            return _gpomdp_vjp_coefficients(
                returns_to_go,
                disc_rewards,
                fused_baseline,
                fused_importance_weights,
                alive,
            )

        return fused_weighted_score_sum(
            states, actions, coefficient_builder
        ) / len(states)

    importance_weights = _resolve_importance_weights(
        states, actions, alive, logps, policy, off_policy, importance_weights,
        per_decision=True,
    )

    if baseline in ['peters', 'mastrangelo']:
        scores = apply_mask(policy.score(states, actions), alive)  # NxHxd
        baseline_weights = scores ** 2
        if baseline == 'mastrangelo':
            baseline_weights = _mastrangelo_weights(
                baseline_weights, importance_weights
            )
        baseline = _leave_one_out_weighted_average(
            returns_to_go, baseline_weights, alive
        )
    else:
        baseline = _gpomdp_scalar_baseline(
            baseline, context, returns_to_go, alive, importance_weights
        )

    weighted_score_samples = getattr(policy, "weighted_score_samples", None)
    if (
        not average
        and baseline_type not in ['peters', 'mastrangelo']
        and baseline.shape[-1] == 1
        and callable(weighted_score_samples)
    ):
        coefficients = _gpomdp_vjp_coefficients(
            returns_to_go, disc_rewards, baseline, importance_weights, alive
        )
        return weighted_score_samples(states, actions, coefficients)

    weighted_score_sum = getattr(policy, "weighted_score_sum", None)
    if (
        average
        and baseline_type not in ['peters', 'mastrangelo']
        and baseline.shape[-1] == 1
        and callable(weighted_score_sum)
    ):
        coefficients = _gpomdp_vjp_coefficients(
            returns_to_go, disc_rewards, baseline, importance_weights, alive
        )
        return weighted_score_sum(states, actions, coefficients) / len(states)

    if baseline_type not in ['peters', 'mastrangelo']:
        scores = apply_mask(policy.score(states, actions), alive)  # NxHxd
    values = returns_to_go[..., None] - baseline  # NxHxd or NxHx1
    if scores.dtype == np.float32:
        values = values.astype(np.float32, copy=False)

    if importance_weights is None:
        grad_samples = np.sum(scores * values, axis=1)  # Nxd
    else:
        # For GPOMDP, every reward at time h is weighted by the policy ratio
        # accumulated only through h.  A single full-trajectory ratio is
        # unbiased but has unnecessarily high variance and is not the
        # per-decision estimator used by variance-reduced PG algorithms.
        if scores.dtype == np.float32:
            importance_weights = importance_weights.astype(
                np.float32, copy=False
            )
            disc_rewards = disc_rewards.astype(np.float32, copy=False)
            baseline = baseline.astype(np.float32, copy=False)
        cumulative_scores = np.cumsum(scores, axis=1)
        weighted_rewards = (
            importance_weights[..., None]
            * disc_rewards[..., None]
            * cumulative_scores
        )
        weighted_baseline = (
            importance_weights[..., None] * scores * baseline
        )
        grad_samples = np.sum(
            weighted_rewards - weighted_baseline,
            axis=1,
        )
    if average:
        return np.mean(grad_samples, axis=0)  # d
    return grad_samples  # Nxd


def nonstationary_pg_estimator(batch, discount, policy, baseline="average", average=True,
                                off_policy=False, importance_weights=None,
                                batch_context=None):
    if baseline not in ["average", "weighted-average", "peters", "mastrangelo", "zero", None]:
        warnings.warn("Unknown baseline type, will default to zero baseline", UserWarning)

    context = _resolve_batch_context(
        batch, discount, baseline, batch_context
    )
    states = context.states
    actions = context.actions
    alive = context.alive
    logps = context.logps

    if not states.shape[-1] == policy.state_dim:
        raise ValueError("Bad shape: state dimension does not match that of given policy")
    if not actions.shape[-1] == policy.action_dim:
        raise ValueError("Bad shape: action dimension does not match that of given policy")

    importance_weights = _resolve_importance_weights(
        states, actions, alive, logps, policy, off_policy, importance_weights,
        per_decision=True,
    )

    scores = policy.score(states, actions)  # NxHxd
    scores = apply_mask(scores, alive)  # NxHxd
    disc_rewards = context.discounted_rewards
    returns_to_go = context.returns_to_go

    if baseline == 'average':
        baseline = context.average_returns_to_go_baseline
        if baseline is None:
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

    if importance_weights is None:
        grad_samples = scores * values
    else:
        weighted_returns_to_go = np.cumsum(
            (importance_weights * disc_rewards)[:, ::-1],
            axis=1,
        )[:, ::-1]
        grad_samples = scores * (
            weighted_returns_to_go[..., None]
            - importance_weights[..., None] * baseline
        )
    grad_samples = np.reshape(grad_samples, (grad_samples.shape[0], -1))

    if average:
        return np.mean(grad_samples, axis=0)  # Hd
    return grad_samples  # NxHd
