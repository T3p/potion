import numpy as np
import pytest
from potion.estimators.gradients import (
    reinforce_estimator,
    gpomdp_estimator,
    nonstationary_pg_estimator,
    prepare_gradient_batch,
    _importance_weights,
)
from potion.policies.gaussian_policies import DeepGaussianPolicy, LinearGaussianPolicy
from potion.policies.wrappers import Staged
from potion.simulation.trajectory_generators import unpack
from torch import nn


@pytest.fixture
def small_policy():
    class MockPolicy:
        state_dim = 1
        action_dim = 1
        num_params = 2

        def score(self, s, a):
            x = np.array([[[-0.3], [0.5]],
                          [[1.], [-1.5]]])
            return np.concatenate((x, 2. * x), -1)

        def log_prob(self, s, a, t=None):
            return np.zeros(s.shape[:-1])
    return MockPolicy()

@pytest.fixture
def small_batch(small_policy):
    return [(np.ones((2, small_policy.state_dim)),
              np.ones((2, small_policy.action_dim)),
              np.array([1., -1.]),
              np.array([True, True]),
              np.zeros(2)),
             (np.ones((2, small_policy.state_dim)),
              np.ones((2, small_policy.action_dim)),
              np.array([4., -2.]),
              np.array([True, True]),
              np.zeros(2))
             ]


@pytest.mark.parametrize("estimator", (reinforce_estimator, gpomdp_estimator, nonstationary_pg_estimator))
def test_gradient_estimators_shapes(batch, discount, policy, n_traj, n_params, max_trajectory_len, estimator):
    grad = estimator(batch, discount, policy, baseline="average")
    grad_samples = estimator(
        batch, discount, policy, baseline="average", average=False
    )

    expected_num_params = n_params if estimator is not nonstationary_pg_estimator else max_trajectory_len * n_params
    assert grad.shape == (expected_num_params,)
    assert grad_samples.shape == (n_traj, expected_num_params)


@pytest.mark.parametrize(
    "estimator", (reinforce_estimator, gpomdp_estimator, nonstationary_pg_estimator)
)
def test_prepared_gradient_batch_matches_regular_estimator(
        batch, discount, policy, estimator):
    context = prepare_gradient_batch(batch, discount, baseline="average")

    expected = estimator(batch, discount, policy, baseline="average", average=False)
    actual = estimator(
        batch,
        discount,
        policy,
        baseline="average",
        average=False,
        batch_context=context,
    )

    assert np.allclose(actual, expected)


def test_prepared_gradient_batch_avoids_restacking(
        small_batch, small_policy, mocker):
    context = prepare_gradient_batch(small_batch, 0.9, baseline="average")
    mocker.patch(
        "potion.estimators.gradients.unpack",
        side_effect=AssertionError("prepared data should be reused"),
    )

    result = gpomdp_estimator(
        small_batch,
        0.9,
        small_policy,
        baseline="average",
        batch_context=context,
    )

    assert result.shape == (small_policy.num_params,)


@pytest.mark.parametrize("estimator", (reinforce_estimator, gpomdp_estimator, nonstationary_pg_estimator))
@pytest.mark.parametrize("baseline", (None, "average", "peters"))
def test_gradient_estimators_invariance(batch, discount, policy, n_traj, estimator, baseline):
    grad = estimator(batch, discount, policy, baseline=baseline)
    samples = estimator(batch, discount, policy, baseline=baseline, average=False)
    batch_2 = []
    for i in range(n_traj):
        batch_2.append((batch[i][0],
                       batch[i][1],
                       2. * batch[i][2],  # double rewards
                       batch[i][3],
                       batch[i][4]))

    grad_2 = estimator(batch_2, discount, policy, baseline=baseline)

    batch_3 = batch[::-1]  # reverse

    grad_3 = estimator(batch_3, discount, policy, baseline=baseline)
    samples_3 = estimator(batch_3, discount, policy, baseline=baseline, average=False)

    assert np.allclose(grad_2, 2. * grad)
    assert np.allclose(grad_3, grad)
    assert np.allclose(samples_3, samples[::-1, :])


def test_reinforce_estimator_values(small_policy, small_batch):
    pol = small_policy

    batch = small_batch

    grad_1 = reinforce_estimator(batch, 0.9, pol, baseline=None)
    grad_2 = reinforce_estimator(batch, 0.9, pol, baseline="average")
    grad_3 = reinforce_estimator(batch, 0.9, pol, baseline="peters")

    g1 = -0.54
    g2 = -0.735
    g3 = -0.735

    assert np.allclose(grad_1, [g1, 2. * g1])
    assert np.allclose(grad_2, [g2, 2. * g2])
    assert np.allclose(grad_3, [g3, 2. * g3])


def test_gpomdp_estimator_values(small_policy, small_batch):
    pol = small_policy

    batch = small_batch

    grad_1 = gpomdp_estimator(batch, 0.9, pol, baseline=None)
    grad_2 = gpomdp_estimator(batch, 0.9, pol, baseline="average")
    grad_3 = gpomdp_estimator(batch, 0.9, pol, baseline="peters")

    g1 = 2.21
    g2 = 2.265
    g3 = 2.265

    assert np.allclose(grad_1, [g1, 2. * g1])
    assert np.allclose(grad_2, [g2, 2. * g2])
    assert np.allclose(grad_3, [g3, 2. * g3])


def test_gpomdp_uses_weighted_score_samples_when_available(small_batch):
    class FastPolicy:
        state_dim = 1
        action_dim = 1

        def score(self, states, actions):
            raise AssertionError("The score tensor should not be materialized")

        def weighted_score_samples(self, states, actions, coefficients):
            assert coefficients.shape == states.shape[:-1]
            return np.full((len(states), 2), 3., dtype=np.float32)

    samples = gpomdp_estimator(
        small_batch, 0.9, FastPolicy(), baseline="average", average=False
    )

    assert np.array_equal(samples, np.full((2, 2), 3., dtype=np.float32))


def test_gpomdp_uses_weighted_score_sum_when_available(small_batch):
    class FastPolicy:
        state_dim = 1
        action_dim = 1

        def score(self, states, actions):
            raise AssertionError("The score tensor should not be materialized")

        def weighted_score_sum(self, states, actions, coefficients):
            assert coefficients.shape == states.shape[:-1]
            return np.full(2, 6., dtype=np.float32)

    gradient = gpomdp_estimator(
        small_batch, 0.9, FastPolicy(), baseline="average", average=True
    )

    assert np.array_equal(gradient, np.full(2, 3., dtype=np.float32))


@pytest.mark.parametrize("estimator", (reinforce_estimator, gpomdp_estimator, nonstationary_pg_estimator))
def test_gradient_estimators_exceptions(batch, discount, policy, estimator):
    batch_1 = [(np.ones((2, policy.state_dim + 1)),
               np.ones((2, policy.action_dim)),
               np.array([1., -1.]),
               np.array([True, True]),
               np.zeros(2))]

    batch_2 = [(np.ones((2, policy.state_dim)),
               np.ones((2, policy.action_dim - 1)),
               np.array([1., -1.]),
               np.array([True, True]),
               np.zeros(2))]

    batch_3 = [(np.ones((2, policy.state_dim)),
               np.ones((2, policy.action_dim)),
               np.array([1., -1.]),
               np.array([True, True]),
               np.zeros(3))]

    with pytest.warns(UserWarning):
        _ = estimator(batch, discount, policy, baseline="xyz")

    with pytest.raises(ValueError):
        _ = estimator(batch_1, discount, policy)

    with pytest.raises(ValueError):
        _ = estimator(batch_2, discount, policy)

    with pytest.raises(ValueError):
        _ = estimator(batch_3, discount, policy, off_policy=True)

    with pytest.raises(ValueError):
        _ = estimator(
            batch, discount, policy,
            importance_weights=np.ones(len(batch) + 1)
        )

    with pytest.raises(ValueError):
        _ = estimator(
            batch, discount, policy,
            importance_weights=-np.ones(len(batch))
        )

    with pytest.raises(ValueError):
        _ = estimator(
            batch, discount, policy, off_policy=True,
            importance_weights=np.ones(len(batch))
        )


@pytest.mark.parametrize("estimator", (reinforce_estimator, gpomdp_estimator, nonstationary_pg_estimator))
def test_gradient_estimators_masking(batch, discount, policy, horizon, estimator):
    grad = estimator(batch, discount, policy, baseline="peters")

    batch_1 = []
    for i in range(len(batch)):
        s, a, r, al, logps = batch[i]
        s[horizon:] = -100. * np.ones(policy.state_dim)
        a[horizon:] = 200. * np.ones(policy.action_dim)
        r[horizon:] = -150.
        batch_1.append((s, a, r, al, logps))

    grad_1 = estimator(batch_1, discount, policy, baseline="peters")

    assert np.allclose(grad_1, grad)


@pytest.mark.parametrize("estimator", (reinforce_estimator, gpomdp_estimator, nonstationary_pg_estimator))
def test_gradient_estimators_off_policy_weights(small_batch, small_policy, estimator):
    weights = np.array([2., 0.5])
    weighted_batch = []
    for trajectory, weight in zip(small_batch, weights):
        states, actions, rewards, alive, _ = trajectory
        behavior_logps = np.array([-np.log(weight), 0.])
        weighted_batch.append((states, actions, rewards, alive, behavior_logps))

    on_policy_samples = estimator(
        small_batch, 0.9, small_policy, baseline=None, average=False
    )
    ignored_logps_samples = estimator(
        weighted_batch, 0.9, small_policy, baseline=None, average=False
    )
    off_policy_samples = estimator(
        weighted_batch, 0.9, small_policy, baseline=None,
        average=False, off_policy=True
    )
    off_policy_grad = estimator(
        weighted_batch, 0.9, small_policy, baseline=None, off_policy=True
    )

    assert np.allclose(ignored_logps_samples, on_policy_samples)
    assert np.allclose(off_policy_samples, weights[..., None] * on_policy_samples)
    assert np.allclose(off_policy_grad, np.mean(off_policy_samples, axis=0))


@pytest.mark.parametrize("estimator", (reinforce_estimator, gpomdp_estimator, nonstationary_pg_estimator))
@pytest.mark.parametrize("baseline", (None, "weighted-average", "mastrangelo"))
def test_precomputed_importance_weights_match_derived_weights(
        small_batch, small_policy, estimator, baseline):
    weights = np.array([2., 0.5])
    weighted_batch = []
    for trajectory, weight in zip(small_batch, weights):
        states, actions, rewards, alive, _ = trajectory
        behavior_logps = np.array([-np.log(weight), 0.])
        weighted_batch.append((states, actions, rewards, alive, behavior_logps))

    derived = estimator(
        weighted_batch, 0.9, small_policy, baseline=baseline,
        average=False, off_policy=True
    )
    precomputed = estimator(
        weighted_batch, 0.9, small_policy, baseline=baseline,
        average=False, importance_weights=weights
    )

    assert np.array_equal(precomputed, derived)


@pytest.mark.parametrize("estimator", (reinforce_estimator, gpomdp_estimator, nonstationary_pg_estimator))
def test_weighted_average_baseline_matches_average_on_policy(
        small_batch, small_policy, estimator):
    average_samples = estimator(
        small_batch, 0.9, small_policy, baseline="average", average=False
    )
    weighted_average_samples = estimator(
        small_batch, 0.9, small_policy, baseline="weighted-average", average=False
    )

    assert np.array_equal(weighted_average_samples, average_samples)


@pytest.mark.parametrize("estimator", (reinforce_estimator, gpomdp_estimator, nonstationary_pg_estimator))
def test_weighted_average_baseline_uses_off_policy_leave_one_out_weights(estimator):
    class StateScorePolicy:
        state_dim = 1
        action_dim = 1

        def score(self, states, actions):
            return states

        def log_prob(self, states, actions, t=None):
            return np.zeros(states.shape[:-1])

    scores = np.array([1., 2., 4.])
    returns = np.array([10., 20., 30.])
    weights = np.array([1., 2., 4.])
    batch = [
        (
            np.array([[score]]),
            np.ones((1, 1)),
            np.array([ret]),
            np.ones(1, dtype=bool),
            np.array([-np.log(weight)]),
        )
        for score, ret, weight in zip(scores, returns, weights)
    ]

    samples = estimator(
        batch,
        1.,
        StateScorePolicy(),
        baseline="weighted-average",
        average=False,
        off_policy=True,
    )
    leave_one_out_baselines = np.array([
        (2. * 20. + 4. * 30.) / (2. + 4.),
        (1. * 10. + 4. * 30.) / (1. + 4.),
        (1. * 10. + 2. * 20.) / (1. + 2.),
    ])
    expected = weights * scores * (returns - leave_one_out_baselines)

    assert np.allclose(samples[:, 0], expected)


@pytest.mark.parametrize("estimator", (reinforce_estimator, gpomdp_estimator, nonstationary_pg_estimator))
def test_mastrangelo_baseline_matches_peters_on_policy(
        small_batch, small_policy, estimator):
    peters_samples = estimator(
        small_batch, 0.9, small_policy, baseline="peters", average=False
    )
    mastrangelo_samples = estimator(
        small_batch, 0.9, small_policy, baseline="mastrangelo", average=False
    )

    assert np.array_equal(mastrangelo_samples, peters_samples)


@pytest.mark.parametrize("estimator", (reinforce_estimator, gpomdp_estimator, nonstationary_pg_estimator))
def test_mastrangelo_baseline_uses_squared_off_policy_weights(estimator):
    class StateScorePolicy:
        state_dim = 1
        action_dim = 1

        def score(self, states, actions):
            return states

        def log_prob(self, states, actions, t=None):
            return np.zeros(states.shape[:-1])

    scores = np.array([1., 2., 4.])
    returns = np.array([10., 20., 30.])
    importance_weights = np.array([1., 2., 4.])
    batch = [
        (
            np.array([[score]]),
            np.ones((1, 1)),
            np.array([ret]),
            np.ones(1, dtype=bool),
            np.array([-np.log(weight)]),
        )
        for score, ret, weight in zip(scores, returns, importance_weights)
    ]

    samples = estimator(
        batch,
        1.,
        StateScorePolicy(),
        baseline="mastrangelo",
        average=False,
        importance_weights=importance_weights,
    )
    baseline_weights = importance_weights ** 2 * scores ** 2
    leave_one_out_baselines = np.array([
        (baseline_weights[1] * returns[1] + baseline_weights[2] * returns[2])
        / (baseline_weights[1] + baseline_weights[2]),
        (baseline_weights[0] * returns[0] + baseline_weights[2] * returns[2])
        / (baseline_weights[0] + baseline_weights[2]),
        (baseline_weights[0] * returns[0] + baseline_weights[1] * returns[1])
        / (baseline_weights[0] + baseline_weights[1]),
    ])
    expected = importance_weights * scores * (returns - leave_one_out_baselines)

    assert np.allclose(samples[:, 0], expected)


@pytest.mark.parametrize("estimator", (gpomdp_estimator, nonstationary_pg_estimator))
def test_mastrangelo_time_baseline_uses_per_decision_weights(estimator):
    class StateScorePolicy:
        state_dim = 1
        action_dim = 1

        def score(self, states, actions):
            return states

        def log_prob(self, states, actions, t=None):
            return np.zeros(states.shape[:-1])

    scores = np.array([[1., 2.], [2., 3.], [4., 5.]])
    rewards = np.array([[1., 10.], [2., 20.], [3., 30.]])
    importance_weights = np.array([1., 2., 4.])
    behavior_logps = np.column_stack((
        np.zeros(len(importance_weights)),
        -np.log(importance_weights),
    ))
    batch = [
        (
            trajectory_scores[..., None],
            np.ones((2, 1)),
            trajectory_rewards,
            np.ones(2, dtype=bool),
            trajectory_logps,
        )
        for trajectory_scores, trajectory_rewards, trajectory_logps
        in zip(scores, rewards, behavior_logps)
    ]

    samples = estimator(
        batch,
        1.,
        StateScorePolicy(),
        baseline="mastrangelo",
        average=False,
        off_policy=True,
    )
    returns_to_go = np.cumsum(rewards[:, ::-1], axis=1)[:, ::-1]
    prefix_weights = np.column_stack((
        np.ones(len(importance_weights)),
        importance_weights,
    ))
    baseline_weights = prefix_weights ** 2 * scores ** 2
    leave_one_out_baselines = (
        np.sum(baseline_weights * returns_to_go, axis=0) -
        baseline_weights * returns_to_go
    ) / (np.sum(baseline_weights, axis=0) - baseline_weights)
    if estimator is gpomdp_estimator:
        expected_steps = (
            prefix_weights * rewards * np.cumsum(scores, axis=1)
            - prefix_weights * scores * leave_one_out_baselines
        )
        expected = np.sum(expected_steps, axis=1, keepdims=True)
    else:
        weighted_returns_to_go = np.cumsum(
            (prefix_weights * rewards)[:, ::-1], axis=1
        )[:, ::-1]
        expected = scores * (
            weighted_returns_to_go
            - prefix_weights * leave_one_out_baselines
        )

    assert np.allclose(samples, expected)


def test_reinforce_off_policy_keeps_full_trajectory_weight():
    class StateScorePolicy:
        state_dim = 1
        action_dim = 1

        def score(self, states, actions):
            return states

        def log_prob(self, states, actions, t=None):
            return np.zeros(states.shape[:-1])

    scores = np.array([1., 2.])
    rewards = np.array([1., 10.])
    incremental_ratios = np.array([2., 3.])
    batch = [(
        scores[..., None],
        np.ones((2, 1)),
        rewards,
        np.ones(2, dtype=bool),
        -np.log(incremental_ratios),
    )]

    samples = reinforce_estimator(
        batch,
        1.,
        StateScorePolicy(),
        baseline=None,
        average=False,
        off_policy=True,
    )
    expected = (
        np.prod(incremental_ratios) * np.sum(scores) * np.sum(rewards)
    )

    assert samples[0, 0] == pytest.approx(expected)


@pytest.mark.parametrize("estimator", (reinforce_estimator, gpomdp_estimator, nonstationary_pg_estimator))
def test_gradient_estimators_off_policy_masking(batch, discount, policy, estimator):
    target_logps = np.zeros((len(batch), len(batch[0][3])))
    for t in range(target_logps.shape[1]):
        states_t = np.stack([trajectory[0][t] for trajectory in batch])
        actions_t = np.stack([trajectory[1][t] for trajectory in batch])
        target_logps[:, t] = policy.log_prob(states_t, actions_t, t)

    matched_batch = []
    for trajectory, trajectory_logps in zip(batch, target_logps):
        states, actions, rewards, alive, _ = trajectory
        trajectory_logps[~alive] = 1e6
        matched_batch.append((states, actions, rewards, alive, trajectory_logps))

    on_policy_grad = estimator(batch, discount, policy, baseline=None)
    off_policy_grad = estimator(matched_batch, discount, policy, baseline=None, off_policy=True)

    assert np.allclose(off_policy_grad, on_policy_grad)


def test_nonstationary_off_policy_weights_staged_policy():
    horizon = 2
    policy = Staged(LinearGaussianPolicy(1, 1), horizon=horizon)
    states = np.ones((2, horizon, 1))
    actions = np.array([[[0.5], [-0.5]], [[1.], [0.]]])
    rewards = np.array([[1., 2.], [3., 4.]])
    alive = np.ones((2, horizon), dtype=bool)
    weights = np.array([2., 0.5])

    target_logps = np.empty((2, horizon))
    for t in range(horizon):
        target_logps[:, t] = policy.log_prob(states[:, t], actions[:, t], t)
    behavior_logps = target_logps - np.log(weights)[..., None] / horizon

    on_policy_batch = [
        (state, action, reward, alive_flags, target_logp)
        for state, action, reward, alive_flags, target_logp
        in zip(states, actions, rewards, alive, target_logps)
    ]
    off_policy_batch = [
        (state, action, reward, alive_flags, behavior_logp)
        for state, action, reward, alive_flags, behavior_logp
        in zip(states, actions, rewards, alive, behavior_logps)
    ]

    on_policy_samples = nonstationary_pg_estimator(
        on_policy_batch, 0.9, policy, baseline=None, average=False
    )
    off_policy_samples = nonstationary_pg_estimator(
        off_policy_batch, 0.9, policy, baseline=None,
        average=False, off_policy=True
    )

    incremental_ratios = np.sqrt(weights)[..., None]
    prefix_weights = np.concatenate(
        (incremental_ratios, weights[..., None]), axis=1
    )
    discounted_rewards = rewards * np.array([1., 0.9])
    weighted_returns_to_go = np.cumsum(
        (prefix_weights * discounted_rewards)[:, ::-1], axis=1
    )[:, ::-1]
    scores = policy.score(states, actions)
    expected = (scores[..., 0] * weighted_returns_to_go).reshape(
        len(states), -1
    )

    assert not np.allclose(
        off_policy_samples, weights[..., None] * on_policy_samples
    )
    assert np.allclose(off_policy_samples, expected)


@pytest.mark.parametrize("estimator", (gpomdp_estimator, nonstationary_pg_estimator))
def test_gpomdp_like_estimators_use_prefix_importance_weights(estimator):
    class StateScorePolicy:
        state_dim = 1
        action_dim = 1

        def score(self, states, actions):
            return states

        def log_prob(self, states, actions, t=None):
            return np.zeros(states.shape[:-1])

    scores = np.array([1., 2.])
    rewards = np.array([1., 10.])
    incremental_ratios = np.array([2., 3.])
    prefix_weights = np.cumprod(incremental_ratios)
    batch = [(
        scores[..., None],
        np.ones((2, 1)),
        rewards,
        np.ones(2, dtype=bool),
        -np.log(incremental_ratios),
    )]

    samples = estimator(
        batch,
        1.,
        StateScorePolicy(),
        baseline=None,
        average=False,
        off_policy=True,
    )
    precomputed_samples = estimator(
        batch,
        1.,
        StateScorePolicy(),
        baseline=None,
        average=False,
        importance_weights=prefix_weights[None, :],
    )
    if estimator is gpomdp_estimator:
        expected = np.sum(
            prefix_weights * rewards * np.cumsum(scores)
        ).reshape(1, 1)
    else:
        weighted_returns_to_go = np.cumsum(
            (prefix_weights * rewards)[::-1]
        )[::-1]
        expected = (scores * weighted_returns_to_go).reshape(1, -1)

    assert np.allclose(samples, expected)
    assert np.array_equal(precomputed_samples, samples)


class _BernoulliPolicy:
    state_dim = 1
    action_dim = 1

    def __init__(self, probability):
        self.probability = probability

    def score(self, states, actions):
        return actions - self.probability

    def log_prob(self, states, actions, t=None):
        actions = np.asarray(actions)[..., 0]
        return np.where(
            actions == 1.,
            np.log(self.probability),
            np.log1p(-self.probability),
        )


def _one_step_batch(actions, rewards=None):
    actions = np.asarray(actions, dtype=float)
    if rewards is None:
        rewards = actions
    return [
        (
            np.zeros((1, 1)),
            np.array([[action]]),
            np.array([reward], dtype=float),
            np.ones(1, dtype=bool),
            np.zeros(1),
        )
        for action, reward in zip(actions, rewards)
    ]


def test_leave_one_out_baseline_has_unscaled_monte_carlo_expectation():
    probability = 0.3
    expected_gradient = probability * (1. - probability)
    policy = _BernoulliPolicy(probability)
    rng = np.random.default_rng(2025)
    estimates = {}

    for batch_size in (5, 100):
        zero_estimates = []
        loo_estimates = []
        for _ in range(1500):
            actions = rng.binomial(1, probability, size=batch_size)
            batch = _one_step_batch(actions)
            zero_estimates.append(
                gpomdp_estimator(batch, 1., policy, baseline="zero")[0]
            )
            loo_estimates.append(
                gpomdp_estimator(batch, 1., policy, baseline="average")[0]
            )
        estimates[batch_size] = np.asarray(loo_estimates)
        assert np.mean(zero_estimates) == pytest.approx(expected_gradient, abs=0.025)
        assert np.mean(loo_estimates) == pytest.approx(expected_gradient, abs=0.025)

    assert np.var(estimates[100]) < np.var(estimates[5])
    assert np.mean(estimates[5]) != pytest.approx(
        (1. - 1. / 5.) * expected_gradient, abs=0.02
    )


@pytest.mark.parametrize(
    "estimator", (reinforce_estimator, gpomdp_estimator, nonstationary_pg_estimator)
)
@pytest.mark.parametrize("baseline", ("average", "peters", "mastrangelo"))
def test_one_trajectory_sample_baseline_falls_back_to_zero(estimator, baseline):
    policy = _BernoulliPolicy(0.3)
    batch = _one_step_batch([1])

    sample_baseline = estimator(batch, 1., policy, baseline=baseline)
    zero = estimator(batch, 1., policy, baseline="zero")

    assert np.array_equal(sample_baseline, zero)


def test_action_independent_constant_reward_has_zero_mean_gradient():
    policy = _BernoulliPolicy(0.4)
    rng = np.random.default_rng(17)
    zero_estimates = []
    loo_estimates = []
    for _ in range(500):
        actions = rng.binomial(1, policy.probability, size=20)
        batch = _one_step_batch(actions, np.ones(20))
        zero_estimates.append(
            gpomdp_estimator(batch, 1., policy, baseline="zero")[0]
        )
        loo_estimates.append(
            gpomdp_estimator(batch, 1., policy, baseline="average")[0]
        )

    assert np.mean(zero_estimates) == pytest.approx(0., abs=0.03)
    assert np.mean(loo_estimates) == pytest.approx(0., abs=1e-14)


def test_gpomdp_leave_one_out_baseline_handles_variable_horizons():
    class StateScorePolicy:
        state_dim = 1
        action_dim = 1

        def score(self, states, actions):
            return states

    policy = StateScorePolicy()
    batch = [
        (
            np.array([[-0.3], [0.5]]),
            np.ones((2, 1)),
            np.array([1., 2.]),
            np.array([True, True]),
            np.zeros(2),
        ),
        (
            np.array([[1.], [999.]]),
            np.ones((2, 1)),
            np.array([4., 999.]),
            np.array([True, False]),
            np.zeros(2),
        ),
    ]

    samples = gpomdp_estimator(
        batch, 1., policy, baseline="average", average=False
    )

    # At t=0 each trajectory uses the other's return-to-go. At t=1 the
    # longer trajectory has no peer and therefore uses the zero fallback.
    expected = np.array([[-0.3 * (3. - 4.) + 0.5 * 2.], [1. * (4. - 3.)]])
    assert np.allclose(samples, expected)


def test_gpomdp_peters_baseline_is_weighted_leave_one_out_return_to_go():
    class StateScorePolicy:
        state_dim = 1
        action_dim = 1

        def score(self, states, actions):
            return states

    scores = np.array([1., 2., 4.])
    returns_to_go = np.array([10., 20., 30.])
    batch = [
        (
            np.array([[score]]),
            np.ones((1, 1)),
            np.array([ret]),
            np.ones(1, dtype=bool),
            np.zeros(1),
        )
        for score, ret in zip(scores, returns_to_go)
    ]

    samples = gpomdp_estimator(
        batch,
        1.,
        StateScorePolicy(),
        baseline="peters",
        average=False,
    )
    leave_one_out_baselines = np.array([
        (2.**2 * 20. + 4.**2 * 30.) / (2.**2 + 4.**2),
        (1.**2 * 10. + 4.**2 * 30.) / (1.**2 + 4.**2),
        (1.**2 * 10. + 2.**2 * 20.) / (1.**2 + 2.**2),
    ])
    expected = scores * (returns_to_go - leave_one_out_baselines)

    assert np.allclose(samples[:, 0], expected)


@pytest.mark.parametrize("off_policy", (False, True))
@pytest.mark.parametrize("baseline", (None, "average", "weighted-average"))
def test_deep_gpomdp_fast_average_matches_per_sample_estimator(
        rng, off_policy, baseline):
    state_dim = 3
    action_dim = 2
    policy = DeepGaussianPolicy(
        state_dim,
        action_dim,
        mean_network=nn.Sequential(
            nn.Linear(state_dim, 4),
            nn.Tanh(),
            nn.Linear(4, action_dim),
        ),
        std_init=np.array([0.5, 0.8]),
        learn_std=True,
    )
    n_trajectories = 4
    horizon = 5
    states = rng.normal(size=(n_trajectories, horizon, state_dim))
    actions = rng.normal(size=(n_trajectories, horizon, action_dim))
    rewards = rng.normal(size=(n_trajectories, horizon))
    alive = np.ones((n_trajectories, horizon), dtype=bool)
    alive[0, -2:] = False
    alive[1, -1] = False
    behavior_logps = policy.log_prob(states, actions)
    if off_policy:
        behavior_logps = behavior_logps - rng.normal(
            scale=0.02, size=behavior_logps.shape
        )
    batch = [
        (states[i], actions[i], rewards[i], alive[i], behavior_logps[i])
        for i in range(n_trajectories)
    ]

    samples = gpomdp_estimator(
        batch,
        0.97,
        policy,
        baseline=baseline,
        average=False,
        off_policy=off_policy,
    )
    fast_average = gpomdp_estimator(
        batch,
        0.97,
        policy,
        baseline=baseline,
        average=True,
        off_policy=off_policy,
    )

    assert np.allclose(
        fast_average,
        np.mean(samples, axis=0),
        rtol=2e-5,
        atol=2e-5,
    )


def test_stationary_importance_log_probs_are_evaluated_in_one_batch():
    class CountingStationaryPolicy:
        state_dim = 1
        action_dim = 1
        is_stationary = True

        def __init__(self):
            self.log_prob_calls = 0

        def score(self, states, actions):
            return np.ones(states.shape[:-1] + (1,))

        def log_prob(self, states, actions, t=None):
            self.log_prob_calls += 1
            return np.zeros(states.shape[:-1])

    policy = CountingStationaryPolicy()
    batch = [
        (
            np.zeros((3, 1)),
            np.zeros((3, 1)),
            np.ones(3),
            np.ones(3, dtype=bool),
            np.zeros(3),
        )
        for _ in range(2)
    ]

    gpomdp_estimator(
        batch,
        1.,
        policy,
        baseline=None,
        average=False,
        off_policy=True,
    )

    assert policy.log_prob_calls == 1


def test_importance_log_ratio_accumulation_uses_float64():
    class Float32StationaryPolicy:
        is_stationary = True

        def score(self, states, actions):
            return np.ones(states.shape[:-1] + (1,))

        def log_prob(self, states, actions, t=None):
            return np.zeros(states.shape[:-1], dtype=np.float32)

    policy = Float32StationaryPolicy()
    batch = [
        (
            np.zeros((3, 1), dtype=np.float32),
            np.zeros((3, 1), dtype=np.float32),
            np.ones(3, dtype=np.float32),
            np.ones(3, dtype=bool),
            np.zeros(3, dtype=np.float32),
        )
        for _ in range(2)
    ]

    states, actions, _, alive, behavior_logps = unpack(batch)
    importance_weights = _importance_weights(
        states, actions, alive, behavior_logps, policy, per_decision=True
    )

    assert importance_weights.dtype == np.float64
