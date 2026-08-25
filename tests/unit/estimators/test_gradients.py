import numpy as np
import pytest
from potion.estimators.gradients import reinforce_estimator, gpomdp_estimator, nonstationary_pg_estimator
from potion.policies.gaussian_policies import LinearGaussianPolicy
from potion.policies.wrappers import Staged


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
    grad = estimator(batch, discount, policy, baseline="average", average=True)

    grad_samples = estimator(batch, discount, policy, baseline="average", average=False)

    expected_num_params = n_params if estimator is not nonstationary_pg_estimator else max_trajectory_len * n_params
    assert grad.shape == (expected_num_params,)

    assert grad_samples.shape == (n_traj, expected_num_params)


@pytest.mark.parametrize("estimator", (reinforce_estimator, gpomdp_estimator, nonstationary_pg_estimator))
@pytest.mark.parametrize("baseline", (None, "average", "peters"))
def test_gradient_estimators_invariance(batch, discount, policy, n_traj, estimator, baseline):
    grad = estimator(batch, discount, policy, baseline=baseline, average=True)
    samples = estimator(batch, discount, policy, baseline=baseline, average=False)
    batch_2 = []
    for i in range(n_traj):
        batch_2.append((batch[i][0],
                       batch[i][1],
                       2. * batch[i][2],  # double rewards
                       batch[i][3],
                       batch[i][4]))

    grad_2 = estimator(batch_2, discount, policy, baseline=baseline, average=True)

    batch_3 = batch[::-1]  # reverse

    grad_3 = estimator(batch_3, discount, policy, baseline=baseline, average=True)
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

    on_policy_samples = estimator(small_batch, 0.9, small_policy, baseline=None, average=False)
    ignored_logps_samples = estimator(weighted_batch, 0.9, small_policy, baseline=None, average=False)
    off_policy_samples = estimator(
        weighted_batch, 0.9, small_policy, baseline=None, average=False, off_policy=True
    )
    off_policy_grad = estimator(weighted_batch, 0.9, small_policy, baseline=None, off_policy=True)

    assert np.allclose(ignored_logps_samples, on_policy_samples)
    assert np.allclose(off_policy_samples, weights[..., None] * on_policy_samples)
    assert np.allclose(off_policy_grad, np.mean(off_policy_samples, axis=0))


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
        off_policy_batch, 0.9, policy, baseline=None, average=False, off_policy=True
    )

    assert np.allclose(off_policy_samples, weights[..., None] * on_policy_samples)


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
@pytest.mark.parametrize("baseline", ("average", "peters"))
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
            gpomdp_estimator(
                batch,
                1.,
                policy,
                baseline="average",
            )[0]
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
