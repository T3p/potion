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
    g2 = -0.3675
    g3 = -0.25344827586206903

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
    g2 = 1.1325
    g3 = 0.6453179373615945

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
