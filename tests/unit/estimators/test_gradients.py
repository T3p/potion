import numpy as np
import pytest
from potion.estimators.gradients import reinforce_estimator, gpomdp_estimator, nonstationary_pg_estimator


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
    return MockPolicy()

@pytest.fixture
def small_batch(small_policy):
    return [(np.ones((2, small_policy.state_dim)),
              np.ones((2, small_policy.action_dim)),
              np.array([1., -1.]),
              np.array([True, True])),
             (np.ones((2, small_policy.state_dim)),
              np.ones((2, small_policy.action_dim)),
              np.array([4., -2.]),
              np.array([True, True]))
             ]


@pytest.mark.parametrize("estimator", (reinforce_estimator, gpomdp_estimator))
@pytest.mark.parametrize("baseline", (None, "average", "peters"))
def test_gradient_estimators_shapes(batch, discount, policy, n_traj, n_params, estimator, baseline):
    grad = estimator(batch, discount, policy, baseline=baseline, average=True)

    grad_samples = estimator(batch, discount, policy, baseline=baseline, average=False)

    assert grad.shape == (n_params,)

    assert grad_samples.shape == (n_traj, n_params)


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
                       batch[i][3]))

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

def test_nonstationary_pg_estimator_values(small_policy, small_batch):
    pol = small_policy

    batch = small_batch

    grad_1 = nonstationary_pg_estimator(batch, 0.9, pol, baseline=None)
    grad_2 = nonstationary_pg_estimator(batch, 0.9, pol, baseline="average")
    grad_3 = nonstationary_pg_estimator(batch, 0.9, pol, baseline="peters")

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
               np.array([True, True]))]

    batch_2 = [(np.ones((2, policy.state_dim)),
               np.ones((2, policy.action_dim - 1)),
               np.array([1., -1.]),
               np.array([True, True]))]

    with pytest.warns(UserWarning):
        _ = estimator(batch, discount, policy, baseline="xyz")

    with pytest.raises(ValueError):
        _ = estimator(batch_1, discount, policy)

    with pytest.raises(ValueError):
        _ = estimator(batch_2, discount, policy)


@pytest.mark.parametrize("estimator", (reinforce_estimator, gpomdp_estimator, nonstationary_pg_estimator))
@pytest.mark.parametrize("baseline", (None, "average", "peters"))
def test_gradient_estimators_masking(batch, discount, policy, horizon, estimator, baseline):
    grad = estimator(batch, discount, policy, baseline=baseline)

    batch_1 = []
    for i in range(len(batch)):
        s, a, r, al = batch[i]
        s[horizon:] = -100. * np.ones(policy.state_dim)
        a[horizon:] = 200. * np.ones(policy.action_dim)
        r[horizon:] = -150.
        batch_1.append((s, a, r, al))

    grad_1 = estimator(batch_1, discount, policy, baseline=baseline)

    assert np.allclose(grad_1, grad)
