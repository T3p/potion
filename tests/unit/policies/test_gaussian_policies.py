import pytest
from potion.policies.gaussian_policies import LinearGaussianPolicy, DeepGaussianPolicy
import numpy as np
from torch import nn


@pytest.fixture
def linear_gaussian_policy_1d():
    return LinearGaussianPolicy(1, 1)


@pytest.fixture
def linear_gaussian_policy(state_d, action_d):
    return LinearGaussianPolicy(state_d, action_d)


@pytest.fixture
def linear_adaptive_gaussian_policy(state_d, action_d):
    return LinearGaussianPolicy(state_d, action_d, learn_std=True)


@pytest.fixture
def deep_gaussian_policy(state_d, action_d, rng):
    def weights_init(layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                layer.bias.data.fill_(0.01)

    net = nn.Sequential(nn.Linear(state_d, 3),
                        nn.Tanh(),
                        nn.Linear(3, action_d, bias=False))
    net.apply(weights_init)

    return DeepGaussianPolicy(state_d, action_d, mean_network=net)


def test_linear_gaussian_policy_1d(linear_gaussian_policy_1d, rng):
    one = np.ones(1)
    zero = np.zeros(1)
    pol = linear_gaussian_policy_1d
    sd = pol.state_dim
    ad = pol.action_dim
    learn_std = pol.learn_std
    mean = pol.mean(one)
    action_1 = pol.act(one, rng)
    action_2 = pol.act(one, rng)
    std = pol.std
    log_pdf_10 = pol.log_prob(one, zero)
    log_pdf_11 = pol.log_prob(one, one)
    score_10 = pol.score(one, zero)
    score_11 = pol.score(one, one)
    entropy = pol.entropy(one)
    entropy_grad = pol.entropy_grad(one)

    assert sd == 1
    assert ad == 1
    assert not learn_std
    assert pol.parameters == 0.
    assert pol.parameters.shape == (1,)
    assert mean == 0.
    assert mean.shape == (1,)
    assert action_1.shape == (1,)
    assert not np.allclose(action_1, action_2)
    assert std == 1.
    assert np.isscalar(std)
    assert np.allclose(log_pdf_10, -0.5 * np.log(2 * np.pi))
    assert np.allclose(log_pdf_11, -0.5 * np.log(2 * np.pi) - 0.5)
    assert np.allclose(score_10, 0.)
    assert np.allclose(score_11, 1.)
    assert np.isscalar(pol.entropy(one))
    assert np.isclose(entropy, 0.5 * np.log(2 * np.pi * np.e))
    assert np.isclose(entropy_grad, 0.)
    assert entropy_grad.shape == (1,)


def test_linear_gaussian_policy_default(linear_gaussian_policy, rng, state_d, action_d):
    s_1 = np.ones(state_d)
    a_0 = np.zeros(action_d)
    a_1 = np.ones(action_d)
    pol = linear_gaussian_policy
    sd = pol.state_dim
    ad = pol.action_dim
    pd = state_d * action_d
    mean = pol.mean(s_1)
    action_1 = pol.act(s_1, rng)
    action_2 = pol.act(s_1, rng)
    std = pol.std
    log_pdf_10 = pol.log_prob(s_1, a_0)
    log_pdf_11 = pol.log_prob(s_1, a_1)
    score_10 = pol.score(s_1, a_0)
    score_11 = pol.score(s_1, a_1)
    entropy = pol.entropy(s_1)
    entropy_grad = pol.entropy_grad(s_1)

    assert sd == state_d
    assert ad == action_d
    assert np.allclose(pol.parameters, 0.)
    assert mean.shape == (action_d,)
    assert np.allclose(mean, 0.)
    assert action_1.shape == (action_d,)
    assert not np.allclose(action_1, action_2)
    assert np.allclose(std, 1.)
    assert std.ndim == 1 and len(std) == action_d
    assert np.allclose(log_pdf_10, -0.5 * np.log(2 * np.pi) * action_d)
    assert np.allclose(log_pdf_11, (-0.5 * np.log(2 * np.pi) - 0.5) * action_d)
    assert np.allclose(score_10, np.zeros(pd))
    assert score_10.shape == (pd,)
    assert np.allclose(score_11, np.ones(pd))
    assert np.isscalar(entropy)
    assert np.allclose(entropy, action_d * 0.5 * (np.log(2. * np.pi) + 1.))
    assert np.allclose(entropy_grad, np.zeros(pd))


def test_linear_gaussian_policy_initialization(state_d, action_d):
    pol1 = LinearGaussianPolicy(state_d, action_d, mean_params_init=np.eye(action_d, state_d))
    pol2 = LinearGaussianPolicy(state_d, action_d, mean_params_init=np.ravel(np.eye(action_d, state_d)))
    pol3 = LinearGaussianPolicy(state_d, action_d, std_init=0.5)
    pol4 = LinearGaussianPolicy(state_d, action_d, std_init=0.5 * np.ones(action_d))
    pol5 = LinearGaussianPolicy(state_d, action_d, mean_params_init=1.)

    assert np.allclose(pol1.parameters, np.ravel(np.eye(action_d, state_d)))
    assert np.allclose(pol1.parameters, pol2.parameters)
    assert pol1.parameters.ndim == 1 and len(pol1.parameters) == state_d * action_d
    assert pol2.parameters.ndim == 1 and len(pol2.parameters) == state_d * action_d
    assert pol3.std.ndim == 1 and len(pol3.std) == action_d
    assert np.allclose(pol3.std, 0.5)
    assert pol4.std.shape == pol3.std.shape
    assert np.allclose(pol4.std, pol3.std * np.ones(action_d))
    assert pol5.parameters.ndim == 1 and len(pol5.parameters) == state_d * action_d
    assert np.allclose(pol5.parameters, 1.)


def test_linear_gaussian_policy(rng):
    pol = LinearGaussianPolicy(2, 3, mean_params_init=np.array([[0., -1.], [2., 1.], [0.5, 0.]]),
                               std_init=np.array([0.5, 2., 1.]))
    s = np.array([0.5, 2.])
    s1 = rng.uniform(low=0., high=1., size=2)
    c1 = rng.uniform(low=0., high=1.)
    s2 = rng.uniform(low=0., high=1., size=2)
    c2 = rng.uniform(low=0., high=1.)
    a = np.array([0.5, -1., 2.])
    params = pol.parameters
    mean = pol.mean(s)
    log_pdf = pol.log_prob(s, a)
    score = pol.score(s, a)
    entropy = pol.entropy(s)
    entropy_grad = pol.entropy_grad(s)

    assert params.shape == (6,)
    assert np.allclose(params, [0., -1., 2., 1., 0.5, 0.])
    assert np.allclose(mean, np.array([-2., 3., 0.25]))
    assert np.allclose(pol.mean(c1 * s1 + c2 * s2), c1 * pol.mean(s1) + c2 * pol.mean(s2))
    assert np.allclose(log_pdf, -0.5 * np.log(2 * np.pi) * 3 - np.log(0.5) - np.log(2.) - 14.5 - 49. / 32)
    assert np.allclose(score, np.array([5., 20., -0.5, -2., 7. / 8, 3.5]))
    assert np.allclose(entropy, 1.5 * (np.log(2 * np.pi) + 1) + np.log(0.5) + np.log(2.))
    assert np.allclose(entropy_grad, np.zeros(6))
    assert pol.num_std_params == 0
    assert pol.num_params == 6 and pol.num_mean_params == pol.num_params


def test_linear_adaptive_gaussian_policy(rng):
    pol1 = LinearGaussianPolicy(2, 3, mean_params_init=np.array([[0., -1.], [2., 1.], [0.5, 0.]]),
                                std_init=np.array([0.5, 2., 1.]), learn_std=True)

    pol2 = LinearGaussianPolicy(2, 3, mean_params_init=np.array([[0., -1.], [2., 1.], [0.5, 0.]]),
                                std_init=0.5, learn_std=True)

    pol3 = LinearGaussianPolicy(1, 1, mean_params_init=0.,
                                std_init=0.5, learn_std=True)

    s = np.array([0.5, 2.])
    s1 = rng.uniform(low=0., high=1., size=2)
    c1 = rng.uniform(low=0., high=1.)
    s2 = rng.uniform(low=0., high=1., size=2)
    c2 = rng.uniform(low=0., high=1.)
    a = np.array([0.5, -1., 2.])
    x = 0.5 * np.ones(1)

    assert pol1.parameters.shape == (9,)
    assert pol1.num_params == 9 and pol1.num_mean_params + pol1.num_std_params == pol1.num_params
    assert np.allclose(pol1.parameters, [0., -1., 2., 1., 0.5, 0., np.log(0.5), np.log(2.), 0.])
    assert np.allclose(pol1.mean(s), np.array([-2., 3., 0.25]))
    assert np.allclose(pol1.mean(c1 * s1 + c2 * s2), c1 * pol1.mean(s1) + c2 * pol1.mean(s2))
    assert np.allclose(pol1.log_prob(s, a), -0.5 * np.log(2 * np.pi) * 3 - np.log(0.5) - np.log(2.) - 14.5 - 49. / 32)
    assert np.allclose(pol1.score(s, a), np.array([5., 20., -0.5, -2., 7. / 8, 3.5, 24., 3., 33. / 16]))
    assert np.allclose(pol1.entropy(s), 1.5 * (np.log(2 * np.pi) + 1) + np.log(0.5) + np.log(2.))
    assert np.allclose(pol1.entropy_grad(s), np.concatenate((np.zeros(6), np.ones(3)), axis=-1))

    assert pol2.parameters.shape == (7,)
    assert pol2.num_params == 7 and pol2.num_mean_params + pol2.num_std_params == pol2.num_params
    assert np.allclose(pol2.parameters, [0., -1., 2., 1., 0.5, 0., np.log(0.5)])
    assert np.allclose(pol2.log_prob(s, a), -0.5 * np.log(2 * np.pi) * 3 - 3 * np.log(0.5) - 50.625)
    assert np.allclose(pol2.score(s, a), np.array([5., 20., -8., -32., 3.5, 14., 24. + 63 + 45. / 4]))
    assert np.allclose(pol2.entropy(s), 1.5 * (np.log(2 * np.pi) + 1) + 3 * np.log(0.5))
    assert np.allclose(pol2.entropy_grad(s), np.concatenate((np.zeros(6), 3 * np.ones(1)), axis=-1))

    assert pol3.parameters.shape == (2,)
    assert pol3.num_params == 2 and pol3.num_mean_params + pol3.num_std_params == pol3.num_params
    assert np.allclose(pol3.parameters, [0., np.log(0.5)])
    assert np.allclose(pol3.log_prob(x, x), -0.5 * np.log(2 * np.pi) - np.log(0.5) - 0.5)
    assert np.allclose(pol3.score(x, x), np.array([1., 0.]))
    assert np.allclose(pol3.entropy(x), 0.5 * (np.log(2 * np.pi) + 1) + np.log(0.5))
    assert np.allclose(pol3.entropy_grad(x), np.concatenate((np.zeros(1), np.ones(1)), axis=-1))


def test_linear_gaussian_policy_setters(linear_gaussian_policy_1d, linear_gaussian_policy, state_d, action_d, rng):
    pd = state_d * action_d
    pol1 = linear_gaussian_policy_1d
    pol1.set_params(np.ones(1))

    pol2 = linear_gaussian_policy
    pol2.set_params(np.ones(pd))

    pol3 = LinearGaussianPolicy(state_d, action_d)
    pol3.set_params(np.ones((action_d, state_d)))

    pol4 = LinearGaussianPolicy(state_d, action_d, learn_std=True)
    pol4.set_params(np.array([1.] * pd + [np.log(0.5)]))

    pol5 = LinearGaussianPolicy(state_d, action_d, learn_std=True, std_init=np.ones(action_d))
    pol5.set_params(np.concatenate((np.ones(pd), np.log(0.5) * np.ones(action_d))))

    pol6 = LinearGaussianPolicy(state_d, action_d, learn_std=False, std_init=1.)
    pol6.set_std(0.5)

    pol7 = LinearGaussianPolicy(state_d, action_d, learn_std=False, std_init=np.ones(action_d))
    pol7.set_std(0.5)

    pol8 = LinearGaussianPolicy(state_d, action_d, learn_std=False, std_init=np.ones(action_d))
    pol8.set_std(0.5 + np.zeros(action_d))

    pol9 = LinearGaussianPolicy(state_d, action_d, learn_std=False)
    pol9.set_params(2.)

    pol10 = LinearGaussianPolicy(state_d, action_d, learn_std=True)
    pol10.set_params(2.)

    pol11 = LinearGaussianPolicy(state_d, action_d, learn_std=True, std_init=np.ones(action_d))
    pol11.set_params(2.)

    assert pol1.parameters.shape == (1,)
    assert np.isclose(pol1.parameters, 1.)
    assert np.isscalar(pol1.std)
    assert np.isclose(pol1.std, 1.)

    assert pol2.parameters.shape == (pd,)
    assert np.allclose(pol2.parameters, 1.)
    assert pol2.std.shape == (action_d,)
    assert np.allclose(pol2.std, 1.)

    assert pol3.parameters.shape == pol2.parameters.shape
    assert np.allclose(pol3.parameters, pol2.parameters)
    assert pol3.std.shape == pol3.std.shape
    assert np.allclose(pol3.std, pol2.std)

    assert pol4.parameters.shape == (pd + 1,)
    assert np.allclose(pol4.parameters, np.array([1.] * pd + [np.log(0.5)]))
    assert pol4.std.shape == (action_d,)
    assert np.allclose(pol4.std, 0.5)

    assert pol5.parameters.shape == (pd + action_d,)
    assert np.allclose(pol5.parameters[:pd], pol4.parameters[:pd])
    assert np.allclose(pol5.parameters[pd:], pol4.parameters[-1])
    assert pol5.std.shape == pol4.std.shape
    assert np.allclose(pol5.std, pol4.std)

    assert pol6.std.shape == (action_d,)
    assert np.allclose(pol6.std, 0.5)

    assert pol7.std.shape == (action_d,)
    assert np.allclose(pol7.std, 0.5)

    assert pol8.std.shape == (action_d,)
    assert np.allclose(pol8.std, 0.5)

    assert np.allclose(pol9.parameters, 2.)
    assert pol9.parameters.shape == (state_d * action_d,)
    assert np.allclose(pol9.mean(np.ones(state_d)), 2. * state_d)

    assert np.allclose(pol10.parameters, 2.)
    assert pol10.parameters.shape == (state_d * action_d + 1,)
    assert np.allclose(pol10.mean(np.ones(state_d)), 2. * state_d)
    assert np.allclose(pol10.std, np.exp(2.))

    assert np.allclose(pol11.parameters, 2.)
    assert pol11.parameters.shape == (state_d * action_d + action_d,)
    assert np.allclose(pol10.mean(np.ones(state_d)), 2. * state_d)
    assert np.allclose(pol10.std, np.exp(2.))


def test_gaussian_exceptions(linear_gaussian_policy_1d, linear_gaussian_policy, linear_adaptive_gaussian_policy,
                             state_d, action_d, rng):
    s = np.ones(state_d)
    bad_s = np.ones(state_d + 1)
    a = np.ones(action_d)
    bad_a = np.ones(action_d - 1)
    batch_s = np.ones((4, state_d))
    batch_a = np.ones((5, action_d))

    with pytest.raises(ValueError):
        _ = LinearGaussianPolicy(state_d, action_d, mean_params_init=np.ones(state_d * action_d + 1))
    with pytest.raises(ValueError):
        _ = LinearGaussianPolicy(state_d, action_d, mean_params_init=np.ones(state_d * action_d - 1))
    with pytest.raises(ValueError):
        _ = LinearGaussianPolicy(state_d, action_d, mean_params_init=np.ones((2 * state_d, action_d)))
    with pytest.raises(ValueError):
        _ = LinearGaussianPolicy(state_d, action_d, mean_params_init=np.ones((state_d, 2 * action_d)))
    with pytest.raises(ValueError):
        _ = LinearGaussianPolicy(state_d, action_d, std_init=-0.5)
    with pytest.raises(ValueError):
        _ = LinearGaussianPolicy(state_d, action_d, std_init=np.ones(action_d + 1))
    with pytest.raises(ValueError):
        _ = LinearGaussianPolicy(state_d, action_d, std_init=np.ones(action_d - 1))
    with pytest.raises(ValueError):
        _ = LinearGaussianPolicy(state_d, 1, std_init=np.ones(1))

    pol1 = linear_gaussian_policy_1d
    with pytest.raises(ValueError):
        pol1.set_params(np.ones(2))
    with pytest.raises(ValueError):
        pol1.set_std(np.ones(2))
    with pytest.raises(ValueError):
        pol1.set_std(np.ones(1))
    with pytest.raises(ValueError):
        pol1.set_std(-0.5)

    pol2 = linear_gaussian_policy
    with pytest.raises(ValueError):
        pol2.set_params(np.ones(state_d * action_d - 1))
    with pytest.raises(ValueError):
        pol2.set_params(np.ones(state_d * action_d + 1))
    with pytest.raises(ValueError):
        pol2.set_params(np.ones((state_d, action_d, 1)))
    with pytest.raises(ValueError):
        pol2.set_std(np.ones(action_d + 1))
    with pytest.raises(ValueError):
        pol2.set_std(np.ones(action_d - 1))
    with pytest.raises(ValueError):
        std = np.ones(action_d)
        std[1] = -0.5
        pol2.set_std(std)

    pol3 = linear_adaptive_gaussian_policy
    with pytest.raises(ValueError):
        pol3.set_params(np.ones(state_d * action_d + pol3.num_std_params - 1))
    with pytest.raises(ValueError):
        pol3.set_params(np.ones(state_d * action_d + pol3.num_std_params + 1))
    with pytest.raises(ValueError):
        pol3.set_params(np.ones((state_d, action_d, 1)))
    with pytest.raises(RuntimeError):
        pol3.set_std(np.ones((state_d, action_d)))

    pol4 = LinearGaussianPolicy(state_d, action_d, learn_std=True, std_init=np.ones(action_d))
    with pytest.raises(ValueError):
        pol4.set_params(np.ones(state_d * action_d + pol4.num_std_params - 1))
    with pytest.raises(ValueError):
        pol4.set_params(np.ones(state_d * action_d + pol4.num_std_params + 1))
    with pytest.raises(ValueError):
        pol4.set_params(np.ones((state_d, action_d, 1)))
    with pytest.raises(RuntimeError):
        pol4.set_std(np.ones((state_d, action_d)))

    with pytest.raises(ValueError):
        _ = pol4.mean(bad_s)

    with pytest.raises(ValueError):
        _ = pol4.act(bad_s, rng)

    with pytest.raises(ValueError):
        _ = pol4.log_prob(bad_s, a)
    with pytest.raises(ValueError):
        _ = pol4.log_prob(s, bad_a)
    with pytest.raises(ValueError):
        _ = pol4.log_prob(batch_s, batch_a)

    with pytest.raises(ValueError):
        _ = pol4.score(bad_s, a)
    with pytest.raises(ValueError):
        _ = pol4.score(s, bad_a)
    with pytest.raises(ValueError):
        _ = pol4.score(batch_s, batch_a)

    with pytest.raises(ValueError):
        _ = pol4.entropy(bad_s)

    with pytest.raises(ValueError):
        _ = pol4.entropy_grad(bad_s)


def test_deep_gaussian_policy_linear(linear_gaussian_policy, state_d, action_d, rng):
    pol = linear_gaussian_policy
    nn_pol = DeepGaussianPolicy(state_d, action_d, mean_network=None)

    s = np.ones(state_d)
    a = np.ones(action_d)
    mean = pol.mean(s)
    nn_mean = nn_pol.mean(s)
    score = pol.score(s, a)
    nn_score = nn_pol.score(s, a)

    assert np.allclose(pol.parameters, nn_pol.parameters)
    assert np.allclose(mean, nn_mean)
    assert np.allclose(score, nn_score)

    params = rng.normal(size=pol.num_params)
    pol.set_params(params)
    nn_pol.set_params(params)
    mean = pol.mean(s)
    nn_mean = nn_pol.mean(s)
    score = pol.score(s, a)
    nn_score = nn_pol.score(s, a)

    assert np.allclose(pol.parameters, nn_pol.parameters)
    assert np.allclose(mean, nn_mean)
    assert np.allclose(score, nn_score)

    nn_pol.set_params(2 * nn_pol.parameters)
    new_mean = nn_pol.mean(s)
    assert np.allclose(new_mean, 2 * nn_mean)


def test_deep_gaussian_policy_nn(state_d, action_d, deep_gaussian_policy, rng):
    pol = deep_gaussian_policy
    s = rng.normal(size=state_d)
    a = rng.normal(size=action_d)

    played = pol.act(s, rng)
    score = pol.score(s, a)

    assert pol.num_std_params + pol.num_mean_params == pol.num_params
    assert isinstance(played, np.ndarray) and played.shape == (action_d, )
    assert isinstance(score, np.ndarray) and score.shape == (pol.num_params,)
    assert not np.isnan(score).any()
    assert not np.allclose(score, 0.)


def test_deep_gaussian_policy_exceptions(state_d, action_d):
    net_1 = nn.Linear(state_d + 1, action_d)
    net_2 = nn.Linear(state_d, action_d + 1)

    with pytest.raises(ValueError):
        _ = DeepGaussianPolicy(state_d, action_d, mean_network=net_1)

    with pytest.raises(ValueError):
        _ = DeepGaussianPolicy(state_d, action_d, mean_network=net_2)


def test_deep_gaussian_policy_adaptive_std(state_d, action_d, deep_gaussian_policy, rng):
    pol1 = DeepGaussianPolicy(state_d, action_d, learn_std=False)
    pol2 = DeepGaussianPolicy(state_d, action_d, learn_std=True)

    mean_params = rng.normal(size=pol1.num_params)
    pol2.set_params(np.concatenate((mean_params, np.array([0.5]))))
    params = pol2.parameters

    assert pol2.num_mean_params + pol2.num_std_params == pol2.num_params
    assert pol2.num_params == pol1.num_params + 1
    assert np.allclose(params[:-1], mean_params)
    assert np.isclose(params[-1], 0.5)


def test_deep_gaussian_policy_broadcast(deep_gaussian_policy):
    pol = deep_gaussian_policy
    pol.set_params(1.)

    assert np.allclose(pol.parameters, np.ones(pol.num_params))
