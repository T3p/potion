import pytest
from potion.policies.softmax_policies import LinearSoftmaxPolicy, DeepSoftmaxPolicy
import numpy as np
from torch import nn


@pytest.fixture
def linear_softmax_policy(state_d, num_actions):
    return LinearSoftmaxPolicy(state_d, num_actions)


def test_linear_softmax_policy_default(linear_softmax_policy, rng, state_d, num_actions):
    s_1 = np.ones(state_d)
    a_0 = np.array([0], dtype=int)
    a_1 = np.array([1], dtype=int)
    pol = linear_softmax_policy
    sd = pol.state_dim
    na = pol.num_actions
    pd = num_actions * state_d
    logits = pol.logits(s_1)
    action_1 = pol.act(s_1, rng)
    action_2 = pol.act(s_1, rng)
    temp = pol.temperature
    log_pdf_10 = pol.log_prob(s_1, a_0)
    log_pdf_11 = pol.log_prob(s_1, a_1)
    score_10 = pol.score(s_1, a_0)
    score_11 = pol.score(s_1, a_1)
    entropy = pol.entropy(s_1)
    entropy_grad = pol.entropy_grad(s_1)

    assert sd == state_d
    assert na == num_actions
    assert np.allclose(pol.parameters, 0.)
    assert logits.shape == (num_actions,)
    assert np.allclose(logits, 0.)
    assert action_1.shape == (1,) and np.issubdtype(action_1.dtype, np.integer) and 0 <= action_1 < num_actions
    assert not np.allclose(action_1, action_2)
    assert np.isscalar(temp) and np.isclose(temp, 1.)
    assert np.allclose(log_pdf_10, np.log(0.5))
    assert np.allclose(log_pdf_11, np.log(0.5))
    assert np.allclose(score_10, [0.5] * sd + [-0.5] * sd)
    assert score_10.shape == (pd,)
    assert np.allclose(score_11, [-0.5] * sd + [0.5] * sd)
    assert np.isscalar(entropy)
    assert np.isclose(entropy, np.log(2.))
    assert np.allclose(entropy_grad, np.zeros(pd))


def test_linear_softmax_policy_initialization(state_d, num_actions):
    pol1 = LinearSoftmaxPolicy(state_d, num_actions, params_init=np.eye(num_actions, state_d))
    pol2 = LinearSoftmaxPolicy(state_d, num_actions, params_init=np.ravel(np.eye(num_actions, state_d)))
    pol3 = LinearSoftmaxPolicy(state_d, num_actions, params_init=1.)
    pol4 = LinearSoftmaxPolicy(state_d, num_actions, temperature=0.5)

    assert np.allclose(pol1.parameters, np.ravel(np.eye(num_actions, state_d)))
    assert np.allclose(pol1.parameters, pol2.parameters)
    assert pol1.parameters.ndim == 1 and len(pol1.parameters) == num_actions * state_d
    assert pol2.parameters.ndim == 1 and len(pol2.parameters) == num_actions * state_d
    assert np.isscalar(pol4.temperature)
    assert np.isclose(pol4.temperature, 0.5)
    assert pol3.parameters.ndim == 1 and len(pol3.parameters) == num_actions * state_d
    assert np.allclose(pol3.parameters, 1.)


def test_linear_softmax_policy(rng):
    pol = LinearSoftmaxPolicy(2, 3, params_init=np.array([[0., -1.], [2., 1.], [0.5, 0.]]),
                              temperature=0.5)
    s = np.array([0.5, 2.])
    s1 = rng.uniform(low=0., high=1., size=2)
    c1 = rng.uniform(low=0., high=1.)
    s2 = rng.uniform(low=0., high=1., size=2)
    c2 = rng.uniform(low=0., high=1.)
    a = np.array((1,))
    params = pol.parameters
    logits = pol.logits(s)
    log_pdf = pol.log_prob(s, a)
    score = pol.score(s, a)
    entropy = pol.entropy(s)
    entropy_grad = pol.entropy_grad(s)

    Z = (np.exp(-4.) + np.exp(6.) + np.exp(0.5))
    score_offset = (np.exp(-4.) * np.array([0.5, 2., 0., 0., 0., 0.])
                    + np.exp(6.) * np.array([0., 0., 0.5, 2., 0., 0.])
                    + np.exp(0.5) * np.array([0., 0., 0., 0., 0.5, 2.])) / Z
    score_0 = 2. * (np.array([0.5, 2., 0., 0., 0., 0.]) - score_offset)
    score_1 = 2. * (np.array([0., 0., 0.5, 2., 0., 0.]) - score_offset)
    score_2 = 2. * (np.array([0., 0., 0., 0., 0.5, 2.]) - score_offset)

    assert pol.num_params == 6
    assert params.shape == (6,)
    assert np.allclose(params, [0., -1., 2., 1., 0.5, 0.])
    assert np.allclose(logits, np.array([-2., 3., 0.25]))
    assert np.allclose(pol.logits(c1 * s1 + c2 * s2), c1 * pol.logits(s1) + c2 * pol.logits(s2))
    assert np.allclose(log_pdf, 6. - np.log(np.exp(-4.) + np.exp(6.) + np.exp(0.5)))
    assert np.allclose(score, score_1)
    assert np.allclose(entropy, - (np.exp(-4.) * (-4.) + np.exp(6.) * 6. + np.exp(0.5) * 0.5) / Z + np.log(Z))
    assert np.allclose(entropy_grad, - (np.exp(-4.) * (-4.) * score_0 + np.exp(6.) * 6. * score_1
                                        + np.exp(0.5) * 0.5 * score_2) / Z)


def test_linear_softmax_policy_setters(linear_softmax_policy, state_d, num_actions):
    pd = state_d * num_actions

    pol1 = linear_softmax_policy
    pol1.set_params(np.ones(pd))

    pol2 = LinearSoftmaxPolicy(state_d, num_actions)
    pol2.set_params(np.ones((num_actions, state_d)))

    pol3 = LinearSoftmaxPolicy(state_d, num_actions)
    pol3.set_params(2.)

    pol4 = LinearSoftmaxPolicy(state_d, num_actions, temperature=1.)
    pol4.set_temperature(0.5)

    assert pol1.parameters.shape == (pd,)
    assert np.allclose(pol1.parameters, 1.)

    assert pol2.parameters.shape == pol1.parameters.shape
    assert np.allclose(pol2.parameters, pol1.parameters)

    assert pol3.parameters.shape == (pd,)
    assert np.allclose(pol3.parameters, 2.)

    assert np.isclose(pol4.temperature, 0.5)

def test_softmax_exceptions(linear_softmax_policy, state_d, num_actions, rng):
    s = np.ones(state_d)
    bad_s = np.ones(state_d + 1)
    a = np.array((1,))
    batch_s = np.ones((4, state_d))
    batch_a = np.ones((5, 1))
    bad_param = np.ones((state_d + 1, num_actions))
    pol1 = linear_softmax_policy

    with pytest.raises(ValueError):
        LinearSoftmaxPolicy(state_d, num_actions, temperature=-1.)

    with pytest.raises(ValueError):
        pol1.set_temperature(-1.)

    with pytest.raises(ValueError):
        LinearSoftmaxPolicy(state_d, num_actions, params_init=bad_param)

    with pytest.raises(ValueError):
        pol1.set_params(bad_param)

    with pytest.raises(ValueError):
        pol1.logits(bad_s)

    with pytest.raises(ValueError):
        pol1.act(bad_s, rng)

    with pytest.raises(ValueError):
        pol1.entropy(bad_s, rng)

    with pytest.raises(ValueError):
        pol1.entropy_grad(bad_s, rng)

    with pytest.raises(ValueError):
        pol1.log_prob(bad_s, a)
    with pytest.raises(ValueError):
        pol1.log_prob(s, np.array((-1,)))
    with pytest.raises(ValueError):
        pol1.log_prob(s, np.array((num_actions,)))
    with pytest.raises(ValueError):
        pol1.log_prob(s, np.array((0.5,)))
    with pytest.raises(ValueError):
        pol1.log_prob(batch_s, batch_a)

    with pytest.raises(ValueError):
        pol1.score(bad_s, a)
    with pytest.raises(ValueError):
        pol1.score(s, np.array((-1,)))
    with pytest.raises(ValueError):
        pol1.score(s, np.array((num_actions,)))
    with pytest.raises(ValueError):
        pol1.score(s, np.array((0.5,)))
    with pytest.raises(ValueError):
        pol1.score(batch_s, batch_a)

@pytest.fixture
def deep_softmax_policy(state_d, num_actions, rng):
    def weights_init(layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                layer.bias.data.fill_(0.01)

    net = nn.Sequential(nn.Linear(state_d, 3),
                        nn.Tanh(),
                        nn.Linear(3, num_actions, bias=False))
    net.apply(weights_init)

    return DeepSoftmaxPolicy(state_d, num_actions, logit_network=net)


def test_deep_softmax_policy_linear(linear_softmax_policy, state_d, num_actions, rng):
    pol = linear_softmax_policy
    nn_pol = DeepSoftmaxPolicy(state_d, num_actions, logit_network=None)

    s = np.ones(state_d)
    a = np.array((1,))
    logits = pol.logits(s)
    nn_logits = nn_pol.logits(s)
    score = pol.score(s, a)
    nn_score = nn_pol.score(s, a)

    assert np.allclose(pol.parameters, nn_pol.parameters)
    assert np.allclose(logits, nn_logits)
    assert np.allclose(score, nn_score)

    params = rng.normal(size=pol.num_params)
    pol.set_params(params)
    nn_pol.set_params(params)
    logits = pol.logits(s)
    nn_logits = nn_pol.logits(s)
    score = pol.score(s, a)
    nn_score = nn_pol.score(s, a)

    assert np.allclose(pol.parameters, nn_pol.parameters)
    assert np.allclose(logits, nn_logits)
    assert np.allclose(score, nn_score)

    nn_pol.set_params(2 * nn_pol.parameters)
    new_logits = nn_pol.logits(s)
    assert np.allclose(new_logits, 2 * logits)


def test_deep_softmax_policy_nn(state_d, num_actions, deep_softmax_policy, rng):
    pol = deep_softmax_policy
    s = rng.normal(size=state_d)
    a = rng.choice(range(num_actions))

    played = pol.act(s, rng)
    score = pol.score(s, a)
    entropy_grad = pol.entropy_grad(s)

    assert isinstance(played, np.ndarray) and played.shape == (1,)

    assert isinstance(score, np.ndarray) and score.shape == (pol.num_params,)
    assert not np.isnan(score).any()
    assert not np.allclose(score, 0.)

    assert isinstance(entropy_grad, np.ndarray) and entropy_grad.shape == (pol.num_params,)
    assert not np.isnan(entropy_grad).any()
    assert not np.allclose(entropy_grad, 0.)


def test_deep_softmax_policy_exceptions(state_d, num_actions, deep_softmax_policy):
    net_1 = nn.Linear(state_d + 1, num_actions)
    net_2 = nn.Linear(state_d, num_actions + 1)
    params = np.ones(shape=(deep_softmax_policy.num_params + 1,))

    with pytest.raises(ValueError):
        _ = DeepSoftmaxPolicy(state_d, num_actions, logit_network=net_1)

    with pytest.raises(ValueError):
        _ = DeepSoftmaxPolicy(state_d, num_actions, logit_network=net_2)

    with pytest.raises(ValueError):
        deep_softmax_policy.set_params(params)


def test_deep_softmax_policy_broadcast(deep_softmax_policy):
    pol = deep_softmax_policy
    pol.set_params(1.)

    assert np.allclose(pol.parameters, np.ones(pol.num_params))
"""







def test_deep_gaussian_policy_exceptions(state_d, action_d):
    net_1 = nn.Linear(state_d + 1, action_d)
    net_2 = nn.Linear(state_d, action_d + 1)

    with pytest.raises(ValueError):
        _ = DeepGaussianPolicy(state_d, action_d, mean_network=net_1)

    with pytest.raises(ValueError):
        _ = DeepGaussianPolicy(state_d, action_d, mean_network=net_2)





def test_deep_gaussian_policy_broadcast(deep_gaussian_policy):
    pol = deep_gaussian_policy
    pol.set_params(1.)

    assert np.allclose(pol.parameters, np.ones(pol.num_params))
"""