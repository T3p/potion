import numpy as np
import pytest

from potion.algorithms import pagepg, reinforce
from potion.evaluation.loggers import SilentLogger
from potion.policies.gaussian_policies import LinearGaussianPolicy


def test_pagepg_recursive_gradient_and_default_probability(env, policy, n_params, mocker):
    rng = mocker.Mock()
    rng.random.return_value = 0.9
    mocker.patch(
        "potion.algorithms.pagepg.initialize_run",
        side_effect=lambda seed, logger: (rng, mocker.Mock(), logger),
    )
    generate_batch = mocker.patch(
        "potion.algorithms.pagepg.generate_batch",
        side_effect=lambda env, policy, n_episodes, horizon, **kwargs: [None] * n_episodes,
    )
    estimator = mocker.patch("potion.algorithms.pagepg.gpomdp_estimator")
    estimator.side_effect = lambda batch, discount, policy, baseline, **kwargs: np.full(
        n_params, 2. if kwargs.get("off_policy") else 3.
    )
    adaptive_step = mocker.Mock(return_value=np.zeros(n_params))

    pagepg(env, policy,
           batch_size=7,
           mini_batch_size=2,
           max_iterations=3,
           step_size=adaptive_step,
           logger=SilentLogger(),
           verbose=False)

    assert [call.args[2] for call in generate_batch.call_args_list] == [7, 2, 2]
    assert [call.kwargs["reset"] for call in adaptive_step.call_args_list] == [
        True, False, False
    ]
    expected_gradients = [3., 4., 5.]
    for call, expected in zip(adaptive_step.call_args_list, expected_gradients):
        assert np.allclose(call.args[0], expected * np.ones(n_params))
    assert rng.random.call_count == 2


def test_pagepg_large_batch_refresh(env, policy, n_params, mocker):
    generate_batch = mocker.patch(
        "potion.algorithms.pagepg.generate_batch",
        side_effect=lambda env, policy, n_episodes, horizon, **kwargs: [None] * n_episodes,
    )
    estimator = mocker.patch(
        "potion.algorithms.pagepg.gpomdp_estimator",
        side_effect=[np.ones(n_params), 5. * np.ones(n_params)],
    )
    adaptive_step = mocker.Mock(return_value=np.zeros(n_params))

    pagepg(env, policy,
           batch_size=7,
           refresh_probability=1.,
           max_iterations=2,
           step_size=adaptive_step,
           logger=SilentLogger(),
           verbose=False)

    assert [call.args[2] for call in generate_batch.call_args_list] == [7, 7]
    assert [call.kwargs["reset"] for call in adaptive_step.call_args_list] == [
        True, True
    ]
    assert np.allclose(adaptive_step.call_args_list[0].args[0], np.ones(n_params))
    assert np.allclose(adaptive_step.call_args_list[1].args[0], 5. * np.ones(n_params))


@pytest.mark.parametrize("refresh_probability", [0., -0.1, 1.1])
def test_pagepg_rejects_invalid_refresh_probability(env, policy, refresh_probability):
    with pytest.raises(ValueError):
        pagepg(env, policy,
               refresh_probability=refresh_probability,
               logger=SilentLogger(),
               verbose=False)


def test_pagepg_unknown_estimator_defaults_to_gpomdp(env, policy, mocker):
    mocker.patch("potion.algorithms.pagepg.generate_batch", return_value=[None])
    estimator = mocker.patch(
        "potion.algorithms.pagepg.gpomdp_estimator",
        return_value=np.zeros(policy.num_params),
    )

    with pytest.warns(UserWarning):
        pagepg(env, policy,
               batch_size=1,
               max_iterations=1,
               estimator="xyz",
               logger=SilentLogger(),
               verbose=False)

    estimator.assert_called_once()


def test_pagepg_trajectory_budget_counts_refresh_batches(env, policy, n_params, mocker):
    generate_batch = mocker.patch(
        "potion.algorithms.pagepg.generate_batch",
        side_effect=lambda env, policy, n_episodes, horizon, **kwargs: [None] * n_episodes,
    )
    mocker.patch(
        "potion.algorithms.pagepg.gpomdp_estimator",
        return_value=np.zeros(n_params),
    )
    adaptive_step = mocker.Mock(return_value=np.zeros(n_params))

    pagepg(env, policy,
           batch_size=7,
           refresh_probability=1.,
           max_iterations=None,
           max_trajectories=10,
           step_size=adaptive_step,
           logger=SilentLogger(),
           verbose=False)

    assert [call.args[2] for call in generate_batch.call_args_list] == [7, 3]
    assert adaptive_step.call_count == 2


def test_pagepg_rejects_missing_stopping_criterion(env, policy):
    with pytest.raises(ValueError):
        pagepg(env, policy,
               max_iterations=None,
               max_trajectories=None,
               logger=SilentLogger(),
               verbose=False)


def test_pagepg_certain_refresh_matches_batch_gpomdp(env):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    reinforce_policy = LinearGaussianPolicy(state_dim, action_dim)
    page_policy = LinearGaussianPolicy(state_dim, action_dim)
    arguments = {
        "horizon": 3,
        "discount": 0.9,
        "step_size": 1e-3,
        "batch_size": 5,
        "max_iterations": None,
        "max_trajectories": 15,
        "estimator": "gpomdp",
        "baseline": "zero",
        "seed": 481,
        "n_jobs": 1,
        "verbose": False,
    }

    reinforce(env, reinforce_policy, logger=SilentLogger(), **arguments)
    pagepg(
        env,
        page_policy,
        refresh_probability=1.,
        logger=SilentLogger(),
        **arguments,
    )

    np.testing.assert_array_equal(page_policy.parameters, reinforce_policy.parameters)
