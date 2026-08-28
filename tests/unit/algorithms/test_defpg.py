import numpy as np
import pytest

from potion.algorithms import (
    def_pagepg,
    def_srvrpg,
    def_stormpg,
    def_svrpg,
    pagepg,
    srvrpg,
    stormpg,
    svrpg,
)
from potion.algorithms.defpg import _defensive_importance_weights
from potion.estimators.gradients import gpomdp_estimator
from potion.evaluation.loggers import SilentLogger
from potion.policies.gaussian_policies import LinearGaussianPolicy
from potion.simulation.trajectory_generators import generate_batch


def test_defensive_importance_weights():
    current_logps = np.array([-2., -1., 0.])
    snapshot_logps = np.array([-1., -1., -2.])

    current_weights, snapshot_weights = _defensive_importance_weights(
        current_logps, snapshot_logps, 0.5
    )

    mixture = 0.5 * np.exp(current_logps) + 0.5 * np.exp(snapshot_logps)
    assert np.allclose(current_weights, np.exp(current_logps) / mixture)
    assert np.allclose(snapshot_weights, np.exp(snapshot_logps) / mixture)
    assert np.all(current_weights <= 2.)
    assert np.all(snapshot_weights <= 2.)


def test_defensive_importance_weighted_monte_carlo_identities():
    rng = np.random.default_rng(991)
    n_samples = 200_000
    current_probability = 0.7
    snapshot_probability = 0.2
    defensive_parameter = 0.5
    use_snapshot = rng.random(n_samples) < defensive_parameter
    behavior_probability = np.where(
        use_snapshot, snapshot_probability, current_probability
    )
    actions = rng.random(n_samples) < behavior_probability

    current_logps = np.where(
        actions,
        np.log(current_probability),
        np.log1p(-current_probability),
    )
    snapshot_logps = np.where(
        actions,
        np.log(snapshot_probability),
        np.log1p(-snapshot_probability),
    )
    current_weights, snapshot_weights = _defensive_importance_weights(
        current_logps, snapshot_logps, defensive_parameter
    )
    current_gradient_samples = (actions - current_probability) * actions
    snapshot_gradient_samples = (actions - snapshot_probability) * actions

    assert np.mean(current_weights * current_gradient_samples) == pytest.approx(
        current_probability * (1. - current_probability), abs=0.003
    )
    assert np.mean(snapshot_weights * snapshot_gradient_samples) == pytest.approx(
        snapshot_probability * (1. - snapshot_probability), abs=0.003
    )


def test_recursive_correction_is_zero_when_policy_parameters_coincide(env):
    policy = LinearGaussianPolicy(
        env.observation_space.shape[0], env.action_space.shape[0]
    )
    batch = generate_batch(
        env, policy, n_episodes=7, max_trajectory_len=3,
        rng=np.random.default_rng(123)
    )

    current_samples = gpomdp_estimator(
        batch, 0.9, policy, baseline="average", average=False
    )
    previous_samples = gpomdp_estimator(
        batch,
        0.9,
        policy,
        baseline="average",
        average=False,
        off_policy=True,
    )

    assert np.allclose(current_samples - previous_samples, 0., atol=1e-14)


@pytest.mark.parametrize("defensive_parameter", [1., -0.1, 1.1])
def test_def_svrpg_rejects_invalid_defensive_parameter(env, policy, defensive_parameter):
    with pytest.raises(ValueError):
        def_svrpg(env, policy,
                  defensive_parameter=defensive_parameter,
                  logger=SilentLogger(),
                  verbose=False)


def test_def_svrpg_with_zero_defensive_parameter_matches_svrpg(env):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    svrpg_policy = LinearGaussianPolicy(state_dim, action_dim)
    def_svrpg_policy = LinearGaussianPolicy(state_dim, action_dim)
    arguments = {
        "horizon": 3,
        "discount": 0.9,
        "step_size": 1e-3,
        "batch_size": 5,
        "mini_batch_size": 3,
        "epoch_length": 2,
        "max_iterations": 2,
        "estimator": "gpomdp",
        "baseline": "zero",
        "seed": 123,
        "n_jobs": 1,
        "verbose": False,
    }

    svrpg(env, svrpg_policy, logger=SilentLogger(), **arguments)
    def_svrpg(
        env,
        def_svrpg_policy,
        defensive_parameter=0.,
        logger=SilentLogger(),
        **arguments,
    )

    np.testing.assert_array_equal(def_svrpg_policy.parameters, svrpg_policy.parameters)


@pytest.mark.parametrize(
    "algorithm, defensive_algorithm, algorithm_arguments",
    [
        (srvrpg, def_srvrpg, {"epoch_length": 3}),
        (stormpg, def_stormpg, {"momentum_parameter": 0.7}),
        (pagepg, def_pagepg, {"refresh_probability": 0.2}),
    ],
    ids=["srvrpg", "stormpg", "pagepg"],
)
def test_defensive_algorithm_with_zero_parameter_matches_regular_algorithm(
        env, algorithm, defensive_algorithm, algorithm_arguments):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    regular_policy = LinearGaussianPolicy(state_dim, action_dim)
    defensive_policy = LinearGaussianPolicy(state_dim, action_dim)
    arguments = {
        "horizon": 3,
        "discount": 0.9,
        "step_size": 1e-3,
        "batch_size": 5,
        "mini_batch_size": 3,
        "max_iterations": 3,
        "estimator": "gpomdp",
        "baseline": "zero",
        "seed": 123,
        "n_jobs": 1,
        "verbose": False,
        **algorithm_arguments,
    }

    algorithm(env, regular_policy, logger=SilentLogger(), **arguments)
    defensive_algorithm(
        env,
        defensive_policy,
        defensive_parameter=0.,
        logger=SilentLogger(),
        **arguments,
    )

    np.testing.assert_array_equal(defensive_policy.parameters, regular_policy.parameters)


def test_def_svrpg_gradient_correction(env, policy, n_params, mocker):
    generate_batch = mocker.patch(
        "potion.algorithms.defpg.generate_batch",
        return_value=[None] * 7,
    )
    defensive_batch = mocker.patch(
        "potion.algorithms.defpg._generate_defensive_batch",
        return_value=[None] * 2,
    )
    mocker.patch(
        "potion.algorithms.defpg._trajectory_log_probabilities",
        side_effect=[np.zeros(2), np.zeros(2)],
    )
    estimator = mocker.patch(
        "potion.algorithms.defpg.gpomdp_estimator",
        side_effect=[np.ones(n_params),
                     3. * np.ones((2, n_params)),
                     2. * np.ones((2, n_params))],
    )
    adaptive_step = mocker.Mock(return_value=np.zeros(n_params))

    def_svrpg(env, policy,
              batch_size=7,
              mini_batch_size=2,
              epoch_length=1,
              max_iterations=1,
              step_size=adaptive_step,
              logger=SilentLogger(),
              verbose=False)

    assert generate_batch.call_args.args[2] == 7
    assert defensive_batch.call_args.args[3] == 0.5
    assert estimator.call_count == 3
    assert adaptive_step.call_args.kwargs["reset"] is True
    assert np.allclose(adaptive_step.call_args.args[0], 2. * np.ones(n_params))


def test_def_svrpg_trajectory_budget_counts_all_training_batches(env, policy, n_params, mocker):
    generate_batch = mocker.patch(
        "potion.algorithms.defpg.generate_batch",
        side_effect=lambda env, policy, n_episodes, horizon, **kwargs: [None] * n_episodes,
    )
    defensive_batch = mocker.patch(
        "potion.algorithms.defpg._generate_defensive_batch",
        side_effect=lambda env, policy, snapshot_params, defensive_parameter,
        n_episodes, horizon, discount, rng, n_jobs: [None] * n_episodes,
    )
    mocker.patch(
        "potion.algorithms.defpg._trajectory_log_probabilities",
        side_effect=lambda batch, policy: np.zeros(len(batch)),
    )
    estimator = mocker.patch("potion.algorithms.defpg.gpomdp_estimator")
    estimator.side_effect = lambda batch, discount, policy, baseline, **kwargs: (
        np.zeros((len(batch), n_params))
        if kwargs.get("average") is False else np.zeros(n_params)
    )
    adaptive_step = mocker.Mock(return_value=np.zeros(n_params))

    def_svrpg(env, policy,
              batch_size=7,
              mini_batch_size=2,
              epoch_length=10,
              max_iterations=None,
              max_trajectories=10,
              step_size=adaptive_step,
              logger=SilentLogger(),
              verbose=False)

    assert generate_batch.call_args.args[2] == 7
    assert [call.args[4] for call in defensive_batch.call_args_list] == [2, 1]
    assert adaptive_step.call_count == 2


def test_def_svrpg_rejects_missing_stopping_criterion(env, policy):
    with pytest.raises(ValueError):
        def_svrpg(env, policy,
                  max_iterations=None,
                  max_trajectories=None,
                  logger=SilentLogger(),
                  verbose=False)


@pytest.mark.parametrize("defensive_parameter", [1., -0.1, 1.1])
def test_def_srvrpg_rejects_invalid_defensive_parameter(env, policy, defensive_parameter):
    with pytest.raises(ValueError):
        def_srvrpg(env, policy,
                   defensive_parameter=defensive_parameter,
                   logger=SilentLogger(),
                   verbose=False)


def test_def_srvrpg_recursive_gradient(env, policy, n_params, mocker):
    generate_batch = mocker.patch(
        "potion.algorithms.defpg.generate_batch",
        return_value=[None] * 7,
    )
    defensive_batch = mocker.patch(
        "potion.algorithms.defpg._generate_defensive_batch",
        return_value=[None] * 2,
    )
    mocker.patch(
        "potion.algorithms.defpg._trajectory_log_probabilities",
        side_effect=[np.zeros(2), np.zeros(2)],
    )
    estimator = mocker.patch(
        "potion.algorithms.defpg.gpomdp_estimator",
        side_effect=[np.ones(n_params),
                     3. * np.ones((2, n_params)),
                     2. * np.ones((2, n_params))],
    )
    adaptive_step = mocker.Mock(return_value=np.zeros(n_params))

    def_srvrpg(env, policy,
               batch_size=7,
               mini_batch_size=2,
               epoch_length=2,
               max_iterations=1,
               step_size=adaptive_step,
               logger=SilentLogger(),
               verbose=False)

    assert generate_batch.call_args.args[2] == 7
    assert defensive_batch.call_args.args[3] == 0.5
    assert estimator.call_count == 3
    assert [call.kwargs["reset"] for call in adaptive_step.call_args_list] == [
        True, False
    ]
    assert np.allclose(adaptive_step.call_args_list[0].args[0], np.ones(n_params))
    assert np.allclose(adaptive_step.call_args_list[1].args[0], 2. * np.ones(n_params))


def test_def_srvrpg_trajectory_budget_counts_all_training_batches(env, policy, n_params, mocker):
    generate_batch = mocker.patch(
        "potion.algorithms.defpg.generate_batch",
        side_effect=lambda env, policy, n_episodes, horizon, **kwargs: [None] * n_episodes,
    )
    defensive_batch = mocker.patch(
        "potion.algorithms.defpg._generate_defensive_batch",
        side_effect=lambda env, policy, previous_params, defensive_parameter,
        n_episodes, horizon, discount, rng, n_jobs: [None] * n_episodes,
    )
    mocker.patch(
        "potion.algorithms.defpg._trajectory_log_probabilities",
        side_effect=lambda batch, policy: np.zeros(len(batch)),
    )
    estimator = mocker.patch("potion.algorithms.defpg.gpomdp_estimator")
    estimator.side_effect = lambda batch, discount, policy, baseline, **kwargs: (
        np.zeros((len(batch), n_params))
        if kwargs.get("average") is False else np.zeros(n_params)
    )
    adaptive_step = mocker.Mock(return_value=np.zeros(n_params))

    def_srvrpg(env, policy,
               batch_size=7,
               mini_batch_size=2,
               epoch_length=10,
               max_iterations=None,
               max_trajectories=10,
               step_size=adaptive_step,
               logger=SilentLogger(),
               verbose=False)

    assert generate_batch.call_args.args[2] == 7
    assert [call.args[4] for call in defensive_batch.call_args_list] == [2, 1]
    assert adaptive_step.call_count == 3


def test_def_srvrpg_rejects_missing_stopping_criterion(env, policy):
    with pytest.raises(ValueError):
        def_srvrpg(env, policy,
                   max_iterations=None,
                   max_trajectories=None,
                   logger=SilentLogger(),
                   verbose=False)


@pytest.mark.parametrize("momentum_parameter", [0., 1., -0.1, 1.1])
def test_def_stormpg_rejects_invalid_momentum_parameter(env, policy, momentum_parameter):
    with pytest.raises(ValueError):
        def_stormpg(env, policy,
                    momentum_parameter=momentum_parameter,
                    logger=SilentLogger(),
                    verbose=False)


@pytest.mark.parametrize("defensive_parameter", [1., -0.1, 1.1])
def test_def_stormpg_rejects_invalid_defensive_parameter(env, policy, defensive_parameter):
    with pytest.raises(ValueError):
        def_stormpg(env, policy,
                    defensive_parameter=defensive_parameter,
                    logger=SilentLogger(),
                    verbose=False)


def test_def_stormpg_momentum_gradient(env, policy, n_params, mocker):
    generate_batch = mocker.patch(
        "potion.algorithms.defpg.generate_batch",
        return_value=[None] * 7,
    )
    defensive_batch = mocker.patch(
        "potion.algorithms.defpg._generate_defensive_batch",
        return_value=[None] * 2,
    )
    mocker.patch(
        "potion.algorithms.defpg._trajectory_log_probabilities",
        side_effect=[np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2)],
    )
    estimator = mocker.patch(
        "potion.algorithms.defpg.gpomdp_estimator",
        side_effect=[np.ones(n_params),
                     3. * np.ones((2, n_params)),
                     2. * np.ones((2, n_params)),
                     3. * np.ones((2, n_params)),
                     2. * np.ones((2, n_params))],
    )
    adaptive_step = mocker.Mock(return_value=np.zeros(n_params))

    def_stormpg(env, policy,
                batch_size=7,
                mini_batch_size=2,
                momentum_parameter=0.5,
                max_iterations=3,
                step_size=adaptive_step,
                logger=SilentLogger(),
                verbose=False)

    assert generate_batch.call_args.args[2] == 7
    assert [call.args[3] for call in defensive_batch.call_args_list] == [0.5, 0.5]
    assert all(not call.kwargs for call in adaptive_step.call_args_list)
    expected_gradients = [1., 2.5, 3.25]
    for call, expected in zip(adaptive_step.call_args_list, expected_gradients):
        assert np.allclose(call.args[0], expected * np.ones(n_params))


def test_def_stormpg_trajectory_budget_counts_all_training_batches(env, policy, n_params, mocker):
    generate_batch = mocker.patch(
        "potion.algorithms.defpg.generate_batch",
        side_effect=lambda env, policy, n_episodes, horizon, **kwargs: [None] * n_episodes,
    )
    defensive_batch = mocker.patch(
        "potion.algorithms.defpg._generate_defensive_batch",
        side_effect=lambda env, policy, previous_params, defensive_parameter,
        n_episodes, horizon, discount, rng, n_jobs: [None] * n_episodes,
    )
    mocker.patch(
        "potion.algorithms.defpg._trajectory_log_probabilities",
        side_effect=lambda batch, policy: np.zeros(len(batch)),
    )
    estimator = mocker.patch("potion.algorithms.defpg.gpomdp_estimator")
    estimator.side_effect = lambda batch, discount, policy, baseline, **kwargs: (
        np.zeros((len(batch), n_params))
        if kwargs.get("average") is False else np.zeros(n_params)
    )
    adaptive_step = mocker.Mock(return_value=np.zeros(n_params))

    def_stormpg(env, policy,
                batch_size=7,
                mini_batch_size=2,
                max_iterations=None,
                max_trajectories=10,
                step_size=adaptive_step,
                logger=SilentLogger(),
                verbose=False)

    assert generate_batch.call_args.args[2] == 7
    assert [call.args[4] for call in defensive_batch.call_args_list] == [2, 1]
    assert adaptive_step.call_count == 3


def test_def_stormpg_rejects_missing_stopping_criterion(env, policy):
    with pytest.raises(ValueError):
        def_stormpg(env, policy,
                    max_iterations=None,
                    max_trajectories=None,
                    logger=SilentLogger(),
                    verbose=False)


@pytest.mark.parametrize("refresh_probability", [0., -0.1, 1.1])
def test_def_pagepg_rejects_invalid_refresh_probability(env, policy, refresh_probability):
    with pytest.raises(ValueError):
        def_pagepg(env, policy,
                   refresh_probability=refresh_probability,
                   logger=SilentLogger(),
                   verbose=False)


@pytest.mark.parametrize("defensive_parameter", [1., -0.1, 1.1])
def test_def_pagepg_rejects_invalid_defensive_parameter(env, policy, defensive_parameter):
    with pytest.raises(ValueError):
        def_pagepg(env, policy,
                   defensive_parameter=defensive_parameter,
                   logger=SilentLogger(),
                   verbose=False)


def test_def_pagepg_recursive_gradient_and_default_probability(env, policy, n_params, mocker):
    rng = mocker.Mock()
    rng.random.return_value = 0.9
    mocker.patch(
        "potion.algorithms.defpg.initialize_run",
        side_effect=lambda seed, logger: (rng, mocker.Mock(), logger),
    )
    generate_batch = mocker.patch(
        "potion.algorithms.defpg.generate_batch",
        return_value=[None] * 7,
    )
    defensive_batch = mocker.patch(
        "potion.algorithms.defpg._generate_defensive_batch",
        return_value=[None] * 2,
    )
    mocker.patch(
        "potion.algorithms.defpg._trajectory_log_probabilities",
        side_effect=[np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2)],
    )
    estimator = mocker.patch(
        "potion.algorithms.defpg.gpomdp_estimator",
        side_effect=[np.ones(n_params),
                     3. * np.ones((2, n_params)),
                     2. * np.ones((2, n_params)),
                     3. * np.ones((2, n_params)),
                     2. * np.ones((2, n_params))],
    )
    adaptive_step = mocker.Mock(return_value=np.zeros(n_params))

    def_pagepg(env, policy,
               batch_size=7,
               mini_batch_size=2,
               max_iterations=3,
               step_size=adaptive_step,
               logger=SilentLogger(),
               verbose=False)

    assert generate_batch.call_args.args[2] == 7
    assert [call.args[3] for call in defensive_batch.call_args_list] == [0.5, 0.5]
    assert [call.kwargs["reset"] for call in adaptive_step.call_args_list] == [
        True, False, False
    ]
    expected_gradients = [1., 2., 3.]
    for call, expected in zip(adaptive_step.call_args_list, expected_gradients):
        assert np.allclose(call.args[0], expected * np.ones(n_params))
    assert rng.random.call_count == 2


def test_def_pagepg_large_batch_refresh(env, policy, n_params, mocker):
    generate_batch = mocker.patch(
        "potion.algorithms.defpg.generate_batch",
        side_effect=lambda env, policy, n_episodes, horizon, **kwargs: [None] * n_episodes,
    )
    estimator = mocker.patch(
        "potion.algorithms.defpg.gpomdp_estimator",
        side_effect=[np.ones(n_params), 5. * np.ones(n_params)],
    )
    adaptive_step = mocker.Mock(return_value=np.zeros(n_params))

    def_pagepg(env, policy,
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


def test_def_pagepg_trajectory_budget_counts_refresh_batches(env, policy, n_params, mocker):
    generate_batch = mocker.patch(
        "potion.algorithms.defpg.generate_batch",
        side_effect=lambda env, policy, n_episodes, horizon, **kwargs: [None] * n_episodes,
    )
    mocker.patch(
        "potion.algorithms.defpg.gpomdp_estimator",
        return_value=np.zeros(n_params),
    )
    adaptive_step = mocker.Mock(return_value=np.zeros(n_params))

    def_pagepg(env, policy,
               batch_size=7,
               refresh_probability=1.,
               max_iterations=None,
               max_trajectories=10,
               step_size=adaptive_step,
               logger=SilentLogger(),
               verbose=False)

    assert [call.args[2] for call in generate_batch.call_args_list] == [7, 3]
    assert adaptive_step.call_count == 2


def test_def_pagepg_rejects_missing_stopping_criterion(env, policy):
    with pytest.raises(ValueError):
        def_pagepg(env, policy,
                   max_iterations=None,
                   max_trajectories=None,
                   logger=SilentLogger(),
                   verbose=False)
