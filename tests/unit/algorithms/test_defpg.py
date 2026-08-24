import numpy as np
import pytest

from potion.algorithms import def_svrpg
from potion.algorithms.defpg import _defensive_importance_weights
from potion.evaluation.loggers import SilentLogger


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


@pytest.mark.parametrize("defensive_parameter", [0., 1., -0.1, 1.1])
def test_def_svrpg_rejects_invalid_defensive_parameter(env, policy, defensive_parameter):
    with pytest.raises(ValueError):
        def_svrpg(env, policy,
                  defensive_parameter=defensive_parameter,
                  logger=SilentLogger(),
                  verbose=False)


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
    assert [call.args[4] for call in defensive_batch.call_args_list] == [2, 2]
    assert adaptive_step.call_count == 2


def test_def_svrpg_rejects_missing_stopping_criterion(env, policy):
    with pytest.raises(ValueError):
        def_svrpg(env, policy,
                  max_iterations=None,
                  max_trajectories=None,
                  logger=SilentLogger(),
                  verbose=False)
