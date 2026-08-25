import numpy as np
import pytest

from potion.algorithms import svrpg
from potion.evaluation.loggers import SilentLogger


def test_svrpg_batch_schedule_and_gradient_correction(env, policy, n_params, mocker, capsys):
    generate_batch = mocker.patch(
        "potion.algorithms.svrpg.generate_batch",
        side_effect=lambda env, policy, n_episodes, horizon, **kwargs: [None] * n_episodes,
    )
    logger = SilentLogger()
    estimator = mocker.patch("potion.algorithms.svrpg.gpomdp_estimator")
    estimator.side_effect = lambda batch, discount, policy, baseline, **kwargs: (
        np.full((2, n_params), 2. if kwargs.get("off_policy") else 3.)
        if kwargs.get("average") is False else np.ones(n_params)
    )
    adaptive_step = mocker.Mock(return_value=np.zeros(n_params))

    svrpg(env, policy,
          batch_size=7,
          mini_batch_size=2,
          epoch_length=2,
          max_iterations=2,
          step_size=adaptive_step,
          logger=logger,
          verbose=True)

    assert [call.args[2] for call in generate_batch.call_args_list] == [7, 2, 2, 7, 2, 2]
    assert adaptive_step.call_count == 4
    for call in adaptive_step.call_args_list:
        assert np.allclose(call.args[0], 2. * np.ones(n_params))

    output = capsys.readouterr().out
    assert output.count("Iteration 1 of 2 running...") == 1
    assert output.count("Iteration 2 of 2 running...") == 1
    assert output.count("Epoch 1 of 2 running...") == 2
    assert output.count("Epoch 2 of 2 running...") == 2


def test_svrpg_unknown_estimator_defaults_to_gpomdp(env, policy, mocker):
    mocker.patch("potion.algorithms.svrpg.generate_batch", return_value=[None])
    estimator = mocker.patch(
        "potion.algorithms.svrpg.gpomdp_estimator",
        side_effect=[np.zeros(policy.num_params),
                     np.zeros((1, policy.num_params)),
                     np.zeros((1, policy.num_params))],
    )

    with pytest.warns(UserWarning):
        svrpg(env, policy,
              batch_size=1,
              mini_batch_size=1,
              epoch_length=1,
              max_iterations=1,
              estimator="xyz",
              logger=SilentLogger(),
              verbose=False)

    assert estimator.call_count == 3


def test_svrpg_trajectory_budget_counts_all_training_batches(env, policy, n_params, mocker):
    generate_batch = mocker.patch(
        "potion.algorithms.svrpg.generate_batch",
        side_effect=lambda env, policy, n_episodes, horizon, **kwargs: [None] * n_episodes,
    )
    estimator = mocker.patch("potion.algorithms.svrpg.gpomdp_estimator")
    estimator.side_effect = lambda batch, discount, policy, baseline, **kwargs: (
        np.zeros((len(batch), n_params))
        if kwargs.get("average") is False else np.zeros(n_params)
    )
    adaptive_step = mocker.Mock(return_value=np.zeros(n_params))

    svrpg(env, policy,
          batch_size=7,
          mini_batch_size=2,
          epoch_length=10,
          max_iterations=None,
          max_trajectories=10,
          step_size=adaptive_step,
          logger=SilentLogger(),
          verbose=False)

    assert [call.args[2] for call in generate_batch.call_args_list] == [7, 2, 1]
    assert adaptive_step.call_count == 2


def test_svrpg_rejects_missing_stopping_criterion(env, policy):
    with pytest.raises(ValueError):
        svrpg(env, policy,
              max_iterations=None,
              max_trajectories=None,
              logger=SilentLogger(),
              verbose=False)
