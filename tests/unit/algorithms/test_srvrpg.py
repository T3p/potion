from importlib import import_module

import numpy as np
import pytest

from potion.algorithms import srvrpg
from potion.evaluation.loggers import SilentLogger

srvrpg_module = import_module("potion.algorithms.srvrpg")


def test_srvrpg_batch_schedule_and_recursive_gradient(env, policy, n_params, mocker, capsys):
    generate_batch = mocker.patch.object(
        srvrpg_module,
        "generate_batch",
        side_effect=lambda env, policy, n_episodes, horizon, **kwargs: [None] * n_episodes,
    )
    estimator = mocker.patch.object(srvrpg_module, "gpomdp_estimator")
    estimator.side_effect = lambda batch, discount, policy, baseline, **kwargs: np.full(
        n_params, 2. if kwargs.get("off_policy") else 3.
    )
    adaptive_step = mocker.Mock(return_value=np.zeros(n_params))

    srvrpg(env, policy,
           batch_size=7,
           mini_batch_size=2,
           epoch_length=2,
           max_iterations=2,
           step_size=adaptive_step,
           logger=SilentLogger(),
           verbose=True)

    assert [call.args[2] for call in generate_batch.call_args_list] == [7, 2, 7, 2]
    assert adaptive_step.call_count == 4
    assert [call.kwargs["reset"] for call in adaptive_step.call_args_list] == [
        True, False, True, False
    ]
    expected_gradients = [3., 4., 3., 4.]
    for call, expected in zip(adaptive_step.call_args_list, expected_gradients):
        assert np.allclose(call.args[0], expected * np.ones(n_params))

    output = capsys.readouterr().out
    assert output.count("Iteration 1 of 2 running...") == 1
    assert output.count("Iteration 2 of 2 running...") == 1
    assert output.count("Epoch 2 of 2 running...") == 2


def test_srvrpg_unknown_estimator_defaults_to_gpomdp(env, policy, mocker):
    mocker.patch.object(srvrpg_module, "generate_batch", return_value=[None])
    estimator = mocker.patch.object(
        srvrpg_module,
        "gpomdp_estimator",
        return_value=np.zeros(policy.num_params),
    )

    with pytest.warns(UserWarning):
        srvrpg(env, policy,
               batch_size=1,
               epoch_length=1,
               max_iterations=1,
               estimator="xyz",
               logger=SilentLogger(),
               verbose=False)

    estimator.assert_called_once()


def test_srvrpg_trajectory_budget_counts_all_training_batches(env, policy, n_params, mocker):
    generate_batch = mocker.patch.object(
        srvrpg_module,
        "generate_batch",
        side_effect=lambda env, policy, n_episodes, horizon, **kwargs: [None] * n_episodes,
    )
    estimator = mocker.patch.object(srvrpg_module, "gpomdp_estimator")
    estimator.return_value = np.zeros(n_params)
    adaptive_step = mocker.Mock(return_value=np.zeros(n_params))

    srvrpg(env, policy,
           batch_size=7,
           mini_batch_size=2,
           epoch_length=10,
           max_iterations=None,
           max_trajectories=10,
           step_size=adaptive_step,
           logger=SilentLogger(),
           verbose=False)

    assert [call.args[2] for call in generate_batch.call_args_list] == [7, 2, 1]
    assert adaptive_step.call_count == 3


def test_srvrpg_rejects_missing_stopping_criterion(env, policy):
    with pytest.raises(ValueError):
        srvrpg(env, policy,
               max_iterations=None,
               max_trajectories=None,
               logger=SilentLogger(),
               verbose=False)
