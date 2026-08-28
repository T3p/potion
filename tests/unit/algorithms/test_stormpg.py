import numpy as np
import pytest

from potion.algorithms import stormpg
from potion.evaluation.loggers import SilentLogger


def test_stormpg_batch_schedule_and_momentum_gradient(env, policy, n_params, mocker, capsys):
    generate_batch = mocker.patch(
        "potion.algorithms.stormpg.generate_batch",
        side_effect=lambda env, policy, n_episodes, horizon, **kwargs: [None] * n_episodes,
    )
    estimator = mocker.patch("potion.algorithms.stormpg.gpomdp_estimator")
    estimator.side_effect = lambda batch, discount, policy, baseline, **kwargs: np.full(
        n_params, 2. if kwargs.get("off_policy") else 3.
    )
    adaptive_step = mocker.Mock(return_value=np.zeros(n_params))

    stormpg(env, policy,
            batch_size=7,
            mini_batch_size=2,
            momentum_parameter=0.5,
            max_iterations=3,
            step_size=adaptive_step,
            logger=SilentLogger(),
            verbose=True)

    assert [call.args[2] for call in generate_batch.call_args_list] == [7, 2, 2]
    assert all(not call.kwargs for call in adaptive_step.call_args_list)
    expected_gradients = [3., 3.5, 3.75]
    for call, expected in zip(adaptive_step.call_args_list, expected_gradients):
        assert np.allclose(call.args[0], expected * np.ones(n_params))

    output = capsys.readouterr().out
    assert output.count("Iteration 1 of 3 running...") == 1
    assert output.count("Iteration 2 of 3 running...") == 1
    assert output.count("Iteration 3 of 3 running...") == 1


@pytest.mark.parametrize("momentum_parameter", [0., 1., -0.1, 1.1])
def test_stormpg_rejects_invalid_momentum_parameter(env, policy, momentum_parameter):
    with pytest.raises(ValueError):
        stormpg(env, policy,
                momentum_parameter=momentum_parameter,
                logger=SilentLogger(),
                verbose=False)


def test_stormpg_unknown_estimator_defaults_to_gpomdp(env, policy, mocker):
    mocker.patch("potion.algorithms.stormpg.generate_batch", return_value=[None])
    estimator = mocker.patch(
        "potion.algorithms.stormpg.gpomdp_estimator",
        return_value=np.zeros(policy.num_params),
    )

    with pytest.warns(UserWarning):
        stormpg(env, policy,
                batch_size=1,
                max_iterations=1,
                estimator="xyz",
                logger=SilentLogger(),
                verbose=False)

    estimator.assert_called_once()


def test_stormpg_trajectory_budget_counts_all_training_batches(env, policy, n_params, mocker):
    generate_batch = mocker.patch(
        "potion.algorithms.stormpg.generate_batch",
        side_effect=lambda env, policy, n_episodes, horizon, **kwargs: [None] * n_episodes,
    )
    estimator = mocker.patch("potion.algorithms.stormpg.gpomdp_estimator")
    estimator.return_value = np.zeros(n_params)
    adaptive_step = mocker.Mock(return_value=np.zeros(n_params))

    stormpg(env, policy,
            batch_size=7,
            mini_batch_size=2,
            max_iterations=None,
            max_trajectories=10,
            step_size=adaptive_step,
            logger=SilentLogger(),
            verbose=False)

    assert [call.args[2] for call in generate_batch.call_args_list] == [7, 2, 1]
    assert adaptive_step.call_count == 3


def test_stormpg_rejects_missing_stopping_criterion(env, policy):
    with pytest.raises(ValueError):
        stormpg(env, policy,
                max_iterations=None,
                max_trajectories=None,
                logger=SilentLogger(),
                verbose=False)


def test_stormpg_uses_one_minus_momentum_for_previous_estimator(
        env, policy, n_params, mocker):
    mocker.patch(
        "potion.algorithms.stormpg.generate_batch",
        side_effect=lambda env, policy, n_episodes, horizon, **kwargs: [None] * n_episodes,
    )
    estimator = mocker.patch("potion.algorithms.stormpg.gpomdp_estimator")
    estimator.side_effect = [
        np.zeros(n_params),
        np.zeros(n_params),
        np.ones(n_params),
    ]
    adaptive_step = mocker.Mock(return_value=np.zeros(n_params))

    stormpg(
        env,
        policy,
        batch_size=7,
        mini_batch_size=2,
        momentum_parameter=0.99,
        max_iterations=2,
        step_size=adaptive_step,
        logger=SilentLogger(),
        verbose=False,
    )

    assert np.allclose(
        adaptive_step.call_args_list[1].args[0], -0.01 * np.ones(n_params)
    )
