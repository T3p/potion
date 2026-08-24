import pytest

from potion.algorithms import reinforce
from potion.evaluation.loggers import SilentLogger
import numpy as np
from unittest.mock import MagicMock


def test_reinforce_estimator_call(env, policy, n_params, mocker):
    m = mocker.patch("potion.algorithms.reinforce.reinforce_estimator", return_value=np.ones(n_params))
    reinforce(env, policy, max_iterations=1, estimator="reinforce", logger=SilentLogger())
    m.assert_called()

    m = mocker.patch("potion.algorithms.reinforce.gpomdp_estimator", return_value=np.ones(n_params))
    reinforce(env, policy, max_iterations=1, estimator="gpomdp", logger=SilentLogger())
    m.assert_called()

    m = mocker.patch("potion.algorithms.reinforce.nonstationary_pg_estimator", return_value=np.ones(n_params))
    reinforce(env, policy, max_iterations=1, estimator="nonstationary", logger=SilentLogger())
    m.assert_called()

    with pytest.warns(UserWarning):
        reinforce(env, policy, max_iterations=1, estimator="xyz", logger=SilentLogger())


def test_reinforce_adaptive_step_call(env, policy, n_params):
    adaptive_step = MagicMock(return_value=np.ones(n_params))
    reinforce(env, policy, max_iterations=1, estimator="reinforce", step_size=adaptive_step, logger=SilentLogger())
    adaptive_step.assert_called()


def test_reinforce_trajectory_budget(env, policy, n_params, mocker):
    generate_batch = mocker.patch(
        "potion.algorithms.reinforce.generate_batch",
        side_effect=lambda env, policy, n_episodes, horizon, **kwargs: [None] * n_episodes,
    )
    estimator = mocker.patch(
        "potion.algorithms.reinforce.gpomdp_estimator",
        return_value=np.zeros(n_params),
    )

    reinforce(env, policy,
              batch_size=2,
              max_trajectories=5,
              logger=SilentLogger(),
              verbose=False)

    assert [call.args[2] for call in generate_batch.call_args_list] == [2, 2, 2]
    assert estimator.call_count == 3


def test_reinforce_rejects_missing_stopping_criterion(env, policy):
    with pytest.raises(ValueError):
        reinforce(env, policy,
                  max_iterations=None,
                  max_trajectories=None,
                  logger=SilentLogger(),
                  verbose=False)
