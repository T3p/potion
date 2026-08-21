import pytest

from potion.algorithms import reinforce
from potion.evaluation.loggers import SilentLogger
import numpy as np
from unittest.mock import MagicMock


def test_reinforce_estimator_call(env, policy, n_params, mocker):
    m = mocker.patch("potion.algorithms.policy_gradient.reinforce_estimator", return_value=np.ones(n_params))
    reinforce(env, policy, max_iterations=1, estimator="reinforce", logger=SilentLogger())
    m.assert_called()

    m = mocker.patch("potion.algorithms.policy_gradient.gpomdp_estimator", return_value=np.ones(n_params))
    reinforce(env, policy, max_iterations=1, estimator="gpomdp", logger=SilentLogger())
    m.assert_called()

    m = mocker.patch("potion.algorithms.policy_gradient.nonstationary_pg_estimator", return_value=np.ones(n_params))
    reinforce(env, policy, max_iterations=1, estimator="nonstationary", logger=SilentLogger())
    m.assert_called()

    with pytest.warns(UserWarning):
        reinforce(env, policy, max_iterations=1, estimator="xyz", logger=SilentLogger())


def test_reinforce_adaptive_step_call(env, policy, n_params):
    adaptive_step = MagicMock(return_value=np.ones(n_params))
    reinforce(env, policy, max_iterations=1, estimator="reinforce", step_size=adaptive_step, logger=SilentLogger())
    adaptive_step.assert_called()

