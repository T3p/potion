import pytest

from potion.algorithms import reinforce
from potion.evaluation.loggers import SilentLogger
import numpy as np


def test_reinforce(env, policy, n_params, mocker):
    m = mocker.patch("potion.learning.policy_gradient.reinforce_estimator", return_value=np.ones(n_params))
    reinforce(env, policy, max_iterations=1, estimator="reinforce", logger=SilentLogger())
    m.assert_called()

    m = mocker.patch("potion.learning.policy_gradient.gpomdp_estimator", return_value=np.ones(n_params))
    reinforce(env, policy, max_iterations=1, estimator="gpomdp", logger=SilentLogger())
    m.assert_called()

    with pytest.warns(UserWarning):
        reinforce(env, policy, max_iterations=1, estimator="xyz", logger=SilentLogger())

