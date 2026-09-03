from importlib import import_module

import numpy as np
import pytest

from potion.algorithms import npg_igt
from potion.evaluation.loggers import SilentLogger


npg_igt_module = import_module("potion.algorithms.npg_igt")


class StatefulPolicy:
    def __init__(self, params):
        self._params = np.asarray(params, dtype=float)

    @property
    def parameters(self):
        return self._params.copy()

    def set_params(self, params):
        self._params = np.asarray(params, dtype=float).copy()


def test_npg_igt_uses_lookahead_and_normalized_momentum(env, mocker):
    policy = StatefulPolicy([0., 0.])
    sampled_params = []
    scored_params = []
    gradients = iter((np.array([3., 4.]), np.array([0., 2.])))

    def generate(env, policy, n_episodes, horizon, **kwargs):
        sampled_params.append(policy.parameters)
        return [None] * n_episodes

    def estimate(batch, discount, policy, baseline):
        scored_params.append(policy.parameters)
        return next(gradients)

    mocker.patch.object(npg_igt_module, "generate_batch", side_effect=generate)
    mocker.patch.object(npg_igt_module, "gpomdp_estimator", side_effect=estimate)

    npg_igt(
        env,
        policy,
        step_size=2.,
        batch_size=1,
        momentum_parameter=0.5,
        max_iterations=2,
        logger=SilentLogger(),
        verbose=False,
    )

    theta_1 = np.array([1.2, 1.6])
    expected_lookaheads = [np.zeros(2), 2. * theta_1]
    np.testing.assert_allclose(sampled_params, expected_lookaheads)
    np.testing.assert_allclose(scored_params, expected_lookaheads)

    second_direction = np.array([1.5, 3.])
    expected_params = theta_1 + 2. * second_direction / np.linalg.norm(
        second_direction
    )
    np.testing.assert_allclose(policy.parameters, expected_params)


def test_npg_igt_honors_trajectory_budget(env, mocker):
    policy = StatefulPolicy([0., 0.])
    generate = mocker.patch.object(
        npg_igt_module,
        "generate_batch",
        side_effect=lambda env, policy, n_episodes, horizon, **kwargs: (
            [None] * n_episodes
        ),
    )
    mocker.patch.object(
        npg_igt_module, "gpomdp_estimator", return_value=np.zeros(2)
    )

    npg_igt(
        env,
        policy,
        batch_size=2,
        max_iterations=None,
        max_trajectories=5,
        logger=SilentLogger(),
        verbose=False,
    )

    assert [call.args[2] for call in generate.call_args_list] == [2, 2, 1]


@pytest.mark.parametrize("momentum_parameter", [0., -0.1, 1.1])
def test_npg_igt_rejects_invalid_momentum(
        env, momentum_parameter):
    with pytest.raises(ValueError):
        npg_igt(
            env,
            StatefulPolicy([0., 0.]),
            momentum_parameter=momentum_parameter,
            max_iterations=1,
            logger=SilentLogger(),
            verbose=False,
        )
