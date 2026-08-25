import numpy as np
import pytest
import gymnasium as gym

from potion.evaluation import loggers
from potion.evaluation.loggers import EpisodicOnlineLogger, EpisodicTestLogger


def _trajectory(rewards, alive=None):
    rewards = np.asarray(rewards, dtype=float)
    if alive is None:
        alive = np.ones_like(rewards, dtype=bool)
    return None, None, rewards, np.asarray(alive, dtype=bool), None


def test_online_logger_records_only_submitted_trajectory_returns(rng):
    logger = EpisodicOnlineLogger(log_every=1, verbose=False, path=None)

    logger.initialize(object(), object(), horizon=10, discount=0.5, rng=rng)
    logger.submit(
        [_trajectory([1., 2.]), _trajectory([3., 4.], alive=[True, False])],
        object(),
    )

    assert logger.buffer == [
        {"tot_trajectories": 1, "return": 2., "normalized_auc": 2.},
        {"tot_trajectories": 2, "return": 3., "normalized_auc": 2.5},
    ]


def test_episodic_logger_prints_cyan_by_default(capsys, rng):
    logger = EpisodicOnlineLogger(verbose=True, path=None)

    logger.initialize(object(), object(), horizon=10, discount=1., rng=rng)

    output = capsys.readouterr().out
    assert "\033[36m>> Episodic Online Logger ***\033[0m" in output


def test_episodic_logger_color_can_be_disabled(capsys, rng):
    logger = EpisodicOnlineLogger(verbose=True, path=None, color=None)

    logger.initialize(object(), object(), horizon=10, discount=1., rng=rng)

    output = capsys.readouterr().out
    assert ">> Episodic Online Logger ***" in output
    assert "\033[" not in output


def test_episodic_logger_rejects_unknown_color():
    with pytest.raises(ValueError, match="color"):
        EpisodicTestLogger(color="ultraviolet")


def test_online_logger_always_records_first_submitted_trajectory(rng):
    logger = EpisodicOnlineLogger(log_every=10, verbose=False, path=None)

    logger.initialize(object(), object(), horizon=10, discount=0.5, rng=rng)
    logger.submit([_trajectory([2., 4.]), _trajectory([100.])], object())

    assert logger.buffer == [
        {"tot_trajectories": 1, "return": 4., "normalized_auc": 4.},
    ]


def test_online_logger_cadence_is_anchored_at_first_trajectory(rng):
    logger = EpisodicOnlineLogger(log_every=10, verbose=False, path=None)
    trajectories = [_trajectory([float(i)]) for i in range(1, 23)]

    logger.initialize(object(), object(), horizon=10, discount=1., rng=rng)
    logger.submit(trajectories, object())

    assert logger.buffer == [
        {"tot_trajectories": 1, "return": 1., "normalized_auc": 1.},
        {"tot_trajectories": 11, "return": 11., "normalized_auc": 6.},
        {"tot_trajectories": 21, "return": 21., "normalized_auc": 11.},
    ]
    assert logger.normalized_auc == 11.5


def test_online_logger_auc_uses_unlogged_trajectories(rng):
    logger = EpisodicOnlineLogger(log_every=3, verbose=False, path=None)
    trajectories = [
        _trajectory([0.]),
        _trajectory([100.]),
        _trajectory([0.]),
        _trajectory([0.]),
    ]

    logger.initialize(object(), object(), horizon=10, discount=1., rng=rng)
    logger.submit(trajectories, object())

    assert len(logger.buffer) == 2
    assert logger.buffer[0] == {
        "tot_trajectories": 1,
        "return": 0.,
        "normalized_auc": 0.,
    }
    assert logger.buffer[1]["tot_trajectories"] == 4
    assert logger.buffer[1]["return"] == 0.
    assert np.isclose(logger.buffer[1]["normalized_auc"], 100. / 3.)


def test_test_logger_evaluates_initial_and_periodic_policies(mocker, rng):
    evaluate = mocker.patch(
        "potion.evaluation.loggers.estimate_average_return",
        side_effect=[12.5, 18.25],
    )
    env = object()
    initial_policy = object()
    submitted_policy = object()
    logger = EpisodicTestLogger(
        log_every=2,
        verbose=False,
        n_test=3,
        path=None,
    )

    logger.initialize(env, initial_policy, horizon=10, discount=0.5, rng=rng)
    logger.submit([_trajectory([1.]), _trajectory([2.]), _trajectory([3.])], submitted_policy)

    assert logger.buffer == [
        {"tot_trajectories": 0, "return": 12.5, "normalized_auc": 12.5},
        {"tot_trajectories": 2, "return": 18.25, "normalized_auc": 15.375},
    ]
    assert logger.normalized_auc == 15.375
    assert evaluate.call_count == 2
    assert evaluate.call_args_list[0].args == (env, initial_policy, 3, 10, rng)
    assert evaluate.call_args_list[1].args == (env, submitted_policy, 3, 10, rng)
    assert evaluate.call_args_list[0].kwargs == {"discount": 0.5}
    assert evaluate.call_args_list[1].kwargs == {"discount": 0.5}


def test_test_logger_skips_evaluation_between_intervals(mocker, rng):
    evaluate = mocker.patch(
        "potion.evaluation.loggers.estimate_average_return",
        return_value=4.,
    )
    logger = EpisodicTestLogger(log_every=2, verbose=False, path=None)
    logger.initialize(object(), object(), horizon=10, discount=1., rng=rng)
    evaluate.reset_mock()

    logger.submit([_trajectory([1.])], object())

    evaluate.assert_not_called()
    assert logger.buffer == [
        {"tot_trajectories": 0, "return": 4., "normalized_auc": 4.},
    ]


def test_test_logger_logs_each_snapshot_when_submission_crosses_multiple_intervals(mocker, rng):
    evaluate = mocker.patch(
        "potion.evaluation.loggers.estimate_average_return",
        return_value=4.,
    )
    logger = EpisodicTestLogger(log_every=1, verbose=False, path=None)
    logger.initialize(object(), object(), horizon=10, discount=1., rng=rng)
    evaluate.reset_mock()

    logger.submit([_trajectory([1.]), _trajectory([2.]), _trajectory([3.])], object())

    assert evaluate.call_count == 3
    assert logger.buffer[1:] == [
        {"tot_trajectories": 1, "return": 4., "normalized_auc": 4.},
        {"tot_trajectories": 2, "return": 4., "normalized_auc": 4.},
        {"tot_trajectories": 3, "return": 4., "normalized_auc": 4.},
    ]


def test_test_logger_retains_records_after_close(mocker, rng):
    mocker.patch(
        "potion.evaluation.loggers.estimate_average_return",
        side_effect=[3., 5.],
    )
    logger = EpisodicTestLogger(
        log_every=2,
        verbose=False,
        path=None,
        keep_records=True,
    )
    logger.initialize(object(), object(), horizon=10, discount=1., rng=rng)
    logger.submit([_trajectory([1.]), _trajectory([2.])], object())

    logger.close()

    assert logger.buffer == []
    assert logger.records == [
        {"tot_trajectories": 0, "return": 3., "normalized_auc": 3.},
        {"tot_trajectories": 2, "return": 5., "normalized_auc": 4.},
    ]


def test_test_logger_n_test_zero_disables_evaluation(mocker, rng):
    evaluate = mocker.patch("potion.evaluation.loggers.estimate_average_return")
    logger = EpisodicTestLogger(verbose=False, n_test=0, path=None)

    logger.initialize(object(), object(), horizon=10, discount=0.5, rng=rng)
    logger.submit([_trajectory([2., 4.])], object())

    evaluate.assert_not_called()
    assert logger.buffer == []
    assert logger.tot_traj == 1


def test_logger_initialize_resets_run_specific_state(rng):
    logger = EpisodicOnlineLogger(log_every=1, verbose=False, path=None)
    logger.initialize(object(), object(), horizon=2, discount=1., rng=rng)
    logger.submit([_trajectory([2.]), _trajectory([4.])], object())
    assert logger.tot_traj == 2
    assert logger.normalized_auc == 3.

    logger.initialize(object(), object(), horizon=2, discount=1., rng=rng)
    logger.submit([_trajectory([10.])], object())

    assert logger.tot_traj == 1
    assert logger.normalized_auc == 10.
    assert logger.buffer == [
        {"tot_trajectories": 1, "return": 10., "normalized_auc": 10.}
    ]


def test_test_logger_can_report_undiscounted_return(rng):
    horizon = 4

    class UnitRewardEnv(gym.Env):
        observation_space = gym.spaces.Box(-1., 1., shape=(1,), dtype=float)
        action_space = gym.spaces.Box(-1., 1., shape=(1,), dtype=float)

        def reset(self, seed=None, options=None):
            self.t = 0
            return np.zeros(1), {}

        def step(self, action):
            self.t += 1
            return np.zeros(1), 1., False, self.t == horizon, {}

    class ConstantPolicy:
        def act(self, state, rng, t=None):
            return np.zeros(1)

    logger = EpisodicTestLogger(
        n_test=1,
        override_discount=1.0,
        verbose=False,
        path=None,
    )
    logger.initialize(
        UnitRewardEnv(), ConstantPolicy(), horizon=horizon, discount=0.25, rng=rng
    )

    assert logger.buffer[0]["return"] == horizon


@pytest.mark.parametrize("n_test", [-1, 1.5])
def test_test_logger_rejects_invalid_n_test(n_test):
    with pytest.raises(ValueError, match="n_test"):
        EpisodicTestLogger(n_test=n_test)


def test_episodic_performance_logger_no_longer_exists():
    assert not hasattr(loggers, "EpisodicPerformanceLogger")
