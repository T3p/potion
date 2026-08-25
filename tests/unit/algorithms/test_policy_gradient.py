import pytest

from potion.algorithms import reinforce
from potion.evaluation.loggers import EpisodicTestLogger, SilentLogger
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

    assert [call.args[2] for call in generate_batch.call_args_list] == [2, 2, 1]
    assert estimator.call_count == 3


def test_reinforce_rejects_missing_stopping_criterion(env, policy):
    with pytest.raises(ValueError):
        reinforce(env, policy,
                  max_iterations=None,
                  max_trajectories=None,
                  logger=SilentLogger(),
                  verbose=False)


def test_evaluation_schedule_does_not_change_training_randomness(
        env, policy, n_params, mocker):
    training_draws = []

    def generate(env, policy, n_episodes, horizon, rng, **kwargs):
        training_draws.extend(rng.integers(0, 2**32, size=n_episodes).tolist())
        return [None] * n_episodes

    class EvaluationScheduleLogger:
        def __init__(self, initialization_draws, draws_per_submission):
            self.initialization_draws = initialization_draws
            self.draws_per_submission = draws_per_submission

        def initialize(self, env, policy, horizon, discount, rng):
            self.rng = rng
            self.rng.random(self.initialization_draws)

        def submit(self, trajectories, policy):
            self.rng.random(self.draws_per_submission * len(trajectories))

        def close(self):
            pass

    mocker.patch("potion.algorithms.reinforce.generate_batch", side_effect=generate)
    mocker.patch(
        "potion.algorithms.reinforce.gpomdp_estimator",
        return_value=np.zeros(n_params),
    )

    reinforce(
        env,
        policy,
        batch_size=3,
        max_iterations=None,
        max_trajectories=8,
        seed=123,
        logger=EvaluationScheduleLogger(1, 2),
        verbose=False,
    )
    first_schedule_draws = training_draws.copy()
    training_draws.clear()
    reinforce(
        env,
        policy,
        batch_size=3,
        max_iterations=None,
        max_trajectories=8,
        seed=123,
        logger=EvaluationScheduleLogger(100, 17),
        verbose=False,
    )

    assert training_draws == first_schedule_draws


def test_reusing_logger_for_two_algorithm_runs_starts_fresh(
        env, policy, n_params, mocker):
    mocker.patch(
        "potion.algorithms.reinforce.generate_batch",
        side_effect=lambda env, policy, n_episodes, horizon, **kwargs: [None] * n_episodes,
    )
    mocker.patch(
        "potion.algorithms.reinforce.gpomdp_estimator",
        return_value=np.zeros(n_params),
    )
    mocker.patch(
        "potion.evaluation.loggers.estimate_average_return",
        side_effect=[1., 2., 10., 20.],
    )
    logger = EpisodicTestLogger(
        log_every=1,
        n_test=1,
        verbose=False,
        path=None,
        keep_records=True,
    )

    for seed in (1, 2):
        reinforce(
            env,
            policy,
            batch_size=1,
            max_iterations=None,
            max_trajectories=1,
            seed=seed,
            logger=logger,
            verbose=False,
        )

    assert logger.tot_traj == 1
    assert [record["tot_trajectories"] for record in logger.records] == [0, 1]
    assert logger.normalized_auc == 15.
