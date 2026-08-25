"""Shared algorithm run setup."""

import numpy as np

from potion.evaluation.loggers import EpisodicOnlineLogger


def initialize_run(seed, logger):
    """Return independent training/evaluation RNGs and a fresh default logger."""
    training_seed, evaluation_seed = np.random.SeedSequence(seed).spawn(2)
    training_rng = np.random.default_rng(training_seed)
    evaluation_rng = np.random.default_rng(evaluation_seed)
    if logger is None:
        logger = EpisodicOnlineLogger()
    return training_rng, evaluation_rng, logger


def capped_batch_size(requested_batch_size, total_trajectories, max_trajectories):
    """Limit a batch request to the remaining trajectory budget."""
    if max_trajectories is None:
        return requested_batch_size
    return min(requested_batch_size, max_trajectories - total_trajectories)
