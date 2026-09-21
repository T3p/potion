"""Shared algorithm run setup."""

import numpy as np
from tqdm.auto import tqdm

from potion.evaluation.loggers import EpisodicOnlineLogger


def initialize_run(seed, logger):
    """Return independent training/evaluation RNGs and a fresh default logger."""
    training_seed, evaluation_seed = np.random.SeedSequence(seed).spawn(2)
    training_rng = np.random.default_rng(training_seed)
    evaluation_rng = np.random.default_rng(evaluation_seed)
    if logger is None:
        logger = EpisodicOnlineLogger()
    return training_rng, evaluation_rng, logger


def capped_batch_size(
    requested_batch_size, total_trajectories, max_trajectories
):
    """Limit a batch request to the remaining trajectory budget."""
    if max_trajectories is None:
        return requested_batch_size
    return min(requested_batch_size, max_trajectories - total_trajectories)


def initialize_progress_bar(max_iterations, max_trajectories, description):
    """Create a progress bar using whichever stopping budget is active."""
    total = (
        max_trajectories if max_trajectories is not None else max_iterations
    )
    unit = "traj" if max_trajectories is not None else "it"
    return tqdm(total=total, desc=description, unit=unit, leave=True)


def update_progress_bar(progress_bar, max_trajectories, trajectories=0):
    """Advance a budget-aware progress bar after an algorithm update."""
    progress_bar.update(trajectories if max_trajectories is not None else 1)
