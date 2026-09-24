"""Search learning rates for CartPole policy-gradient algorithms.

Experiments run in single-threaded worker processes and explicitly reset every
RNG for reproducible, scheduling-independent results. Intermediate test
evaluations are used to compute the learning-curve AUC but are not written to
individual files.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os
from pathlib import Path
import random
import time

# Parallelize experiments, not numerical kernels or trajectory simulation.
# These settings must precede NumPy and PyTorch imports.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from torch import nn

from potion.algorithms import (
    def_pagepg,
    def_svrpg,
    def_srvrpg,
    def_stormpg,
    pagepg,
    reinforce,
    srvrpg,
    stormpg,
    svrpg,
)
from potion.evaluation.loggers import EpisodicTestLogger
from potion.evaluation.wandb_config import make_wandb_kwargs, wandb_enabled
from potion.policies.softmax_policies import DeepSoftmaxPolicy
from potion.simulation.vectorized_env import VectorizedBatchEnv


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
RESULTS_DIRECTORY = SCRIPT_DIRECTORY / "cartpole_results"

# CartPole policy and training configuration.
SEED = 3484
HIDDEN_SIZES = (32, 32)
TEMPERATURE = 1.0
INITIALIZATION = "xavier_uniform"

ESTIMATOR = "gpomdp"
BASELINE = "average"
DISCOUNT = 0.998
BATCH_SIZE = 100

HORIZON = 500
N_JOBS = 1
N_ENVS = 10
N_TEST = 100
LOG_PARAMETERS = False
WANDB_ENABLED = wandb_enabled()

LEARNING_RATES = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
SEEDS = tuple(SEED + offset for offset in range(5))
TRAINING_EPISODES = 10_000
LOG_EVERY = TRAINING_EPISODES - 1
MINI_BATCH_SIZE = 10
EPOCH_LENGTH = 10
MOMENTUM_PARAMETER = 0.9
REFRESH_PROBABILITY = 0.5
DEFENSIVE_PARAMETER = 0.5

ALGORITHMS = {
    "reinforce": reinforce,
    "svrpg": svrpg,
    "def-svrpg": def_svrpg,
    "srvrpg": srvrpg,
    "def-srvrpg": def_srvrpg,
    "storm-pg": stormpg,
    "def-storm-pg": def_stormpg,
    "page-pg": pagepg,
    "def-page-pg": def_pagepg,
}


def build_env():
    return gym.make("CartPole-v1")


def build_policy(env):
    layer_sizes = (env.observation_space.shape[0], *HIDDEN_SIZES, env.action_space.n)
    layers = []
    for input_size, output_size in zip(layer_sizes[:-2], layer_sizes[1:-1]):
        layers.extend((nn.Linear(input_size, output_size), nn.Tanh()))
    layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
    network = nn.Sequential(*layers)

    if INITIALIZATION != "xavier_uniform":
        raise ValueError("INITIALIZATION must be 'xavier_uniform'")
    for layer in network:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    return DeepSoftmaxPolicy(
        state_dim=env.observation_space.shape[0],
        num_actions=env.action_space.n,
        logit_network=network,
        temperature=TEMPERATURE,
    )


def run_experiment(algorithm_name, learning_rate, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = VectorizedBatchEnv(build_env, num_envs=N_ENVS)
    policy = build_policy(env)
    logger = EpisodicTestLogger(
        log_every=LOG_EVERY,
        n_test=N_TEST,
        verbose=False,
        log_params=LOG_PARAMETERS,
        override_discount=1.0,
        path=None,
        wandb=WANDB_ENABLED,
        wandb_kwargs=make_wandb_kwargs(
            default_project="potion-cartpole",
            name="lr-search-{}-{:.0e}-seed-{}".format(
                algorithm_name, learning_rate, seed
            ),
            group="learning-rate-search",
            tags=("cartpole", "learning-rate-search", algorithm_name),
            config={
                "algorithm": algorithm_name,
                "learning_rate": learning_rate,
                "seed": seed,
                "hidden_sizes": HIDDEN_SIZES,
                "temperature": TEMPERATURE,
                "estimator": ESTIMATOR,
                "baseline": BASELINE,
                "discount": DISCOUNT,
                "batch_size": BATCH_SIZE,
                "mini_batch_size": MINI_BATCH_SIZE,
                "epoch_length": EPOCH_LENGTH,
                "momentum_parameter": MOMENTUM_PARAMETER,
                "refresh_probability": REFRESH_PROBABILITY,
                "defensive_parameter": DEFENSIVE_PARAMETER,
                "training_episodes": TRAINING_EPISODES,
                "num_envs": N_ENVS,
            },
        ),
    )

    common_arguments = {
        "horizon": HORIZON,
        "discount": DISCOUNT,
        "step_size": learning_rate,
        "batch_size": BATCH_SIZE,
        "max_iterations": None,
        "max_trajectories": TRAINING_EPISODES,
        "estimator": ESTIMATOR,
        "baseline": BASELINE,
        "seed": seed,
        "logger": logger,
        "n_jobs": N_JOBS,
        "verbose": False,
    }
    if algorithm_name in {"svrpg", "def-svrpg", "srvrpg", "def-srvrpg"}:
        common_arguments.update(
            mini_batch_size=MINI_BATCH_SIZE,
            epoch_length=EPOCH_LENGTH,
        )
    elif algorithm_name in {"storm-pg", "def-storm-pg"}:
        common_arguments.update(
            mini_batch_size=MINI_BATCH_SIZE,
            momentum_parameter=MOMENTUM_PARAMETER,
        )
    elif algorithm_name in {"page-pg", "def-page-pg"}:
        common_arguments.update(
            mini_batch_size=MINI_BATCH_SIZE,
            refresh_probability=REFRESH_PROBABILITY,
        )
    if algorithm_name.startswith("def-"):
        common_arguments["defensive_parameter"] = DEFENSIVE_PARAMETER

    try:
        ALGORITHMS[algorithm_name](env, policy, **common_arguments)
    finally:
        env.close()

    if logger.tot_traj != TRAINING_EPISODES:
        raise RuntimeError(
            "{} used {} training episodes instead of {}".format(
                algorithm_name, logger.tot_traj, TRAINING_EPISODES
            )
        )
    if logger.normalized_auc is None:
        raise RuntimeError("The test logger did not produce an AUC")
    return float(logger.normalized_auc)


def _initialize_worker():
    """Keep each experiment process on a single compute thread."""
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _run_experiment_worker(algorithm_name, learning_rate, seed):
    started = time.perf_counter()
    auc = run_experiment(algorithm_name, learning_rate, seed)
    return auc, time.perf_counter() - started


def _seed_columns():
    return ["auc_seed_{}".format(seed) for seed in SEEDS]


def _update_summary(results):
    seed_columns = _seed_columns()
    complete = results[seed_columns].notna().all(axis=1)
    results["mean_auc"] = np.nan
    results.loc[complete, "mean_auc"] = results.loc[
        complete, seed_columns
    ].mean(axis=1)
    results["best"] = False
    for algorithm_name in ALGORITHMS:
        candidates = results[(results["algorithm"] == algorithm_name) & complete]
        if not candidates.empty:
            results.loc[candidates["mean_auc"].idxmax(), "best"] = True
    return results


def _load_results(path):
    seed_columns = _seed_columns()
    rows = [
        {
            "algorithm": algorithm_name,
            "learning_rate": learning_rate,
            **{column: np.nan for column in seed_columns},
        }
        for algorithm_name in ALGORITHMS
        for learning_rate in LEARNING_RATES
    ]
    results = pd.DataFrame(rows)

    if not path.exists():
        return _update_summary(results)

    saved = pd.read_csv(path)
    required_columns = {"algorithm", "learning_rate"}
    missing_columns = required_columns.difference(saved.columns)
    if missing_columns:
        raise ValueError(
            "Existing results file is missing columns: {}".format(
                ", ".join(sorted(missing_columns))
            )
        )

    for _, saved_row in saved.iterrows():
        matches = ((results["algorithm"] == saved_row["algorithm"])
                   & np.isclose(results["learning_rate"], saved_row["learning_rate"]))
        if not matches.any():
            continue
        if matches.sum() != 1:
            raise ValueError("Duplicate configuration in existing results file")
        result_index = results.index[matches][0]
        for column in seed_columns:
            if column in saved.columns and pd.notna(saved_row[column]):
                results.at[result_index, column] = saved_row[column]
    return _update_summary(results)


def _save_results(results, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    results.to_csv(temporary_path, index=False)
    temporary_path.replace(path)


def _configuration_index(results, algorithm_name, learning_rate):
    matches = ((results["algorithm"] == algorithm_name)
               & np.isclose(results["learning_rate"], learning_rate))
    if matches.sum() != 1:
        raise RuntimeError("Could not identify a unique result row")
    return results.index[matches][0]


def run_search(output_path, max_workers=None):
    total_runs = len(ALGORITHMS) * len(LEARNING_RATES) * len(SEEDS)
    results = _load_results(output_path)
    completed_runs = int(results[_seed_columns()].notna().sum().sum())
    search_started = time.perf_counter()

    print("Starting CartPole learning-rate search", flush=True)
    print("Learning-rate schedule: constant", flush=True)
    print(
        "Algorithms: {} | candidates: {} | seeds: {} | training episodes per run: {}".format(
            ", ".join(ALGORITHMS),
            len(LEARNING_RATES),
            ", ".join(str(seed) for seed in SEEDS),
            TRAINING_EPISODES,
        ),
        flush=True,
    )
    print("Total experiments: {}\n".format(total_runs), flush=True)
    if completed_runs:
        print(
            "Loaded {} completed experiment(s) from {}; they will be skipped.\n".format(
                completed_runs, output_path
            ),
            flush=True,
        )

    pending = []
    for algorithm_index, algorithm_name in enumerate(ALGORITHMS, start=1):
        print(
            "=== Algorithm {}/{}: {} ===".format(
                algorithm_index, len(ALGORITHMS), algorithm_name
            ),
            flush=True,
        )
        for candidate_index, learning_rate in enumerate(LEARNING_RATES, start=1):
            print(
                "  Candidate {}/{}: learning_rate={:.0e}".format(
                    candidate_index, len(LEARNING_RATES), learning_rate
                ),
                flush=True,
            )
            result_index = _configuration_index(results, algorithm_name, learning_rate)
            for seed in SEEDS:
                seed_column = "auc_seed_{}".format(seed)
                saved_auc = results.at[result_index, seed_column]
                if pd.notna(saved_auc):
                    print(
                        "    Experiment skipped: algorithm={}, learning_rate={:.0e}, seed={} "
                        "is already present in {} (AUC={:.6g})".format(
                            algorithm_name,
                            learning_rate,
                            seed,
                            output_path,
                            saved_auc,
                        ),
                        flush=True,
                    )
                    continue

                pending.append((algorithm_name, learning_rate, seed))

    available_cores = os.cpu_count() or 1
    if max_workers is None:
        max_workers = available_cores
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")

    worker_count = min(max_workers, len(pending)) if pending else 0
    print(
        "Pending experiments: {} | available cores: {} | workers: {}\n".format(
            len(pending), available_cores, worker_count
        ),
        flush=True,
    )

    if pending:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
            initializer=_initialize_worker,
        ) as executor:
            future_to_configuration = {}
            for configuration in pending:
                algorithm_name, learning_rate, seed = configuration
                print(
                    "Queueing algorithm={}, learning_rate={:.0e}, seed={}".format(
                        algorithm_name, learning_rate, seed
                    ),
                    flush=True,
                )
                future = executor.submit(_run_experiment_worker, *configuration)
                future_to_configuration[future] = configuration

            for future in as_completed(future_to_configuration):
                algorithm_name, learning_rate, seed = future_to_configuration[future]
                auc, run_elapsed = future.result()
                result_index = _configuration_index(
                    results, algorithm_name, learning_rate
                )
                results.at[result_index, "auc_seed_{}".format(seed)] = auc
                completed_runs += 1
                results = _update_summary(results)
                _save_results(results, output_path)
                print(
                    "[{}/{}] Completed algorithm={}, learning_rate={:.0e}, seed={}: "
                    "AUC={:.6g}, elapsed={:.1f}s; saved".format(
                        completed_runs,
                        total_runs,
                        algorithm_name,
                        learning_rate,
                        seed,
                        auc,
                        run_elapsed,
                    ),
                    flush=True,
                )

    results = _update_summary(results)
    _save_results(results, output_path)
    print(
        "Search completed in {:.1f}s.\n".format(time.perf_counter() - search_started),
        flush=True,
    )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIRECTORY / "cartpole_learning_rate_search.csv",
        help="Summary CSV path (default: %(default)s)",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Delete the existing output CSV and run every experiment from scratch",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help=(
            "Number of single-threaded experiment processes to run concurrently "
            "(default: number of available CPU cores)"
        ),
    )
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")

    if args.override:
        if args.output.exists():
            args.output.unlink()
            print("Removed existing results file: {}".format(args.output), flush=True)
        else:
            print("No existing results file to remove: {}".format(args.output), flush=True)

    print("\nSummary will be saved to {}".format(args.output.resolve()))

    results = run_search(
        args.output,
        max_workers=args.workers,
    )

    print("\nCartPole learning-rate search")
    print(results.to_string(index=False, float_format=lambda value: "{:.6g}".format(value)))
    print("\nSaved summary to {}".format(args.output.resolve()))


if __name__ == "__main__":
    main()
