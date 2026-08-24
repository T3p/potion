"""Train a deep softmax policy on Gymnasium's CartPole-v1."""

import gymnasium as gym
from torch import nn
import torch

from potion.algorithms import reinforce
from potion.evaluation.loggers import EpisodicTestLogger
from potion.optimization.gradient_descent import Adam
from potion.policies.softmax_policies import DeepSoftmaxPolicy


# Edit this block to change the experiment.
SEED = 42

HIDDEN_SIZES = (32, 32)
ACTIVATION = "tanh"
TEMPERATURE = 1.0
INITIALIZATION = "xavier_uniform"

ESTIMATOR = "gpomdp"
BASELINE = "average"
DISCOUNT = 0.995
BATCH_SIZE = 100
LEARNING_RATE = 1e-3
OPTIMIZER = "constant"

MAX_ITERATIONS = None
MAX_TRAJECTORIES = 10000

HORIZON = 200
N_JOBS = 1

LOG_EVERY = 1000
N_TEST = 100
LOG_PARAMETERS = False
ALGORITHM_VERBOSE = True
LOGGER_VERBOSE = True


def build_policy(env):
    activation_types = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
    }
    if ACTIVATION not in activation_types:
        raise ValueError("ACTIVATION must be 'relu' or 'tanh'")

    layer_sizes = (env.observation_space.shape[0], *HIDDEN_SIZES, env.action_space.n)
    layers = []
    for input_size, output_size in zip(layer_sizes[:-2], layer_sizes[1:-1]):
        layers.extend((nn.Linear(input_size, output_size), activation_types[ACTIVATION]()))
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


def main():
    torch.manual_seed(SEED)
    env = gym.make("CartPole-v1")
    policy = build_policy(env)

    if OPTIMIZER == "adam":
        step_size = Adam(alpha=LEARNING_RATE)
    elif OPTIMIZER == "constant":
        step_size = LEARNING_RATE
    else:
        raise ValueError("OPTIMIZER must be 'adam' or 'constant'")

    logger = EpisodicTestLogger(
        log_every=LOG_EVERY,
        n_test=N_TEST,
        verbose=LOGGER_VERBOSE,
        log_params=LOG_PARAMETERS,
        path=None,
    )

    print("Training DeepSoftmaxPolicy on CartPole-v1")
    print("seed={}, batch_size={}, max_trajectories={}".format(
        SEED, BATCH_SIZE, MAX_TRAJECTORIES
    ))

    try:
        reinforce(
            env,
            policy,
            horizon=HORIZON,
            discount=DISCOUNT,
            step_size=step_size,
            batch_size=BATCH_SIZE,
            max_iterations=MAX_ITERATIONS,
            max_trajectories=MAX_TRAJECTORIES,
            estimator=ESTIMATOR,
            baseline=BASELINE,
            seed=SEED,
            logger=logger,
            n_jobs=N_JOBS,
            verbose=ALGORITHM_VERBOSE,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
