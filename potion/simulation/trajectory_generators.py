import numpy as np
from joblib import Parallel, delayed
from collections.abc import Sequence



def generate_trajectory(env, policy, max_trajectory_len, seed):
    # Infer state and action dimensions from the environment
    ds = max(1, sum(env.observation_space.shape))
    da = max(1, sum(env.action_space.shape))

    # Prepare storage
    states = np.zeros((max_trajectory_len, ds), dtype=float)
    actions = np.zeros((max_trajectory_len, da), dtype=float)
    rewards = np.zeros(max_trajectory_len, dtype=float)
    alive = np.full(max_trajectory_len, False)

    # Generate independent seeds for environment and agent (low collision probability)
    seed_seq = np.random.SeedSequence(seed)
    env_seed, agent_seed = seed_seq.generate_state(2)
    agent_rng = np.random.default_rng(agent_seed)

    # Seed and reset the environment
    s, _ = env.reset(seed=env_seed.item())

    done = False
    t = 0
    while not done and t < max_trajectory_len:
        # Act
        a = policy.act(s, agent_rng)

        # Step
        next_s, r, terminated, truncated, info = env.step(a)
        done = terminated or truncated

        # Save data
        states[t] = s
        actions[t] = a
        rewards[t] = r
        alive[t] = True  # Mark episode as not yet finished

        s = next_s
        t += 1

    # Store terminal state (if there is one, and there is space for it)
    if t < max_trajectory_len and s is not None:
        states[t] = s

    return states, actions, rewards, alive


def blackbox_simulate_episode(env, policy, max_trajectory_len, seed, discount=1.):
    # Generate independent seeds for environment and agent (low collision probability)
    seed_seq = np.random.SeedSequence(seed)
    env_seed, agent_seed = seed_seq.generate_state(2)
    agent_rng = np.random.default_rng(agent_seed)

    # Seed and reset the environment
    s, _ = env.reset(env_seed)

    done = False
    t = 0
    ret = 0.
    while not done and t < max_trajectory_len:
        # Act
        a = policy.act(s, agent_rng)

        # Step
        next_s, r, terminated, truncated, info = env.step(a)
        done = terminated or truncated

        # Update return
        ret += discount**t * r

        s = next_s
        t += 1

    return ret, t


def generate_batch(env, policy, n_episodes, max_trajectory_len, rng, parallel=False, n_jobs=2):
    # A batch is a list of trajectories, a trajectory is a tuple of numpy arrays (states, actions, rewards, alive)

    # Generate independent seeds for the different episodes
    seeds = rng.bit_generator.seed_seq.generate_state(n_episodes)

    if not parallel:
        batch = [generate_trajectory(env, policy, max_trajectory_len, s) for s in seeds]
    else:
        # Joblib (with processes)
        batch = Parallel(backend="loky", n_jobs=n_jobs)(delayed(generate_trajectory)
                                                        (env, policy, max_trajectory_len, s)
                                                        for s in seeds)
    return batch


def blackbox_simulate_batch(env, policy, n_episodes, max_trajectory_len, rng, discount=1., parallel=False, n_jobs=2):
    # Generate independent seeds for the different episodes
    seeds = rng.bit_generator.seed_seq.generate_state(n_episodes)

    if not parallel:
        batch = [blackbox_simulate_episode(env, policy, max_trajectory_len, s, discount) for s in seeds]
    else:
        # Joblib (with processes)
        batch = Parallel(backend="loky", n_jobs=n_jobs)(delayed(blackbox_simulate_episode)
                                                        (env, policy, max_trajectory_len, s, discount)
                                                        for s in seeds)
    return batch


def unpack(batch):
    if not (isinstance(batch, Sequence) and isinstance(batch[0], tuple) and len(batch[0]) == 4):
        raise ValueError("batch should be a list of 4-tuples")
    return (np.stack(x) for x in zip(*batch))


def apply_mask(data, mask):
    if data.shape != mask.shape:
        if data.shape[:-1] != mask.shape:
            raise ValueError("Dimensions of data and mask should match, except possibly the last dimension of data")
        return data * mask[..., None]
    return data * mask


def apply_discount(rewards, disc):
    if not 0 <= disc <= 1:
        raise ValueError("discount factor should be between zero and one")

    horizon = rewards.shape[-1]
    factors = disc ** np.indices(dimensions=(horizon,))
    return rewards * factors
