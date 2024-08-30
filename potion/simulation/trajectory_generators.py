import numpy as np
from joblib import Parallel, delayed
from collections.abc import Sequence
import scipy.stats as sts


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
        #print(s)
        a = policy.act(s, agent_rng, t)

        # Step
        next_s, r, terminated, truncated, info = env.step(a)
        #print(s, r)
        done = terminated or truncated

        # Save data
        states[t] = s
        actions[t] = a
        rewards[t] = r
        alive[t] = True  # Mark episode as not yet finished

        s = next_s
        t += 1

    # Store final state (if there is one, and there is space for it)
    if t < max_trajectory_len and s is not None:
        states[t] = s

    return states, actions, rewards, alive


def blackbox_simulate_episode(env, policy, max_trajectory_len, seed, discount=1.):
    # Generate independent seeds for environment and agent (low collision probability)
    seed_seq = np.random.SeedSequence(seed)
    env_seed, agent_seed = seed_seq.generate_state(2)
    agent_rng = np.random.default_rng(agent_seed)

    # Seed and reset the environment
    s, _ = env.reset(seed=env_seed.item())

    done = False
    t = 0
    ret = 0.
    while not done and t < max_trajectory_len:
        # Act
        a = policy.act(s, agent_rng, t)

        # Step
        next_s, r, terminated, truncated, info = env.step(a)
        done = terminated or truncated

        # Update return
        ret += discount**t * r

        s = next_s
        t += 1

    return ret, t


def generate_batch(env, policy, n_episodes, max_trajectory_len, rng, discount=None, parallel=False, n_jobs=4):
    # A batch is a list of trajectories, a trajectory is a tuple of numpy arrays (states, actions, rewards, alive)

    if max_trajectory_len is None:
        max_trajectory_len = int(2. / (1. - discount))
        return generate_batch_continual(env, policy, n_episodes, discount, rng, max_trajectory_len, parallel, n_jobs)

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
    if max_trajectory_len is None:
        return blackbox_simulate_batch_continual(env, policy, n_episodes, discount, rng, parallel, n_jobs)

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


def estimate_average_return(env, policy, n_episodes, horizon, rng, discount=1., parallel=False, n_jobs=2):
    batch = blackbox_simulate_batch(env, policy, n_episodes, horizon, rng, discount, parallel, n_jobs)
    rets, _ = zip(*batch)
    return np.mean(rets)


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


def apply_discount(rewards, discount):
    if not 0 <= discount <= 1:
        raise ValueError("discount factor should be between zero and one")

    horizon = rewards.shape[-1]
    factors = discount ** np.indices(dimensions=(horizon,))
    return rewards * factors


def simulate_infinite_trajectory(env, policy, discount, seed, max_trajectory_len):
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
    env_seed, agent_seed, horizon_seed = seed_seq.generate_state(3)
    agent_rng = np.random.default_rng(agent_seed)
    horizon_rng = np.random.default_rng(horizon_seed)

    # Seed and reset the environment
    s, _ = env.reset(seed=env_seed.item())

    done = False
    t = 0
    random_horizon = horizon_rng.geometric(1 - discount)
    while not done and t < min(random_horizon, max_trajectory_len):
        # Act
        a = policy.act(s, agent_rng, t)

        # Step
        next_s, r, terminated, truncated, info = env.step(a)
        done = terminated or truncated

        # Save data
        states[t] = s
        actions[t] = a
        rewards[t] = r
        alive[t] = t < random_horizon  # Mark episode as not yet finished

        s = next_s
        t += 1

    # Store final state (if there is one, and there is space for it)
    if t < max_trajectory_len and s is not None:
        states[t] = s

    return states, actions, rewards, alive


def generate_batch_continual(env, policy, n_episodes, discount, rng, max_trajectory_len, parallel=False, n_jobs=4):
    # A batch is a list of trajectories, a trajectory is a tuple of numpy arrays (states, actions, rewards, alive)
    # Generate independent seeds for the different episodes
    seeds = rng.bit_generator.seed_seq.generate_state(n_episodes)

    if not parallel:
        batch = [simulate_infinite_trajectory(env, policy, discount, s, max_trajectory_len) for s in seeds]
    else:
        # Joblib (with processes)
        batch = Parallel(backend="loky", n_jobs=n_jobs)(delayed(simulate_infinite_trajectory)
                                                        (env, policy, max_trajectory_len, s)
                                                        for s in seeds)
    return batch


def blackbox_simulate_infinite_trajectory(env, policy, discount, seed, max_trajectory_len=1000):
    # Generate independent seeds for environment and agent (low collision probability)
    seed_seq = np.random.SeedSequence(seed)
    env_seed, agent_seed, horizon_seed = seed_seq.generate_state(3)
    agent_rng = np.random.default_rng(agent_seed)
    horizon_rng = np.random.default_rng(horizon_seed)

    # Seed and reset the environment
    s, _ = env.reset(seed=env_seed.item())

    random_horizon = horizon_rng.geometric(1 - discount)
    done = False
    t = 0
    ret = 0.
    while not done and t < min(random_horizon, max_trajectory_len):
        # Act
        a = policy.act(s, agent_rng, t)

        # Step
        next_s, r, terminated, truncated, info = env.step(a)
        done = terminated or truncated

        # Update return
        ret += r

        s = next_s
        t += 1

    return ret, t


def blackbox_simulate_batch_continual(env, policy, n_episodes, discount, rng, parallel=False, n_jobs=2):
    # Generate independent seeds for the different episodes
    seeds = rng.bit_generator.seed_seq.generate_state(n_episodes)

    if not parallel:
        batch = [blackbox_simulate_infinite_trajectory(env, policy, discount, s) for s in seeds]
    else:
        # Joblib (with processes)
        batch = Parallel(backend="loky", n_jobs=n_jobs)(delayed(blackbox_simulate_infinite_trajectory)
                                                        (env, policy, discount, s)
                                                        for s in seeds)
    return batch
