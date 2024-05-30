import numpy as np
from joblib import Parallel, delayed


def sequential_trajectory_generator(env, policy, max_trajectory_len=200, max_episodes=None, seed_sequence=None):
    # Infer state and action dimensions from the environment
    ds = max(1, sum(env.observation_space.shape))
    da = max(1, sum(env.action_space.shape))

    # Stream of random seeds minimizing collision probability
    if seed_sequence is None:
        seed_sequence = np.random.SeedSequence(None)

    n = 0
    while max_episodes is None or n < max_episodes:
        # Prepare storage
        states = np.zeros((max_trajectory_len, ds), dtype=float)
        actions = np.zeros((max_trajectory_len, da), dtype=float)
        rewards = np.zeros(max_trajectory_len, dtype=float)
        alive = np.full(max_trajectory_len, False)

        # Reset (and seed) environment
        init_seed, = seed_sequence.generate_state(1)
        s, _ = env.reset(init_seed)

        done = False
        t = 0
        while not done and t < max_trajectory_len:
            # Generate a seed for the next action
            action_seed, = seed_sequence.generate_state(1)

            # Step
            a = policy.act(s, action_seed)
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

        yield states, actions, rewards, alive
        n += 1


def parallel_episode_generator(env, policy, max_trajectory_len=200, seed_sequence=None):
    # Infer state and action dimensions from the environment
    ds = max(1, sum(env.observation_space.shape))
    da = max(1, sum(env.action_space.shape))

    # Stream of random seeds minimizing collision probability
    if seed_sequence is None:
        seed_sequence = np.random.SeedSequence(None)

    # Prepare storage
    states = np.zeros((max_trajectory_len, ds), dtype=float)
    actions = np.zeros((max_trajectory_len, da), dtype=float)
    rewards = np.zeros(max_trajectory_len, dtype=float)
    alive = np.full(max_trajectory_len, False)

    # Reset (and seed) the environment
    init_seed, = seed_sequence.generate_state(1)
    s, _ = env.reset(seed=init_seed)

    done = False
    t = 0
    while not done and t < max_trajectory_len:
        # Act
        action_seed, = seed_sequence.generate_state(1)
        a = policy.act(s, action_seed)

        # Step
        next_s, r, terminated, truncated, info = env.step(a.numpy())
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


def minimal_episode_generator(env, policy, max_trajectory_len=200, seed_sequence=None, discount=1.):
    # Stream of random seeds minimizing collision probability
    if seed_sequence is None:
        seed_sequence = np.random.SeedSequence(None)

    # Reset (and seed) the environment
    init_seed, = seed_sequence.generate_state(1)
    s, _ = env.reset(init_seed)

    done = False
    t = 0
    ret = 0.
    undiscounted_return = 0.
    while not done and t < max_trajectory_len:
        # Act
        action_seed, = seed_sequence.generate_state(1)
        a = policy.act(s, action_seed)

        # Step
        next_s, r, terminated, truncated, info = env.step(a.numpy())
        done = terminated or truncated

        # Update return
        ret += discount**t * r
        undiscounted_return += r

        s = next_s
        t += 1

    return ret, undiscounted_return, t


def generate_batch(env, policy, max_trajectory_len=200, episodes=100, parallel=False, n_jobs=2, seed_sequence=None):
    # A batch is a list of trajectories, a trajectory is a tuple of numpy arrays (states, actions, rewards, alive)
    if not parallel:
        traj_gen = sequential_trajectory_generator(env, policy, max_trajectory_len, max_episodes=episodes,
                                                   seed_sequence=seed_sequence)
        batch = [traj for traj in traj_gen]
    else:
        # Each process has its own independent sequence of seeds
        if seed_sequence is None:
            seed_sequence = np.random.SeedSequence(None)
        parallel_seed_sequences = seed_sequence.spawn(episodes)

        # Joblib (with processes)
        batch = Parallel(n_jobs=n_jobs)(delayed(parallel_episode_generator)(env, policy, max_trajectory_len, pss)
                                        for pss in parallel_seed_sequences)
    return batch
