import numpy as np
from joblib import Parallel, delayed


def sequential_trajectory_generator(env, policy, max_trajectory_len, rng):
    # Infer state and action dimensions from the environment
    ds = max(1, sum(env.observation_space.shape))
    da = max(1, sum(env.action_space.shape))

    while True:
        # Prepare storage
        states = np.zeros((max_trajectory_len, ds), dtype=float)
        actions = np.zeros((max_trajectory_len, da), dtype=float)
        rewards = np.zeros(max_trajectory_len, dtype=float)
        alive = np.full(max_trajectory_len, False)

        # Generate independent pseudo-random stream to seed the environment
        # If the env accepts a rng instead of a seed, use env_rng, = rng.spawn(1)
        init_seed, = rng.bit_generator.seed_seq.generate_state(1)

        # Seed and reset the environment
        s, _ = env.reset(init_seed)

        done = False
        t = 0
        while not done and t < max_trajectory_len:
            # Act
            a = policy.act(s, rng)

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

        yield states, actions, rewards, alive


def parallel_episode_generator(env, policy, max_trajectory_len, rng):
    # Infer state and action dimensions from the environment
    ds = max(1, sum(env.observation_space.shape))
    da = max(1, sum(env.action_space.shape))

    # Prepare storage
    states = np.zeros((max_trajectory_len, ds), dtype=float)
    actions = np.zeros((max_trajectory_len, da), dtype=float)
    rewards = np.zeros(max_trajectory_len, dtype=float)
    alive = np.full(max_trajectory_len, False)

    # Generate independent pseudo-random stream to seed the environment
    # If the env accepts a rng instead of a seed, use env_rng, = rng.spawn(1)
    # noinspection
    init_seed, = rng.bit_generator.seed_seq.generate_state(1)

    # Seed and reset the environment
    s, _ = env.reset(init_seed)

    done = False
    t = 0
    while not done and t < max_trajectory_len:
        # Act
        a = policy.act(s, rng)

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


def minimal_episode_generator(env, policy, max_trajectory_len, rng, discount=1.):
    while True:
        # Generate independent pseudo-random stream to seed the environment
        # If the env accepts a rng instead of a seed, use env_rng, = rng.spawn(1)
        init_seed, = rng.bit_generator.seed_seq.generate_state(1)

        # Seed and reset the environment
        s, _ = env.reset(init_seed)

        done = False
        t = 0
        ret = 0.
        while not done and t < max_trajectory_len:
            # Act
            a = policy.act(s, rng)

            # Step
            next_s, r, terminated, truncated, info = env.step(a.numpy())
            done = terminated or truncated

            # Update return
            ret += discount**t * r

            s = next_s
            t += 1

        yield ret, t


def generate_batch(env, policy, rng, max_trajectory_len=200, episodes=100, parallel=False, n_jobs=2):
    # A batch is a list of trajectories, a trajectory is a tuple of numpy arrays (states, actions, rewards, alive)
    if not parallel:
        traj_gen = sequential_trajectory_generator(env, policy, max_trajectory_len, rng)
        batch = [next(traj_gen) for _ in range(episodes)]
    else:
        # Each process has its own independent pseudo-random stream
        parallel_generators = rng.spawn(episodes)
        # Joblib (with processes)
        batch = Parallel(backend="loky", n_jobs=n_jobs)(delayed(parallel_episode_generator)
                                                        (env, policy, max_trajectory_len, r)
                                                        for r in parallel_generators)
    return batch
