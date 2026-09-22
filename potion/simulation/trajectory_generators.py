import numpy as np
from joblib import Parallel, delayed
from collections.abc import Sequence
import scipy.stats as sts

from potion.simulation.vectorized_env import VectorizedBatchEnv


class TrajectoryBatch(Sequence):
    """Sequence-compatible batch backed by five contiguous NumPy arrays."""

    def __init__(self, states, actions, rewards, alive, logps):
        arrays = (states, actions, rewards, alive, logps)
        if any(not isinstance(array, np.ndarray) for array in arrays):
            raise TypeError("trajectory batch entries should be NumPy arrays")
        if any(array.ndim == 0 for array in arrays):
            raise ValueError("trajectory batch arrays should include a batch dimension")
        if any(len(array) != len(states) for array in arrays[1:]):
            raise ValueError("trajectory batch arrays should have the same length")
        (self.states, self.actions, self.rewards, self.alive, self.logps) = (
            np.ascontiguousarray(array) for array in arrays
        )

    @property
    def arrays(self):
        return self.states, self.actions, self.rewards, self.alive, self.logps

    @classmethod
    def allocate(cls, env, n_episodes, horizon):
        state_dim = max(1, sum(env.observation_space.shape))
        action_dim = max(1, sum(env.action_space.shape))
        action_dtype = (
            np.float32
            if np.issubdtype(env.action_space.dtype, np.floating)
            else env.action_space.dtype
        )
        return cls(
            np.zeros((n_episodes, horizon, state_dim), dtype=np.float32),
            np.zeros((n_episodes, horizon, action_dim), dtype=action_dtype),
            np.zeros((n_episodes, horizon), dtype=np.float32),
            np.zeros((n_episodes, horizon), dtype=bool),
            np.zeros((n_episodes, horizon), dtype=np.float32),
        )

    @classmethod
    def from_trajectories(cls, trajectories):
        if not trajectories:
            raise ValueError("cannot build a trajectory batch from no trajectories")
        if any(not isinstance(trajectory, tuple) or len(trajectory) != 5
               for trajectory in trajectories):
            raise ValueError("trajectories should be 5-tuples")
        return cls(*(np.stack(values) for values in zip(*trajectories)))

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        arrays = tuple(array[index] for array in self.arrays)
        if isinstance(index, (int, np.integer)):
            return arrays
        return TrajectoryBatch(*arrays)

    def __iter__(self):
        return iter(zip(*self.arrays))


def _generate_episode_seeds(rng, n_episodes):
    """Draw advancing, reproducible uint32 seeds from ``rng``."""
    return rng.integers(
        0,
        np.iinfo(np.uint32).max,
        size=n_episodes,
        dtype=np.uint32,
    )


def _batched_actions(policy, states, rngs, t):
    act_batch = getattr(policy, "act_batch_and_log_prob", None)
    if callable(act_batch):
        return act_batch(states, rngs, t)

    samples = []
    for state, agent_rng in zip(states, rngs):
        act_and_log_prob = getattr(policy, "act_and_log_prob", None)
        if callable(act_and_log_prob):
            samples.append(act_and_log_prob(state, agent_rng, t))
        else:
            action = policy.act(state, agent_rng, t)
            samples.append((action, policy.log_prob(state, action, t)))
    actions, logps = zip(*samples)
    return np.asarray(actions), np.asarray(logps)


def _default_vector_actions(action_space, num_envs):
    if hasattr(action_space, "n"):
        return np.full(num_envs, getattr(action_space, "start", 0), dtype=int)
    action = np.clip(np.zeros(action_space.shape), action_space.low, action_space.high)
    return np.broadcast_to(action, (num_envs,) + action_space.shape).copy()


def _replace_autoreset_observations(next_states, infos):
    """Recover terminal observations hidden by Gymnasium autoreset."""
    final_observations = infos.get("final_observation")
    final_mask = infos.get("_final_observation")
    if final_observations is None or final_mask is None:
        return next_states
    actual_next_states = np.array(next_states, copy=True)
    for lane in np.flatnonzero(final_mask):
        actual_next_states[lane] = final_observations[lane]
    return actual_next_states


def _generate_vectorized_batch(
        env, policy, n_episodes, max_trajectory_len, seeds, discount=None):
    batch = TrajectoryBatch.allocate(env, n_episodes, max_trajectory_len)
    if n_episodes == 0:
        return batch

    vector_env = env.vector_env
    num_envs = env.num_envs
    default_actions = _default_vector_actions(
        vector_env.single_action_space, num_envs
    )

    for start in range(0, n_episodes, num_envs):
        stop = min(start + num_envs, n_episodes)
        wave_size = stop - start
        wave_seeds = seeds[start:stop]
        env_seeds = []
        agent_rngs = []
        episode_limits = np.full(wave_size, max_trajectory_len, dtype=int)

        for episode_seed in wave_seeds:
            seed_seq = np.random.SeedSequence(episode_seed)
            if discount is None:
                env_seed, agent_seed = seed_seq.generate_state(2)
            else:
                env_seed, agent_seed, horizon_seed = seed_seq.generate_state(3)
                horizon_rng = np.random.default_rng(horizon_seed)
                episode_limits[len(env_seeds)] = min(
                    horizon_rng.geometric(1 - discount), max_trajectory_len
                )
            env_seeds.append(env_seed.item())
            agent_rngs.append(np.random.default_rng(agent_seed))

        # SyncVectorEnv has a fixed width. Seeds for unused lanes do not affect
        # collected episodes and those lanes are kept inactive throughout.
        reset_seeds = env_seeds + [0] * (num_envs - wave_size)
        current_states, _ = vector_env.reset(seed=reset_seeds)
        active = np.zeros(num_envs, dtype=bool)
        active[:wave_size] = episode_limits > 0

        for t in range(max_trajectory_len):
            lanes = np.flatnonzero(active)
            if len(lanes) == 0:
                break
            episode_indices = start + lanes
            policy_states = current_states[lanes]
            if policy_states.ndim == 1 and batch.states.ndim == 3:
                policy_states = policy_states[:, None]
            actions, logps = _batched_actions(
                policy,
                policy_states,
                [agent_rngs[lane] for lane in lanes],
                t,
            )
            actions = np.asarray(actions)
            logps = np.asarray(logps)

            vector_actions = default_actions.copy()
            environment_actions = actions
            if (vector_env.single_action_space.shape == ()
                    and environment_actions.ndim == 2
                    and environment_actions.shape[-1] == 1):
                environment_actions = environment_actions[:, 0]
            vector_actions[lanes] = environment_actions
            next_states, rewards, terminated, truncated, infos = vector_env.step(
                vector_actions
            )
            actual_next_states = _replace_autoreset_observations(
                next_states, infos
            )

            batch.states[episode_indices, t] = policy_states
            stored_actions = actions
            if stored_actions.ndim == 1 and batch.actions.ndim == 3:
                stored_actions = stored_actions[:, None]
            batch.actions[episode_indices, t] = stored_actions
            batch.rewards[episode_indices, t] = rewards[lanes]
            batch.alive[episode_indices, t] = True
            batch.logps[episode_indices, t] = logps
            if t + 1 < max_trajectory_len:
                stored_next_states = actual_next_states[lanes]
                if stored_next_states.ndim == 1 and batch.states.ndim == 3:
                    stored_next_states = stored_next_states[:, None]
                batch.states[episode_indices, t + 1] = stored_next_states

            finished = terminated | truncated
            reached_horizon = t + 1 >= episode_limits
            active[lanes] = ~(finished[lanes] | reached_horizon[lanes])
            current_states = next_states

    return batch


def _generate_trajectory_into(env, policy, max_trajectory_len, seed, storage):
    states, actions, rewards, alive, logps = storage
    for array in storage:
        array.fill(0)

    # Generate independent seeds for environment and agent (low collision probability)
    seed_seq = np.random.SeedSequence(seed)
    env_seed, agent_seed = seed_seq.generate_state(2)
    agent_rng = np.random.default_rng(agent_seed)

    # Seed and reset the environment
    s, _ = env.reset(seed=env_seed.item())

    done = False
    t = 0
    while not done and t < max_trajectory_len:
        act_and_log_prob = getattr(policy, "act_and_log_prob", None)
        if callable(act_and_log_prob):
            a, logp = act_and_log_prob(s, agent_rng, t)
        else:
            a = policy.act(s, agent_rng, t)
            logp = policy.log_prob(s, a, t)

        next_s, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated

        states[t] = s
        actions[t] = a
        rewards[t] = r
        alive[t] = True
        logps[t] = np.asarray(logp).item()

        s = next_s
        t += 1

    if t < max_trajectory_len and s is not None:
        states[t] = s


def generate_trajectory(env, policy, max_trajectory_len, seed):
    # Infer state and action dimensions from the environment
    ds = max(1, sum(env.observation_space.shape))
    da = max(1, sum(env.action_space.shape))

    # Prepare storage
    states = np.zeros((max_trajectory_len, ds), dtype=np.float32)
    action_dtype = (
        np.float32
        if np.issubdtype(env.action_space.dtype, np.floating)
        else env.action_space.dtype
    )
    actions = np.zeros((max_trajectory_len, da), dtype=action_dtype)
    rewards = np.zeros(max_trajectory_len, dtype=np.float32)
    alive = np.full(max_trajectory_len, False)
    logps = np.zeros(max_trajectory_len, dtype=np.float32)
    trajectory = states, actions, rewards, alive, logps
    _generate_trajectory_into(
        env, policy, max_trajectory_len, seed, trajectory
    )
    return trajectory


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
    # A batch contains states, actions, rewards, alive flags, and logps in
    # contiguous arrays while retaining sequence-style trajectory access.

    if max_trajectory_len is None:
        max_trajectory_len = int(2. / (1. - discount))
        return generate_batch_continual(env, policy, n_episodes, discount, rng, max_trajectory_len, parallel, n_jobs)

    # Generate independent seeds for the different episodes
    seeds = _generate_episode_seeds(rng, n_episodes)
    if isinstance(env, VectorizedBatchEnv):
        if parallel:
            raise ValueError(
                "VectorizedBatchEnv cannot be combined with parallel collection"
            )
        return _generate_vectorized_batch(
            env, policy, n_episodes, max_trajectory_len, seeds
        )
    if n_episodes == 0:
        return TrajectoryBatch.allocate(env, 0, max_trajectory_len)

    if not parallel:
        batch = TrajectoryBatch.allocate(env, n_episodes, max_trajectory_len)
        for index, seed in enumerate(seeds):
            _generate_trajectory_into(
                env, policy, max_trajectory_len, seed, batch[index]
            )
    else:
        # Joblib (with processes)
        trajectories = Parallel(backend="loky", n_jobs=n_jobs)(
            delayed(generate_trajectory)(env, policy, max_trajectory_len, seed)
            for seed in seeds
        )
        batch = TrajectoryBatch.from_trajectories(trajectories)
    return batch


def blackbox_simulate_batch(env, policy, n_episodes, max_trajectory_len, rng, discount=1., parallel=False, n_jobs=2):
    if max_trajectory_len is None:
        return blackbox_simulate_batch_continual(env, policy, n_episodes, discount, rng, parallel, n_jobs)

    # Generate independent seeds for the different episodes
    seeds = _generate_episode_seeds(rng, n_episodes)
    if n_episodes == 0:
        return []

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
    if isinstance(batch, TrajectoryBatch):
        return batch.arrays
    if not (isinstance(batch, Sequence) and batch
            and isinstance(batch[0], tuple) and len(batch[0]) == 5):
        raise ValueError("batch should be a list of 5-tuples")
    return tuple(np.stack(values) for values in zip(*batch))


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


def _simulate_infinite_trajectory_into(
        env, policy, discount, seed, max_trajectory_len, storage):
    states, actions, rewards, alive, logps = storage
    for array in storage:
        array.fill(0)

    seed_seq = np.random.SeedSequence(seed)
    env_seed, agent_seed, horizon_seed = seed_seq.generate_state(3)
    agent_rng = np.random.default_rng(agent_seed)
    horizon_rng = np.random.default_rng(horizon_seed)

    s, _ = env.reset(seed=env_seed.item())

    done = False
    t = 0
    random_horizon = horizon_rng.geometric(1 - discount)
    while not done and t < min(random_horizon, max_trajectory_len):
        act_and_log_prob = getattr(policy, "act_and_log_prob", None)
        if callable(act_and_log_prob):
            a, logp = act_and_log_prob(s, agent_rng, t)
        else:
            a = policy.act(s, agent_rng, t)
            logp = policy.log_prob(s, a, t)

        next_s, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated

        states[t] = s
        actions[t] = a
        rewards[t] = r
        alive[t] = t < random_horizon
        logps[t] = np.asarray(logp).item()

        s = next_s
        t += 1

    if t < max_trajectory_len and s is not None:
        states[t] = s


def simulate_infinite_trajectory(env, policy, discount, seed, max_trajectory_len):
    # Infer state and action dimensions from the environment
    ds = max(1, sum(env.observation_space.shape))
    da = max(1, sum(env.action_space.shape))

    # Prepare storage
    states = np.zeros((max_trajectory_len, ds), dtype=np.float32)
    action_dtype = (
        np.float32
        if np.issubdtype(env.action_space.dtype, np.floating)
        else env.action_space.dtype
    )
    actions = np.zeros((max_trajectory_len, da), dtype=action_dtype)
    rewards = np.zeros(max_trajectory_len, dtype=np.float32)
    alive = np.full(max_trajectory_len, False)
    logps = np.zeros(max_trajectory_len, dtype=np.float32)
    trajectory = states, actions, rewards, alive, logps
    _simulate_infinite_trajectory_into(
        env, policy, discount, seed, max_trajectory_len, trajectory
    )
    return trajectory


def generate_batch_continual(env, policy, n_episodes, discount, rng, max_trajectory_len, parallel=False, n_jobs=4):
    # Generate independent seeds for the different episodes
    seeds = _generate_episode_seeds(rng, n_episodes)
    if isinstance(env, VectorizedBatchEnv):
        if parallel:
            raise ValueError(
                "VectorizedBatchEnv cannot be combined with parallel collection"
            )
        return _generate_vectorized_batch(
            env, policy, n_episodes, max_trajectory_len, seeds, discount
        )
    if n_episodes == 0:
        return TrajectoryBatch.allocate(env, 0, max_trajectory_len)

    if not parallel:
        batch = TrajectoryBatch.allocate(env, n_episodes, max_trajectory_len)
        for index, seed in enumerate(seeds):
            _simulate_infinite_trajectory_into(
                env, policy, discount, seed, max_trajectory_len, batch[index]
            )
    else:
        # Joblib (with processes)
        trajectories = Parallel(backend="loky", n_jobs=n_jobs)(
            delayed(simulate_infinite_trajectory)(
                env, policy, discount, seed, max_trajectory_len
            )
            for seed in seeds
        )
        batch = TrajectoryBatch.from_trajectories(trajectories)
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
    seeds = _generate_episode_seeds(rng, n_episodes)

    if not parallel:
        batch = [blackbox_simulate_infinite_trajectory(env, policy, discount, s) for s in seeds]
    else:
        # Joblib (with processes)
        batch = Parallel(backend="loky", n_jobs=n_jobs)(delayed(blackbox_simulate_infinite_trajectory)
                                                        (env, policy, discount, s)
                                                        for s in seeds)
    return batch
