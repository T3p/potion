from potion.simulation.trajectory_generators import (generate_trajectory,
                                                     blackbox_simulate_episode,
                                                     generate_batch,
                                                     blackbox_simulate_batch)
import numpy as np


def test_generate_trajectory_shapes(env, policy, max_trajectory_len, seed, state_d, action_d):
    traj = generate_trajectory(env, policy, max_trajectory_len, seed)

    # A trajectory is a tuple (states, actions, rewards, alive)
    assert len(traj) == 4
    states, actions, rewards, alive = traj
    assert states.shape == (max_trajectory_len, state_d)
    assert actions.shape == (max_trajectory_len, action_d)
    assert rewards.shape == (max_trajectory_len,)
    assert alive.shape == (max_trajectory_len,)


def test_generate_trajectory_1d(env_1d, policy_1d, max_trajectory_len, seed):
    traj = generate_trajectory(env_1d, policy_1d, max_trajectory_len, seed)

    # A trajectory is a tuple (states, actions, rewards, alive)
    assert len(traj) == 4
    states, actions, rewards, alive = traj
    assert states.shape == (max_trajectory_len, 1)
    assert actions.shape == (max_trajectory_len, 1)
    assert rewards.shape == (max_trajectory_len,)
    assert alive.shape == (max_trajectory_len,)


def test_generate_trajectory_alive(env, policy, max_trajectory_len, seed, horizon):
    _, _, _, alive = generate_trajectory(env, policy, max_trajectory_len, seed)

    for i in range(horizon):
        assert alive[i]
    for i in range(horizon, max_trajectory_len):
        assert not alive[i]


def test_blackbox_simulate_episode(env, policy, max_trajectory_len, seed, discount, horizon):
    ret, traj_len = blackbox_simulate_episode(env, policy, max_trajectory_len, seed, discount)

    assert traj_len == horizon
    assert np.isclose(ret, -(1-discount**horizon) / (1-discount))


def test_blackbox_simulate_episode_1d(env_1d, policy_1d, max_trajectory_len, seed, discount, horizon):
    ret, traj_len = blackbox_simulate_episode(env_1d, policy_1d, max_trajectory_len, seed, discount)

    assert traj_len == horizon
    assert np.isclose(ret, -(1-discount**horizon) / (1-discount))


def test_generate_batch_shape(env, policy, max_trajectory_len, rng, n_jobs, state_d, action_d):
    # Sequential
    batch = generate_batch(env, policy, n_episodes=1, max_trajectory_len=max_trajectory_len, rng=rng, parallel=False)
    traj = generate_trajectory(env, policy, max_trajectory_len, seed=None)
    for i, x in enumerate(batch[0]):
        assert x.shape == traj[i].shape

    # Parallel
    batch = generate_batch(env, policy, n_episodes=1, max_trajectory_len=max_trajectory_len, rng=rng, parallel=True,
                           n_jobs=n_jobs)
    traj = generate_trajectory(env, policy, max_trajectory_len, seed=None)
    for i, x in enumerate(batch[0]):
        assert x.shape == traj[i].shape


def test_generate_batch_independence(env, policy, n_episodes, max_trajectory_len, rng, n_jobs):
    # Clone rng for "what if" scenario (just deepcopying the rng does not work!)
    seed_clone = rng.bit_generator.seed_seq.entropy
    rng_clone = np.random.default_rng(seed_clone)

    seq_batch = generate_batch(env, policy, n_episodes, max_trajectory_len, rng, parallel=False)
    seq_states_1, _, _, _ = seq_batch[0]
    seq_states_2, _, _, _ = seq_batch[1]

    # What if we parallelized instead?
    par_batch = generate_batch(env, policy, n_episodes, max_trajectory_len, rng_clone, parallel=True, n_jobs=n_jobs)
    par_states_1, _, _, _ = par_batch[0]
    par_states_2, _, _, _ = par_batch[1]

    assert len(seq_batch) == n_episodes
    assert not np.allclose(seq_states_1, seq_states_2)

    assert len(par_batch) == n_episodes
    assert not np.allclose(par_states_1, par_states_2)

    # What if: being seeded, results should be exactly the same in the two cases
    assert np.allclose(seq_states_1, par_states_1)
    assert np.allclose(seq_states_2, par_states_2)


def test_blackbox_simulate_batch(env_stochastic_reward, policy, n_episodes, max_trajectory_len, rng, discount, n_jobs,
                                 horizon):
    # Clone rng for "what if" scenario (just deepcopying the rng does not work!)
    seed_clone = rng.bit_generator.seed_seq.entropy
    rng_clone = np.random.default_rng(seed_clone)

    seq_batch = blackbox_simulate_batch(env_stochastic_reward, policy, n_episodes, max_trajectory_len, rng, discount,
                                        parallel=False)
    seq_ret_1, seq_t_1 = seq_batch[0]
    seq_ret_2, seq_t_2 = seq_batch[1]

    # What if we parallelized instead?
    par_batch = blackbox_simulate_batch(env_stochastic_reward, policy, n_episodes, max_trajectory_len, rng_clone,
                                        discount, parallel=True, n_jobs=n_jobs)
    par_ret_1, par_t_1 = par_batch[0]
    par_ret_2, par_t_2 = par_batch[1]

    assert len(seq_batch) == n_episodes
    assert not np.isclose(seq_ret_1, seq_ret_2)
    assert seq_t_1 == horizon
    assert seq_t_2 == horizon

    assert len(par_batch) == n_episodes
    assert not np.isclose(par_ret_1, par_ret_2)
    assert par_t_1 == horizon
    assert par_t_2 == horizon

    # What if: being seeded, results should be exactly the same in the two cases
    assert np.allclose(seq_ret_1, par_ret_1)
    assert np.allclose(seq_ret_2, par_ret_2)
