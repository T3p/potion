from potion.simulation.trajectory_generators import (generate_trajectory,
                                                     blackbox_simulate_episode,
                                                     generate_batch,
                                                     blackbox_simulate_batch,
                                                     unpack,
                                                     apply_mask,
                                                     apply_discount,
                                                     estimate_average_return)
import numpy as np
import pytest

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
    b = generate_batch(env, policy, n_episodes=1, max_trajectory_len=max_trajectory_len, rng=rng, parallel=False)
    traj = generate_trajectory(env, policy, max_trajectory_len, seed=None)
    for i, x in enumerate(b[0]):
        assert x.shape == traj[i].shape

    # Parallel
    b = generate_batch(env, policy, n_episodes=1, max_trajectory_len=max_trajectory_len, rng=rng, parallel=True,
                       n_jobs=n_jobs)
    traj = generate_trajectory(env, policy, max_trajectory_len, seed=None)
    for i, x in enumerate(b[0]):
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


def test_unpack(env, policy, max_trajectory_len, rng, state_d, action_d):
    b = generate_batch(env, policy, n_episodes=7, max_trajectory_len=max_trajectory_len, rng=rng, parallel=False)
    states, actions, rewards, alive = unpack(b)

    assert states.shape == (7, max_trajectory_len, state_d)
    assert actions.shape == (7, max_trajectory_len, action_d)
    assert rewards.shape == (7, max_trajectory_len)
    assert alive.shape == (7, max_trajectory_len)


def test_unpack_exceptions():
    with pytest.raises(ValueError):
        _ = unpack(np.ones((2, 2, 2)))
    with pytest.raises(ValueError):
        _ = unpack([np.ones((2, 2))])
    with pytest.raises(ValueError):
        _ = unpack([(np.ones((2, 2)), )])


def test_apply_mask():
    data_1 = 2. * np.ones((2, 5, 3))
    mask = np.array([True] * 3 + [False] * 2)
    mask = np.stack((mask, mask))
    filtered_1 = apply_mask(data_1, mask)
    data_2 = 2 * np.ones((2, 5))
    filtered_2 = apply_mask(data_2, mask)

    assert np.allclose(filtered_1[:, 0:3, :], 2.)
    assert np.allclose(filtered_1[:, 3:5, :], 0.)
    assert np.allclose(filtered_2[:, 0:3], 2.)
    assert np.allclose(filtered_2[:, 3:5], 0.)


def test_apply_mask_exceptions():
    with pytest.raises(ValueError):
        _ = apply_mask(np.ones((2, 2)), np.ones(3))
    with pytest.raises(ValueError):
        _ = apply_mask(np.ones(2), np.ones(3))


def test_apply_discount(max_trajectory_len, discount, rng):
    n_traj = 2
    rewards = rng.normal(size=(n_traj, max_trajectory_len))
    discounted = apply_discount(rewards, discount)
    correct = np.zeros_like(rewards)
    for i in range(n_traj):
        for j in range(max_trajectory_len):
            correct[i][j] = rewards[i][j] * discount ** j

    assert np.allclose(discounted, correct)


def test_apply_discount_exceptions():
    with pytest.raises(ValueError):
        _ = apply_discount(np.ones(2), -0.1)
    with pytest.raises(ValueError):
        _ = apply_discount(np.ones(2), 1.1)


def test_estimate_average_return(env, env_stochastic_reward, policy, n_episodes, max_trajectory_len, rng, discount,
                                 n_jobs, horizon):
    ret = estimate_average_return(env, policy, n_episodes, max_trajectory_len, rng, discount)
    ret_1 = estimate_average_return(env_stochastic_reward, policy, n_episodes, max_trajectory_len, rng, discount,
                                    parallel=False)
    ret_2 = estimate_average_return(env_stochastic_reward, policy, n_episodes, max_trajectory_len, rng, discount,
                                    parallel=True)

    assert np.isclose(ret, - (1 - discount**horizon) / (1 - discount))
    assert np.isclose(ret_1, ret_2)
