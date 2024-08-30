import numpy as np
from potion.simulation.trajectory_generators import unpack, apply_mask, apply_discount
import warnings
from potion.policies.wrappers import Staged


def reinforce_estimator(batch, discount, policy, baseline="average", average=True):
    if baseline not in ["average", "peters", "zero", None]:
        warnings.warn("Unknown baseline type, will default to zero baseline", UserWarning)

    states, actions, rewards, alive = unpack(batch)  # NxHxS, NxHxA, NxH, NxH

    if not states.shape[-1] == policy.state_dim:
        raise ValueError("Bad shape: state dimension does not match that of given policy")
    if not actions.shape[-1] == policy.action_dim:
        raise ValueError("Bad shape: action dimension does not match that of given policy")

    scores = policy.score(states, actions)  # NxHxd
    scores = apply_mask(scores, alive)
    cum_scores = np.sum(scores, 1)  # Nxm
    rewards = apply_mask(rewards, alive)
    disc_rewards = apply_discount(rewards, discount)  # NxH
    returns = np.sum(disc_rewards, 1)  # N

    if baseline == 'average':
        baseline = np.mean(returns, 0, keepdims=True)[..., None]  # Nx1
    elif baseline == 'peters':
        baseline = np.mean(cum_scores ** 2 * returns[..., None], 0) / np.mean(cum_scores ** 2, 0,
                                                                              keepdims=True)  # Nxd
    else:
        baseline = np.zeros((1, 1))  # 1x1
    baseline[baseline != baseline] = 0.  # replaces nan with zero
    values = returns[..., None] - baseline  # Nxd or Nx1

    grad_samples = cum_scores * values  # Nxd
    if average:
        return np.mean(grad_samples, 0)  # d
    else:
        return grad_samples  # Nxd


def gpomdp_estimator(batch, discount, policy, baseline='average', average=True):
    if baseline not in ["average", "peters", "zero", None]:
        warnings.warn("Unknown baseline type, will default to zero baseline", UserWarning)

    states, actions, rewards, alive = unpack(batch)  # NxHxS, NxHxA, NxH, NxH

    if not states.shape[-1] == policy.state_dim:
        raise ValueError("Bad shape: state dimension does not match that of given policy")
    if not actions.shape[-1] == policy.action_dim:
        raise ValueError("Bad shape: action dimension does not match that of given policy")

    scores = policy.score(states, actions)  # NxHxd
    cum_scores = np.cumsum(scores, 1)  # NxHxd
    cum_scores = apply_mask(cum_scores, alive)
    disc_rewards = apply_discount(rewards, discount)  # NxH
    n_k = np.sum(alive, axis=0, keepdims=True)  # NxH
    n_k[n_k == 0] = 1

    if baseline == 'average':
        baseline = (np.sum(disc_rewards, axis=0, keepdims=True) / n_k)[..., None]  # NxHx1
    elif baseline == 'peters':
        denominator = np.sum(cum_scores ** 2, axis=0, keepdims=True)
        denominator[np.isclose(denominator, 0)] = 1.
        baseline = np.sum(cum_scores ** 2 * disc_rewards[..., None], axis=0, keepdims=True) / denominator  # NxHxd
    else:
        baseline = np.zeros((1, 1, 1))  # 1x1x1
    values = disc_rewards[..., None] - baseline  # NxHxd or NxHx1

    grad_samples = np.sum(cum_scores * values, axis=1)  # Nxd
    if average:
        return np.mean(grad_samples, axis=0)  # d
    else:
        return grad_samples


def nonstationary_pg_estimator(batch, discount, policy, baseline="average", average=True):
    if baseline not in ["average", "peters", "zero", None]:
        warnings.warn("Unknown baseline type, will default to zero baseline", UserWarning)

    states, actions, rewards, alive = unpack(batch)  # NxHxS, NxHxA, NxH, NxH

    if not states.shape[-1] == policy.state_dim:
        raise ValueError("Bad shape: state dimension does not match that of given policy")
    if not actions.shape[-1] == policy.action_dim:
        raise ValueError("Bad shape: action dimension does not match that of given policy")

    scores = policy.score(states, actions)  # NxHxd
    scores = apply_mask(scores, alive)  # NxHxd
    rewards = apply_mask(rewards, alive)
    disc_rewards = apply_discount(rewards, discount)  # NxH
    returns_to_go = np.cumsum(disc_rewards[:, ::-1], 1)[:, ::-1]  # NxH
    #print(states[0], actions[0], scores[0], returns_to_go[0])

    if baseline == 'average':
        baseline = np.mean(returns_to_go, 0, keepdims=True)[..., None]  # NxHx1
    elif baseline == 'peters':
        baseline = np.mean(scores ** 2 * returns_to_go[..., None], 0) / np.mean(scores ** 2, 0,
                                                                                keepdims=True)  # NxHxd
    else:
        baseline = np.zeros((1, 1, 1))  # 1x1x1
    baseline[baseline != baseline] = 0.  # replaces nan with zero
    values = returns_to_go[..., None] - baseline  # NxHxd or NxHx1

    grad_samples = scores * values
    grad_samples = np.reshape(grad_samples, (grad_samples.shape[0], -1))

    if average:
        return np.mean(grad_samples, 0)  # Hd
    else:
        return grad_samples  # NxHd
