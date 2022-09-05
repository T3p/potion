#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Policy gradient estimators
@author: Matteo Papini

N.B.: baselines are not univocally defined for variable-length trajectories.
We use here the absorbing state convention: for steps after the end of the
episode, all rewards are zero and all action probabilities are one regardless
of policy parameters. Under this convention, all steps of all trajectories 
contribute to baseline computation.
"""

import torch
import potion.common.torch_utils as tu
from potion.common.misc_utils import unpack, discount
from potion.common.torch_utils import tensormat, jacobian
from potion.estimation.moments import incr_mean, incr_var


def gpomdp_estimator(batch, disc, policy, baselinekind='avg', result='mean',
                     shallow=False):
    """G(PO)MDP policy gradient estimator
       
    batch: list of N trajectories. Each trajectory is a tuple 
        (states, actions, rewards, mask). Each element of the tuple is a 
        tensor where the first dimension is time.
    disc: discount factor
    policy: the one used to collect the data
    baselinekind: kind of baseline to employ in the estimator. 
        Either 'avg' (average reward, default), 'peters' 
        (variance-minimizing),  or 'zero' (no baseline)
    result: whether to return the final estimate ('mean', default), or the 
        single per-trajectory estimates ('samples')
    shallow: whether to use precomputed score functions (only available
        for shallow policies)
    """
    if shallow:
        return _shallow_gpomdp_estimator(batch, disc, policy, baselinekind, result)
    
    N = len(batch)
    states, actions, rewards, mask, _ = unpack(batch) #NxHxd_s, NxHxd_a, NxH, NxH
    H = rewards.shape[1]
    m = policy.num_params()
    
    disc_rewards = discount(rewards * mask, disc) #NxH
    logps = policy.log_pdf(states, actions) * mask #NxH
    cm_logps = torch.cumsum(logps, 1) #NxH
    
    if baselinekind == 'peters':
        jac = jacobian(policy, cm_logps.view(-1)).reshape((N,H,m)) #NxHxm   
        b_num = torch.sum(tensormat(jac**2, 
                                    disc_rewards), 0) #Hxm
        b_den = torch.sum(jac**2, 0) #Hxm
        baseline = b_num / b_den #Hxm
        baseline[baseline != baseline] = 0
        values = disc_rewards.unsqueeze(2) - baseline.unsqueeze(0) #NxHxm
        _samples = torch.sum(values * jac, 1) #Nxm
    else:
        if baselinekind == 'avg':
            baseline = torch.mean(disc_rewards, 0) #H
        else:
            baseline = torch.zeros(1) #1
        values = (disc_rewards - baseline) #NxH
        
        _samples = torch.stack([tu.flat_gradients(policy, cm_logps[i,:], 
                                              values[i,:])
                                       for i in range(N)], 0) #Nxm
    if result == 'samples':
        return _samples #Nxm
    else:
        return torch.mean(_samples, 0) #m

def reinforce_estimator(batch, disc, policy, baselinekind='avg', 
                        result='mean', shallow=False):
    """REINFORCE policy gradient estimator
       
    batch: list of N trajectories. Each trajectory is a tuple 
        (states, actions, rewards, mask). Each element of the tuple is a 
        tensor where the first dimension is time.
    disc: discount factor
    policy: the one used to collect the data
    baselinekind: kind of baseline to employ in the estimator. 
        Either 'avg' (average reward, default), 'peters' 
        (variance-minimizing),  or 'zero' (no baseline)
    result: whether to return the final estimate ('mean', default), or the 
        single per-trajectory estimates ('samples')
    shallow: whether to use precomputed score functions (only available
        for shallow policies)
    """
    if shallow:
        return _shallow_reinforce_estimator(batch, disc, policy, baselinekind, result)
    
    N = len(batch)
    states, actions, rewards, mask, _ = unpack(batch) #NxHxm, NxHxd, NxH, NxH
    
    disc_rewards = discount(rewards * mask, disc) #NxH
    rets = torch.sum(disc_rewards, 1) #N
    logps = policy.log_pdf(states, actions) * mask #NxH
    
    if baselinekind == 'peters':
        logp_sums = torch.sum(logps, 1) #N
        jac = jacobian(policy, logp_sums) #Nxm   
        b_num = torch.sum(jac ** 2 * rets.unsqueeze(1), 0) #m
        b_den = torch.sum(jac **2, 0) #m
        baseline = b_num / b_den #m
        baseline[baseline != baseline] = 0
        values = rets.unsqueeze(1) - baseline.unsqueeze(0) #Nxm
        _samples = jac * values
    else:
        if baselinekind == 'avg':
            baseline = torch.mean(rets, 0) #1
        else:
            baseline = torch.zeros(1) #1
        baseline[baseline != baseline] = 0
        values = rets - baseline #N
        logp_sums = torch.sum(logps, 1)
        if result == 'mean':
            return tu.flat_gradients(policy, logp_sums, values) / N
        _samples = torch.stack([tu.flat_gradients(policy, logp_sums[i])
                                * values[i]
                                for i in range(N)], 0) #Nxm
    if result == 'samples':
        return _samples #Nxm
    else:
        return torch.mean(_samples, 0) #m

def _shallow_gpomdp_estimator(batch, disc, policy, baselinekind='peters', result='mean'):
    with torch.no_grad():        
        states, actions, rewards, mask, _ = unpack(batch) # NxHxm, NxHxd, NxH, NxH
        
        disc_rewards = discount(rewards * mask, disc) #NxH
        scores = policy.score(states, actions) * mask.unsqueeze(-1) #NxHxM
        G = torch.cumsum(scores, 1) #NxHxm
        
        if baselinekind == 'avg':
            baseline = torch.mean(disc_rewards, 0).unsqueeze(-1) #Hx1
        elif baselinekind == 'peters':
            baseline = torch.sum(tensormat(G ** 2, disc_rewards), 0) / \
                            torch.sum(G ** 2, 0) #Hxm
        elif baselinekind == 'peters2':
            G2 = torch.cumsum(scores ** 2, 1) #NxHxm
            baseline = torch.sum(tensormat(G ** 2, disc_rewards), 0) / \
                            torch.sum(G2, 0) #Hxm
        elif baselinekind == 'optimal':
            g = scores #NxHxm
            vanilla = tensormat(G, disc_rewards) #NxHxm
            term1 = torch.flip(torch.cumsum(torch.flip(vanilla, (1,)), 1), (1,)) #NxHxm (reverse cumsum)
            m2 = torch.mean(g ** 2, dim=0, keepdim=True).unsqueeze(0).unsqueeze(-1) #(N)xHxm
            scores_on_m2 = g / m2 #NxHxm
            scores_on_m2[g==0] = 0
            shifted = torch.zeros_like(scores_on_m2) #NxHxm
            shifted[:,:-1,:] = scores_on_m2[:,1:,:]
            term2 = scores_on_m2 - shifted #NxHxm
            baseline = torch.mean(term1 * term2, dim=0) #Hxm
        elif baselinekind == 'system':
            A = torch.mean(torch.einsum('ijk,ihk->ijhk', (G, G)), 0).transpose(0,2) #mxHxH
            vanilla = torch.sum(tensormat(G, disc_rewards), dim=1, keepdim=True) #Nx(H)xm
            c = torch.mean(vanilla * G, 0).transpose(0,1).unsqueeze(-1) #Hxmx1
            #print(scores[0,:,1], G[0,:,1])
            x, _ = torch.solve(c, A) #mxHx1
            baseline = x.squeeze(-1).transpose(0,1)
        elif baselinekind == 'zero':
            baseline = torch.zeros(1,1) #1x1
        else:
            raise ValueError('Unknown baseline')
        baseline[baseline != baseline] = 0 #removes non-real values
        values = disc_rewards.unsqueeze(2) - baseline.unsqueeze(0) #NxHxm
        
        _samples = torch.sum(G * values, 1) #Nxm
        if result == 'samples':
            return _samples #Nxm
        else:
            return torch.mean(_samples, 0) #m
        
def _shallow_reinforce_estimator(batch, disc, policy, baselinekind='peters', result='mean'):
    with torch.no_grad():        
        states, actions, rewards, mask, _ = unpack(batch) #NxHxm, NxHxd, NxH, NxH
        
        scores = policy.score(states, actions) * mask.unsqueeze(-1) #NxHxm
        G = torch.sum(scores, 1) #Nxm
        disc_rewards = discount(rewards * mask, disc) #NxH
        rets = torch.sum(disc_rewards, 1) #N
        
        if baselinekind == 'avg':
            baseline = torch.mean(rets, 0) #1
        elif baselinekind == 'peters':
            baseline = torch.mean(G ** 2 * rets.unsqueeze(1), 0) /\
                torch.mean(G ** 2, 0) #m
        else:
            baseline = torch.zeros(1) #1
        baseline[baseline != baseline] = 0
        values = rets.unsqueeze(1) - baseline.unsqueeze(0) #Nxm
        
        _samples = G * values #Nxm
        if result == 'samples':
            return _samples #Nxm
        else:
            return torch.mean(_samples, 0) #m
        
"""Testing"""
if __name__ == '__main__':
    from potion.actors.continuous_policies import ShallowGaussianPolicy as Gauss
    from potion.simulation.trajectory_generators import generate_batch
    from potion.common.misc_utils import seed_all_agent
    import potion.envs
    import gym.spaces
    env = gym.make('ContCartPole-v0')
    env.seed(0)
    seed_all_agent(0)
    N = 100
    H = 100
    disc = 0.99
    pol = Gauss(4,1, mu_init=[0.,0.,0.,0.], learn_std=True)
    TOL = 1e-4
    batch = generate_batch(env, pol, H, N)
    
    #Just checking that shallow and deep implementations are consistent
    for b, res in zip(["zero", "avg", "peters"], ["mean", "samples"]):
        deep = gpomdp_estimator(batch, disc, pol, baselinekind=b, result=res,
                         shallow=False)
        shallow = gpomdp_estimator(batch, disc, pol, baselinekind=b, result=res,
                         shallow=True)
        assert torch.allclose(deep, shallow, atol=TOL)
        deep = reinforce_estimator(batch, disc, pol, baselinekind=b, result=res,
                         shallow=False)
        shallow = reinforce_estimator(batch, disc, pol, baselinekind=b, result=res,
                         shallow=True)
        assert torch.allclose(deep, shallow, atol=TOL)    