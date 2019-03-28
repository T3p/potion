#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Policy gradient stimators
@author: Matteo Papini
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
    
    disc_rewards = discount(rewards, disc) #NxH
    logps = policy.log_pdf(states, actions) * mask #NxH
    cum_logps = torch.cumsum(logps, 1) #NxH
    
    if baselinekind == 'peters':
        jac = jacobian(policy, cum_logps.view(-1)).reshape((N,H,m)) #NxHxm   
        b_num = torch.sum(tensormat(jac**2, 
                                    disc_rewards), 0) #Hxm
        b_den = torch.sum(jac**2, 0) #Hxm
        baseline = b_num / b_den #Hxm
        baseline[baseline != baseline] = 0
        values = disc_rewards.unsqueeze(2) - baseline.unsqueeze(0) #NxHxm
        _samples = torch.sum(tensormat(values * jac, mask), 1) #Nxm
    else:
        if baselinekind == 'avg':
            baseline = torch.mean(disc_rewards, 0) #H
        else:
            baseline = torch.zeros(1) #1
        values = (disc_rewards - baseline) * mask #NxH
        
        _samples = torch.stack([tu.flat_gradients(policy, cum_logps[i,:], 
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
    
    disc_rewards = discount(rewards, disc) #NxH
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
        
        if result == 'mean':
            logp_sums = torch.sum(logps, 1)
            return tu.flat_gradients(policy, logp_sums, values) / N
        
        _samples = torch.stack([tu.flat_gradients(policy, logps[i,:]) * 
                                                     values[i,:]
                                       for i in range(N)], 0) #Nxm
    if result == 'samples':
        return _samples #Nxm
    else:
        return torch.mean(_samples, 0) #m

def _shallow_gpomdp_estimator(batch, disc, policy, baselinekind='peters', result='mean'):
    with torch.no_grad():        
        states, actions, rewards, mask, _ = unpack(batch) # NxHxm, NxHxd, NxH, NxH
        
        disc_rewards = discount(rewards, disc) #NxH
        scores = policy.score(states, actions) #NxHxM
        G = torch.cumsum(tensormat(scores, mask), 1) #NxHxm
        
        if baselinekind == 'avg':
            baseline = torch.mean(disc_rewards, 0).unsqueeze(1) #Hx1
        elif baselinekind == 'peters':
            baseline = torch.sum(tensormat(G ** 2, disc_rewards), 0) / \
                            torch.sum(G ** 2, 0) #Hxm
        else:
            baseline = torch.zeros(1,1) #1x1
        baseline[baseline != baseline] = 0
        values = disc_rewards.unsqueeze(2) - baseline.unsqueeze(0) #NxHxm
        
        _samples = torch.sum(tensormat(G * values, mask), 1) #Nxm
        if result == 'samples':
            return _samples #Nxm
        else:
            return torch.mean(_samples, 0) #m
        
def _shallow_reinforce_estimator(batch, disc, policy, baselinekind='peters', result='mean'):
    with torch.no_grad():        
        states, actions, rewards, mask, _ = unpack(batch) #NxHxm, NxHxd, NxH, NxH
        
        scores = policy.score(states, actions) #NxHxm
        scores = tensormat(scores, mask) #NxHxm
        G = torch.sum(scores, 1) #Nxm
        disc_rewards = discount(rewards, disc) #NxH
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

def _incr_shallow_gpomdp_estimator(traj, disc, policy, baselinekind='peters', result='mean', cum_1 = 0., cum_2 = 0., cum_3 = 0., tot_trajs = 1):
    with torch.no_grad():
        states, actions, rewards, mask, _ = traj #Hxm, Hxd, H, H
    
        disc_rewards = discount(rewards, disc) #H
        scores = policy.score(states, actions).squeeze() #Hxm
        G = torch.cumsum(scores * mask.unsqueeze(1), 0) #Hxm
        if baselinekind == 'avg':
            baseline = incr_mean(cum_2, disc_rewards, tot_trajs) #H
            res_2 = baseline
            res_3 = 0.
            values = (disc_rewards - baseline).unsqueeze(1)
        elif baselinekind == 'peters':
            num = incr_mean(cum_2, G**2 * disc_rewards.unsqueeze(1), tot_trajs) #Hxm
            den = incr_mean(cum_3, G**2, tot_trajs) #Hxm
            baseline = num / den #Hxm
            res_2 = num
            res_3 = den
            baseline[baseline!=baseline] = 0
            values = disc_rewards.unsqueeze(1) - baseline
        else:
            values = disc_rewards.unsqueeze(1)
            res_2 = 0.
            res_3 = 0.
        
        _sample = torch.sum(G * values * mask.unsqueeze(1), 0)
        if result == 'samples':
            return _sample, res_2, res_3
        else:
            return incr_mean(cum_1, _sample, tot_trajs), res_2, res_3
        
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
    
    batch = generate_batch(env, pol, H, N)
    
    o = gpomdp_estimator(batch, disc, pol, baselinekind='peters', 
                         shallow=True)
    print('Shallow GPOMDP (peters):', o)
    #o = gpomdp_estimator(batch, disc, pol, baselinekind='peters')
    #print('GPOMDP (peters)', o)
    #print()
    
    print('Cum version')
    cum_1 = cum_2 = cum_3 = 0.
    i = 0
    for t in batch:
        i+=1
        cum_1, cum_2, cum3 = _incr_shallow_gpomdp_estimator(t, disc, pol, 'peters', 'mean', cum_1, cum_2, cum_3, i)
        #print(cum_1, cum_2, cum_3)
    o = cum_1
    print(o)
    print()
    
    """
  
    o = gpomdp_estimator(batch, disc, pol, baselinekind='avg', 
                         shallow=True)
    print('Shallow GPOMDP (avg):', o)
    o = gpomdp_estimator(batch, disc, pol, baselinekind='avg')
    print('GPOMDP (avg)', o)
    print()
    
    o = gpomdp_estimator(batch, disc, pol, baselinekind='zero', 
                         shallow=True)
    print('Shallow GPOMDP (zero):', o)
    o = gpomdp_estimator(batch, disc, pol, baselinekind='zero')
    print('GPOMDP (zero)', o)
    print()
    
    o = reinforce_estimator(batch, disc, pol, baselinekind='peters', 
                            shallow=True)
    print('Shallow REINFORCE (peters):', o)
    o = reinforce_estimator(batch, disc, pol, baselinekind='peters')
    print('REINFORCE (peters):', o)
    print()
    
    o = reinforce_estimator(batch, disc, pol, baselinekind='avg', 
                            shallow=True)
    print('Shallow REINFORCE (avg):', o)
    o = reinforce_estimator(batch, disc, pol, baselinekind='avg')
    print('REINFORCE (avg):', o)
    print()
    
    o = reinforce_estimator(batch, disc, pol, baselinekind='zero',
                            shallow=True)
    print('Shallow REINFORCE (zero):', o)
    o = reinforce_estimator(batch, disc, pol, baselinekind='zero')
    print('REINFORCE (zero):', o)
    #"""
    