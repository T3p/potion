#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 16:18:45 2019

@author: matteo
"""
from potion.simulation.trajectory_generators import generate_batch
from potion.common.misc_utils import performance, avg_horizon
from potion.estimation.gradients import gpomdp_estimator
from potion.estimation.metagradients import metagrad
from potion.common.logger import Logger
from potion.common.misc_utils import clip, seed_all_agent, mean_sum_info, max_reward, max_feature
from potion.actors.continuous_policies import ShallowGaussianPolicy
from potion.meta.smoothing_constants import gauss_lip_const, std_lip_const
from potion.meta.safety_requirements import MonotonicImprovement
import scipy.stats as sts
from scipy.sparse.linalg import eigsh
import torch
import math

def sepg(env, policy, 
            horizon,
            batchsize = 100, 
            iterations = 200,
            disc = 0.99,
            max_feat = 1.,
            max_rew = 1.,
            safety_req = MonotonicImprovement(0.),
            conf = 0.2,
            adapt_batchsize = False,
            test_batchsize = False,
            render = False,
            seed = None,
            action_filter = None,
            parallel = False,
            logger = Logger(name='SEPG'),
            info_key = 'danger',
            save_params = 50,
            log_params = True,
            verbose = True,
            emp = False):
    """
        SEPG algorithm
        Only for shallow Gaussian policy w/ scalar variance
    """
        
    #Defaults
    assert type(policy) == ShallowGaussianPolicy
    assert policy.learn_std
    if action_filter is None:
        action_filter = clip(env)
    
    #Seed agent
    if seed is not None:
        seed_all_agent(seed)
    
    #Prepare logger
    algo_info = {'Algorithm': 'MEPG', 
                 'Environment': str(env), 
                 'BatchSize': batchsize, 
                 'Horizon': horizon,
                 'Iterations': iterations,
                 'Disc': disc, 
                 'Seed': seed,
                 'ActionFilter': action_filter}
    logger.write_info({**algo_info, **policy.info()})
    log_keys = ['Perf', 'UPerf', 'AvgHorizon', 
                'StepSize', 'MetaStepSize', 'BatchSize', 'Exploration', 
                'OmegaGrad', 'OmegaMetagrad', 'UpsilonGradNorm',
                'UpsilonGradVar', 'UpsilonEps', 'OmegaGradVar', 'OmegaEps',
                'Req', 'MinBatchSize', 'MaxReq', 'Info', 'MaxRew', 'MaxFeat']
    if log_params:
        log_keys += ['param%d' % i for i in range(policy.num_params())]
    if test_batchsize:
        log_keys.append('DetPerf')
    log_row = dict.fromkeys(log_keys)

    logger.open(log_row.keys())
    
    #Learning loop
    it = 0
    stepsize = 0.
    metastepsize = 0.
    omega_grad_var = 0.
    omega_eps = 0.
    omega_metagrad = torch.zeros(1)
    while(it < iterations):
        #Begin iteration
        if verbose:
            print('\nIteration ', it)
        if verbose:
            print('Params: ', policy.get_flat())
    
        #Test mean parameters on deterministic policy
        if test_batchsize:
            test_batch = generate_batch(env, policy, horizon, test_batchsize, 
                                        action_filter=action_filter,
                                        seed=seed,
                                        njobs=parallel,
                                        deterministic=True,
                                        key=info_key)
            log_row['DetPerf'] = performance(test_batch, disc)
        #Render behavior
        if render:
            generate_batch(env, policy, horizon, 1, action_filter, render=True, key=info_key)

        #Set metaparameters
        omega = policy.get_scale_params()
        sigma = torch.exp(omega)
        
        #Collect trajectories
        batch = generate_batch(env, policy, horizon, batchsize, 
                               action_filter=action_filter, 
                               seed=seed, 
                               n_jobs=parallel,
                               key=info_key)
        perf = performance(batch, disc)
        H = avg_horizon(batch)
        
        if emp:
            max_rew = max_reward(batch)
            max_feat = max_feature(batch)
        
        #Estimate policy gradient
        grad_samples = gpomdp_estimator(batch, disc, policy, 
                                    baselinekind='peters', 
                                    shallow=True,
                                    result='samples')
        grad = torch.mean(grad_samples, 0)
        upsilon_grad = grad[1:]
        upsilon_grad_norm = torch.norm(upsilon_grad)
        omega_grad = grad[0]        
        dfn = upsilon_grad.shape[0]
        
        ### Mean-update iteration
        if it % 2 == 0:
            #Compute gradient estimation error for mean parameters
            if conf < 1 and grad_samples.size()[1] > 2:
                centered = grad_samples[:, 1:] - upsilon_grad.unsqueeze(0)
                grad_cov = batchsize/(batchsize - 1) * torch.mean(torch.bmm(centered.unsqueeze(2), centered.unsqueeze(1)),0)
                upsilon_grad_var = torch.sum(torch.diag(grad_cov)).item()
                max_eigv = eigsh(grad_cov.numpy(), 1)[0][0]
                quant = sts.f.ppf(1 - conf, dfn, batchsize - dfn)
                upsilon_eps = math.sqrt(max_eigv * dfn * quant)
            elif conf < 1:
                upsilon_grad_var = torch.var(grad_samples[:, 1]).item()
                quant = sts.t.ppf(1 - conf/2, batchsize - 1)
                upsilon_eps = quant * math.sqrt(upsilon_grad_var)
            else:
                upsilon_eps = 0.
                upsilon_grad_var = 0.
        
            #Compute safe step size for mean parameters
            req = safety_req.next(perf)
            F = gauss_lip_const(max_feat, max_rew, disc, std=1.) * (1 - disc**H)
            delta = torch.clamp(upsilon_grad_norm - upsilon_eps, min=0, max=float('inf'))
            max_req = sigma**2 * \
                        (delta / math.sqrt((batchsize - dfn)))**2 / \
                        (2 * F)
            req = min(req, max_req)
            if max_req != 0:
                alpha = (delta / math.sqrt((batchsize - dfn)))  / \
                            F * \
                            (1 + math.sqrt(1 - req / max_req))
            else:
                alpha = 0.
            stepsize = (alpha * sigma**2 / upsilon_grad_norm).item()
            
            #Ensure minimum safe batchsize
            min_batchsize = math.ceil((upsilon_eps**2 / upsilon_grad_norm**2).item() + 1e-12) + dfn
            if conf < 1 and adapt_batchsize:
                batchsize = max(batchsize, min_batchsize)
            
            #Update mean parameters
            upsilon = policy.get_loc_params()
            new_upsilon = upsilon + alpha * sigma**2 * upsilon_grad / upsilon_grad_norm
            policy.set_loc_params(new_upsilon)
        ###  
        ### Variance-update iteration
        else:
            #Estimate meta gradient (alpha from previous step)
            omega_metagrad = metagrad(batch, disc, policy, alpha, 
                                      grad_samples=grad_samples)
            omega_metagrad_norm = torch.norm(omega_metagrad)
            
            #Compute gradient estimation error for variance parameter
            if conf < 1:
                omega_grad_var = torch.var(grad_samples[:, 0]).item()
                quant = sts.t.ppf(1 - conf/2, batchsize - 1)
                omega_eps = quant * math.sqrt(omega_grad_var / batchsize)
            else:
                omega_grad_var = 0.
                omega_eps = 0.
                
            #Compute safe meta step size
            req = safety_req.next(perf)
            G = std_lip_const(max_rew, disc) * (1 - disc**H)
            proj = omega_grad.view(-1).dot(omega_metagrad.view(-1)) / torch.norm(omega_metagrad)
            delta = torch.clamp(torch.abs(proj) - omega_eps, min=0, max=float('inf'))
            max_req = (delta / math.sqrt(batchsize))**2 / (2 * G)
            req = min(req, max_req)
            if max_req != 0:
                eta = (delta / math.sqrt(batchsize)) / \
                        G * \
                        (torch.sign(proj) + torch.sqrt(1 - req / max_req))
            else:
                eta = 0.
            metastepsize = (eta / omega_metagrad_norm).item()
            
            #Ensure minimum safe batchsize
            min_batchsize = math.ceil((omega_eps**2 / proj**2).item() + 1e-12)
            if conf < 1 and adapt_batchsize:
                batchsize = max(batchsize, min_batchsize)
        
            #Update variance parameters
            new_omega = omega + eta * omega_metagrad / omega_metagrad_norm
            policy.set_scale_params(new_omega)

        # Log
        log_row['Exploration'] = sigma.item()
        log_row['StepSize'] = stepsize
        log_row['MetaStepSize'] = metastepsize
        log_row['OmegaGrad'] = omega_grad.item()
        log_row['OmegaMetagrad'] = omega_metagrad.item()
        log_row['UpsilonGradNorm'] = torch.norm(upsilon_grad).item()
        log_row['BatchSize'] = batchsize
        log_row['Perf'] = perf
        log_row['UPerf'] = performance(batch, 1.)
        log_row['AvgHorizon'] = avg_horizon(batch)
        log_row['UpsilonGradVar'] = upsilon_grad_var
        log_row['UpsilonEps'] = upsilon_eps
        log_row['OmegaGradVar'] = upsilon_grad_var
        log_row['OmegaEps'] = upsilon_eps
        log_row['Req'] = req
        log_row['Info'] = mean_sum_info(batch).item()
        print(min_batchsize)
        log_row['MinBatchSize'] = min_batchsize
        log_row['MaxReq'] = max_req.item()
        log_row['MaxRew'] = max_rew
        log_row['MaxFeat'] = max_feat
        params = policy.get_flat()
        if log_params:
            for i in range(policy.num_params()):
                log_row['param%d' % i] = params[i].item()
        logger.write_row(log_row, it)
        if save_params and it % save_params == 0:
            logger.save_params(params, it)
        
        # Next iteration
        it += 1
    
    # Final policy
    if save_params:
        logger.save_params(params, it)


def naive_sepg(env, policy, 
            horizon,
            batchsize = 100, 
            iterations = 200,
            disc = 0.99,
            max_feat = 1.,
            max_rew = 1.,
            safety_req = MonotonicImprovement(0.),
            conf = 0.2,
            adapt_batchsize = False,
            test_batchsize = False,
            render = False,
            seed = None,
            action_filter = None,
            parallel = False,
            logger = Logger(name='NSEPG'),
            save_params = 50,
            log_params = True,
            verbose = True):
    """
        SEPG algorithm
        Only for shallow Gaussian policy w/ scalar variance
    """
        
    #Defaults
    assert type(policy) == ShallowGaussianPolicy
    assert policy.learn_std
    if action_filter is None:
        action_filter = clip(env)
    
    #Seed agent
    if seed is not None:
        seed_all_agent(seed)
    
    #Prepare logger
    algo_info = {'Algorithm': 'MEPG', 
                 'Environment': str(env), 
                 'BatchSize': batchsize, 
                 'Horizon': horizon,
                 'Iterations': iterations,
                 'Disc': disc, 
                 'Seed': seed,
                 'ActionFilter': action_filter}
    logger.write_info({**algo_info, **policy.info()})
    log_keys = ['Perf', 'UPerf', 'AvgHorizon', 
                'StepSize', 'MetaStepSize', 'BatchSize', 'Exploration', 
                'OmegaGrad', 'UpsilonGradNorm',
                'UpsilonGradVar', 'UpsilonEps', 'OmegaGradVar', 'OmegaEps',
                'Req', 'MinBatchSize', 'MaxReq', 'MaxRew']
    if log_params:
        log_keys += ['param%d' % i for i in range(policy.num_params())]
    if test_batchsize:
        log_keys.append('DetPerf')
    log_row = dict.fromkeys(log_keys)

    logger.open(log_row.keys())
    
    #Learning loop
    it = 0
    stepsize = 0.
    metastepsize = 0.
    omega_grad_var = 0.
    omega_eps = 0.
    omega_grad = torch.zeros(1)
    while(it < iterations):
        #Begin iteration
        if verbose:
            print('\nIteration ', it)
        if verbose:
            print('Params: ', policy.get_flat())
    
        #Test mean parameters on deterministic policy
        if test_batchsize:
            test_batch = generate_batch(env, policy, horizon, test_batchsize, 
                                        action_filter=action_filter,
                                        seed=seed,
                                        njobs=parallel,
                                        deterministic=True,)
            log_row['DetPerf'] = performance(test_batch, disc)
        #Render behavior
        if render:
            generate_batch(env, policy, horizon, 1, action_filter, render=True)

        #Set metaparameters
        omega = policy.get_scale_params()
        sigma = torch.exp(omega)
        
        #Collect trajectories
        batch = generate_batch(env, policy, horizon, batchsize, 
                               action_filter=action_filter, 
                               seed=seed, 
                               n_jobs=parallel)
        perf = performance(batch, disc)
        
        #Estimate policy gradient
        grad_samples = gpomdp_estimator(batch, disc, policy, 
                                    baselinekind='peters', 
                                    shallow=True,
                                    result='samples')
        grad = torch.mean(grad_samples, 0)
        upsilon_grad = grad[1:]
        upsilon_grad_norm = torch.norm(upsilon_grad)
        omega_grad = grad[0]
        omega_grad_norm = torch.norm(omega_grad)
        dfn = grad.shape[0]
        
        ### Mean-update iteration
        if it % 2 == 0:
            #Compute gradient estimation error for mean parameters
            if conf < 1 and grad_samples.size()[1] > 2:
                centered = grad_samples[:, 1:] - upsilon_grad.unsqueeze(0)
                grad_cov = batchsize/(batchsize - 1) * torch.mean(torch.bmm(centered.unsqueeze(2), centered.unsqueeze(1)),0)
                upsilon_grad_var = torch.sum(torch.diag(grad_cov)).item()
                max_eigv = eigsh(grad_cov.numpy(), 1)[0][0]
                quant = sts.f.ppf(1 - conf, dfn, batchsize - dfn)
                upsilon_eps = math.sqrt(max_eigv * dfn * quant)
            elif conf < 1:
                upsilon_grad_var = torch.var(grad_samples[:, 1]).item()
                quant = sts.t.ppf(1 - conf/2, batchsize - 1)
                upsilon_eps = quant * math.sqrt(upsilon_grad_var)
            else:
                upsilon_eps = 0.
                upsilon_grad_var = 0.
        
            #Compute safe step size for mean parameters
            req = safety_req.next(perf)
            F = gauss_lip_const(max_feat, max_rew, disc, std=1.)
            max_req = sigma**2 * \
                        (upsilon_grad_norm - upsilon_eps / math.sqrt((batchsize - dfn)))**2 / \
                        (2 * F)
            req = min(req, max_req)
            alpha = (upsilon_grad_norm - upsilon_eps / math.sqrt((batchsize - dfn)))  / \
                        F * \
                        (1 + math.sqrt(1 - req / max_req))
            stepsize = (alpha * sigma**2 / upsilon_grad_norm).item()
            
            #Ensure minimum safe batchsize
            min_batchsize = math.ceil((upsilon_eps**2 / upsilon_grad_norm**2).item() + 1e-12) + dfn
            if conf < 1 and adapt_batchsize:
                batchsize = max(batchsize, min_batchsize)
            
            #Update mean parameters
            upsilon = policy.get_loc_params()
            new_upsilon = upsilon + alpha * sigma**2 * upsilon_grad / upsilon_grad_norm
            policy.set_loc_params(new_upsilon)
        ###  
        ### Variance-update iteration
        else:
            #Compute gradient estimation error for variance parameter
            if conf < 1:
                omega_grad_var = torch.var(grad_samples[:, 0]).item()
                quant = sts.t.ppf(1 - conf/2, batchsize - 1)
                omega_eps = quant * math.sqrt(omega_grad_var)
            else:
                omega_grad_var = 0.
                omega_eps = 0.
                
            #Compute safe meta step size
            req = safety_req.next(perf)
            G = std_lip_const(max_rew, disc)
            max_req = (omega_grad_norm - omega_eps / math.sqrt(batchsize))**2 / (2 * G)
            req = min(req, max_req)
            eta = (omega_grad_norm - omega_eps / math.sqrt(batchsize)) / \
                    G * \
                    (1 + torch.sqrt(1 - req / max_req))
            metastepsize = (eta / omega_grad_norm).item()
            
            #Ensure minimum safe batchsize
            min_batchsize = math.ceil((omega_eps**2 / omega_grad_norm**2).item() + 1e-12)
            if conf < 1 and adapt_batchsize:
                batchsize = max(batchsize, min_batchsize)
        
            #Update variance parameters
            new_omega = omega + eta * omega_grad / omega_grad_norm
            policy.set_scale_params(new_omega)

        # Log
        log_row['Exploration'] = sigma.item()
        log_row['StepSize'] = stepsize
        log_row['MetaStepSize'] = metastepsize
        log_row['OmegaGrad'] = omega_grad.item()
        log_row['UpsilonGradNorm'] = torch.norm(upsilon_grad).item()
        log_row['BatchSize'] = batchsize
        log_row['Perf'] = perf
        log_row['UPerf'] = performance(batch, 1.)
        log_row['AvgHorizon'] = avg_horizon(batch)
        log_row['UpsilonGradVar'] = upsilon_grad_var
        log_row['UpsilonEps'] = upsilon_eps
        log_row['OmegaGradVar'] = upsilon_grad_var
        log_row['OmegaEps'] = upsilon_eps
        log_row['Req'] = req
        print(min_batchsize)
        log_row['MinBatchSize'] = min_batchsize
        log_row['MaxReq'] = max_req.item()
        log_row['MaxRew'] = max_rew
        params = policy.get_flat()
        if log_params:
            for i in range(policy.num_params()):
                log_row['param%d' % i] = params[i].item()
        logger.write_row(log_row, it)
        if save_params and it % save_params == 0:
            logger.save_params(params, it)
        
        # Next iteration
        it += 1
    
    # Final policy
    if save_params:
        logger.save_params(params, it)
        
#def ssepg(env, policy, 
#            horizon,
#            batchsize = 100, 
#            iterations = 200,
#            disc = 0.99,
#            pow_alpha = 0.01,
#            pow_err_tol = 0.1,
#            max_pow_it = 100,
#            max_pow_attempts = 3,
#            pow_clip = 0.2,
#            safety_req = MonotonicImprovement(0.),
#            conf = 0.2,
#            adapt_batchsize = False,
#            test_batchsize = False,
#            render = False,
#            seed = None,
#            action_filter = None,
#            parallel = False,
#            logger = Logger(name='SEPG'),
#            save_params = 50,
#            log_params = True,
#            verbose = True):
#    """
#        SSEPG algorithm
#        Only for shallow Gaussian policy w/ scalar variance
#    """
#        
#    #Defaults
#    assert type(policy) == ShallowGaussianPolicy
#    assert policy.learn_std
#    if action_filter is None:
#        action_filter = clip(env)
#    
#    #Seed agent
#    if seed is not None:
#        seed_all_agent(seed)
#    
#    #Prepare logger
#    algo_info = {'Algorithm': 'MEPG', 
#                 'Environment': str(env), 
#                 'BatchSize': batchsize, 
#                 'Horizon': horizon,
#                 'Iterations': iterations,
#                 'Disc': disc, 
#                 'Seed': seed,
#                 'ActionFilter': action_filter}
#    logger.write_info({**algo_info, **policy.info()})
#    log_keys = ['Perf', 'UPerf', 'AvgHorizon', 
#                'StepSize', 'MetaStepSize', 'BatchSize', 'Exploration', 
#                'OmegaGrad', 'OmegaMetagrad', 'UpsilonGradNorm',
#                'UpsilonGradVar', 'UpsilonEps', 'OmegaGradVar', 'OmegaEps',
#                'Req', 'MinBatchSize', 'MaxReq', 'UpsilonLip', 'OmegaLip']
#    if log_params:
#        log_keys += ['param%d' % i for i in range(policy.num_params())]
#    if test_batchsize:
#        log_keys.append('DetPerf')
#    log_row = dict.fromkeys(log_keys)
#
#    logger.open(log_row.keys())
#    
#    #Learning loop
#    it = 0
#    stepsize = 0.
#    metastepsize = 0.
#    omega_grad_var = 0.
#    omega_eps = 0.
#    omega_metagrad = torch.zeros(1)
#    G = 0.
#    while(it < iterations):
#        #Begin iteration
#        if verbose:
#            print('\nIteration ', it)
#        if verbose:
#            print('Params: ', policy.get_flat())
#    
#        #Test mean parameters on deterministic policy
#        if test_batchsize:
#            test_batch = generate_batch(env, policy, horizon, test_batchsize, 
#                                        action_filter=action_filter,
#                                        seed=seed,
#                                        njobs=parallel,
#                                        deterministic=True)
#            log_row['DetPerf'] = performance(test_batch, disc)
#        #Render behavior
#        if render:
#            generate_batch(env, policy, horizon, 1, action_filter, render=True)
#
#        #Set metaparameters
#        omega = policy.get_scale_params()
#        sigma = torch.exp(omega)
#        
#        #Collect trajectories
#        batch = generate_batch(env, policy, horizon, batchsize, 
#                               action_filter=action_filter, 
#                               seed=seed, 
#                               n_jobs=parallel)
#        perf = performance(batch, disc)
#        
#        #Estimate policy gradient
#        grad_samples = gpomdp_estimator(batch, disc, policy, 
#                                    baselinekind='peters', 
#                                    shallow=True,
#                                    result='samples')
#        grad = torch.mean(grad_samples, 0)
#        upsilon_grad = grad[1:]
#        upsilon_grad_norm = torch.norm(upsilon_grad)
#        omega_grad = grad[0]        
#        dfn = upsilon_grad.shape[0]
#        
#        ### Mean-update iteration
#        if it % 2 == 0:
#            #Compute gradient estimation error for mean parameters
#            if conf < 1 and grad_samples.size()[1] > 2:
#                centered = grad_samples[:, 1:] - upsilon_grad.unsqueeze(0)
#                grad_cov = batchsize/(batchsize - 1) * torch.mean(torch.bmm(centered.unsqueeze(2), centered.unsqueeze(1)),0)
#                upsilon_grad_var = torch.sum(torch.diag(grad_cov)).item()
#                max_eigv = eigsh(grad_cov.numpy(), 1)[0][0]
#                quant = sts.f.ppf(1 - conf, dfn, batchsize - dfn)
#                upsilon_eps = math.sqrt(max_eigv * dfn * quant)
#            elif conf < 1:
#                upsilon_grad_var = torch.var(grad_samples[:, 1]).item()
#                quant = sts.t.ppf(1 - conf/2, batchsize - 1)
#                upsilon_eps = quant * math.sqrt(upsilon_grad_var)
#            else:
#                upsilon_eps = 0.
#                upsilon_grad_var = 0.
#                    
#            #Compute safe step size for mean parameters
#            mask = torch.ones_like(grad)
#            mask[0] = 0.
#            F = power(policy, batch, grad, disc, 
#                  pow_alpha=pow_alpha, 
#                  err_tol=pow_err_tol, 
#                  max_it=max_pow_it, 
#                  max_attempts=max_pow_attempts,
#                  shallow=True, 
#                  clip=pow_clip,
#                  verbose=verbose,
#                  mask=mask)
#            req = safety_req.next(perf)
#            max_req = (upsilon_grad_norm - upsilon_eps / math.sqrt((batchsize - dfn)))**2 / \
#                        (2 * F)
#            req = min(req, max_req)
#            alpha = (upsilon_grad_norm - upsilon_eps / math.sqrt((batchsize - dfn)))  / \
#                        F * \
#                        (1 + math.sqrt(1 - req / max_req))
#            stepsize = (alpha / upsilon_grad_norm).item()
#            
#            #Ensure minimum safe batchsize
#            min_batchsize = math.ceil((upsilon_eps**2 / upsilon_grad_norm**2).item() + 1e-12) + dfn
#            if conf < 1 and adapt_batchsize:
#                batchsize = max(batchsize, min_batchsize)
#            
#            #Update mean parameters
#            upsilon = policy.get_loc_params()
#            new_upsilon = upsilon + alpha * sigma**2 * upsilon_grad / upsilon_grad_norm
#            policy.set_loc_params(new_upsilon)
#        ###  
#        ### Variance-update iteration
#        else:
#            #Estimate meta gradient (alpha from previous step)
#            omega_metagrad = metagrad(batch, disc, policy, alpha, 
#                                      grad_samples=grad_samples)
#            omega_metagrad_norm = torch.norm(omega_metagrad)
#            
#            #Compute gradient estimation error for variance parameter
#            if conf < 1:
#                omega_grad_var = torch.var(grad_samples[:, 0]).item()
#                quant = sts.t.ppf(1 - conf/2, batchsize - 1)
#                omega_eps = quant * math.sqrt(omega_grad_var / batchsize)
#            else:
#                omega_grad_var = 0.
#                omega_eps = 0.
#                
#            #Compute safe meta step size
#            mask = torch.zeros_like(grad)
#            mask[0] = 1.
#            G = power(policy, batch, grad, disc, 
#                      pow_alpha=pow_alpha, 
#                      err_tol=pow_err_tol, 
#                      max_it=max_pow_it, 
#                      max_attempts=max_pow_attempts,
#                      shallow=True, 
#                      clip=pow_clip,
#                      verbose=verbose,
#                      mask=mask)
#            req = safety_req.next(perf)
#            proj = omega_grad.view(-1).dot(omega_metagrad.view(-1)) / torch.norm(omega_metagrad)
#            max_req = (torch.abs(proj) - omega_eps / math.sqrt(batchsize))**2 / (2 * G)
#            req = min(req, max_req)
#            eta = (torch.abs(proj) - omega_eps / math.sqrt(batchsize)) / \
#                    G * \
#                    (torch.sign(proj) + torch.sqrt(1 - req / max_req))
#            metastepsize = (eta / omega_metagrad_norm).item()
#            
#            #Ensure minimum safe batchsize
#            min_batchsize = math.ceil((omega_eps**2 / proj**2).item() + 1e-12)
#            if conf < 1 and adapt_batchsize:
#                batchsize = max(batchsize, min_batchsize)
#        
#            #Update variance parameters
#            new_omega = omega + eta * omega_metagrad / omega_metagrad_norm
#            policy.set_scale_params(new_omega)
#
#        # Log
#        log_row['UpsilonLip'] = F
#        log_row['OmegaLip'] = G
#        log_row['Exploration'] = sigma.item()
#        log_row['StepSize'] = stepsize
#        log_row['MetaStepSize'] = metastepsize
#        log_row['OmegaGrad'] = omega_grad.item()
#        log_row['OmegaMetagrad'] = omega_metagrad.item()
#        log_row['UpsilonGradNorm'] = torch.norm(upsilon_grad).item()
#        log_row['BatchSize'] = batchsize
#        log_row['Perf'] = perf
#        log_row['UPerf'] = performance(batch, 1.)
#        log_row['AvgHorizon'] = avg_horizon(batch)
#        log_row['UpsilonGradVar'] = upsilon_grad_var
#        log_row['UpsilonEps'] = upsilon_eps
#        log_row['OmegaGradVar'] = upsilon_grad_var
#        log_row['OmegaEps'] = upsilon_eps
#        log_row['Req'] = req
#        print(min_batchsize)
#        log_row['MinBatchSize'] = min_batchsize
#        log_row['MaxReq'] = max_req.item()
#        params = policy.get_flat()
#        if log_params:
#            for i in range(policy.num_params()):
#                log_row['param%d' % i] = params[i].item()
#        logger.write_row(log_row, it)
#        if save_params and it % save_params == 0:
#            logger.save_params(params, it)
#        
#        # Next iteration
#        it += 1
#    
#    # Final policy
#    if save_params:
#        logger.save_params(params, it)
