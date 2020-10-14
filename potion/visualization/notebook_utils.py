#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 17:26:37 2019

@author: matteo
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as sts
import math
import os
import glob
import warnings

small_eps = 1e-6

def bootstrap_ci(x, conf=0.95, resamples=10000):
    means = [np.mean(x[np.random.choice(x.shape[0], size=x.shape[0], replace=True), :], axis=0) for _ in range(resamples)]
    low = np.percentile(means, (1-conf)/2 * 100, axis=0)
    high = np.percentile(means, (1 - (1-conf)/2) * 100, axis=0)
    low = np.nan_to_num(low)
    high = np.nan_to_num(high)
    return low, high

def plot_all(dfs, key='Perf', name='', xkey=None):
    lines = []
    for df in dfs:
        value = df[key]
        xx = range(len(value)) if xkey is None else df[xkey]
        line, = plt.plot(xx, value, label=name)
        lines.append(line)
    plt.xlabel('Iterations')
    plt.ylabel(key)
    return lines

def moments(dfs):
    cdf = pd.concat(dfs, sort=True).groupby(level=0)
    return cdf.mean(), cdf.std().fillna(0)
    
def plot_ci(dfs, key='Perf', conf=0.95, name='', xkey=None, bootstrap=False, resamples=10000, mult=1.):
    n_runs = len(dfs)
    mean_df, std_df = moments(dfs)
    mean = mean_df[key] * mult
    std = std_df[key] * mult
    if xkey is None:
        xx = range(len(mean))
    elif xkey in mean_df:
        xx = mean_df[xkey]
    else:
        xx = np.array(range(len(mean))) * 100
    line, = plt.plot(xx, mean, label=name)
    if bootstrap:
        data = np.array([df[key] * mult for df in dfs])
        interval = bootstrap_ci(data, conf, resamples)
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered in multiply")
            interval = sts.t.interval(conf, n_runs-1,loc=mean,scale=std/math.sqrt(n_runs))
        
    plt.fill_between(xx, interval[0], interval[1], alpha=0.3)
    print('%s: %f +- %f' % (name, np.mean(mean), np.mean(std)/n_runs))
    return line

def save_csv(env, name, key, conf=0.95, path='.', rows=200, batchsize=500, xkey=None, bootstrap=False, resamples=10000, mult=1., step=1):
    dfs = load_all(env + '_' + name, rows)
    n_runs = len(dfs)
    mean_df, std_df = moments(dfs)
    mean = mean_df[key].values * mult
    std = std_df[key].values * mult + 1e-24
    if bootstrap:
        data = np.array([df[key] * mult for df in dfs])
        interval = bootstrap_ci(data, conf, resamples)     
    else:
        interval = sts.t.interval(conf, n_runs-1,loc=mean,scale=std/math.sqrt(n_runs))
    low, high = interval
    if rows is not None:
        mean = mean[:rows]
        low = low[:rows]
        high = high[:rows]
    if xkey is None:
        xx = range(len(mean))
    elif xkey in mean_df:
        xx = mean_df[xkey]
    else:
        xx = np.array(range(len(mean))) * 100
    
    for i in range(len(mean)):
        if not np.isfinite(low[i]):
            low[i] = mean[i]
        if not np.isfinite(high[i]):
            high[i] = mean[i]
    
    plotdf = pd.DataFrame({("it" if xkey is None else xkey): xx, "mean" : mean, "low" : low, "high": high})
    plotdf = plotdf.iloc[0:-1:step]
    print(len(plotdf))
    plotdf.to_csv(path + '/' + env.lower() + '_' + name.lower() + '_' + key.lower() + '.csv', index=False, header=False)



def load_all(name, rows=200):
    dfs = [pd.read_csv(file, index_col=False, nrows=rows) for file in glob.glob("*.csv") if file.startswith(name + '_')]
    #for df in dfs:
    #    df['CumInfo'] = np.cumsum(df['Info'])
    return dfs

def compare(env, names, keys=['Perf'], conf=0.95, logdir=None, separate=False, ymin=None, ymax=None, rows=200, xkey=None, xmax=None, bootstrap=False, resamples=10000, mult=None, roll=1.):
    figures = []
    for key in keys:
        figures.append(plt.figure())
        if ymin is not None and ymax is not None:
            plt.ylim(ymin, ymax)
        if xmax is not None:
            plt.xlim(0, xmax)
        if logdir is not None:
            os.chdir(logdir)
        handles = []
        if type(roll) is int or type(roll) is float:
            roll = [int(roll)]*len(names)
        if mult is None:
            mult = [1.] * len(names)
        for i, name in enumerate(names):
            dfs = load_all(env + '_' + name, rows=rows)
            dfs = [dfs[j].rolling(roll[i]).mean() for j in range(len(dfs))]
            if separate:
                handles+=(plot_all(dfs, key, name, xkey=xkey))
            else:
                handles.append(plot_ci(dfs, key, conf, name, xkey=xkey, bootstrap=bootstrap, resamples=resamples, mult=mult[i]))
        plt.legend(handles=handles)
        plt.show()
    return figures