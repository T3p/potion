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
    
def plot_ci(dfs, key='Perf', conf=0.95, name='', xkey=None):
    n_runs = len(dfs)
    mean_df, std_df = moments(dfs)
    mean = mean_df[key]
    std = std_df[key]
    xx = range(len(mean)) if xkey is None else mean_df[xkey]
    line, = plt.plot(xx, mean, label=name)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in multiply")
        interval = sts.t.interval(conf, n_runs-1,loc=mean,scale=std/math.sqrt(n_runs))
    plt.fill_between(xx, interval[0], interval[1], alpha=0.3)
    print('%s: %f +- %f' % (name, np.mean(mean), np.mean(std)/n_runs))
    return line

def save_csv(env, name, key, conf=0.95, path='.', rows=200, batchsize=500):
    dfs = load_all(env + '_' + name, rows)
    n_runs = len(dfs)
    mean_df, std_df = moments(dfs)
    mean = mean_df[key].values
    std = std_df[key].values + 1e-24
    interval = sts.t.interval(conf, n_runs-1,loc=mean,scale=std/math.sqrt(n_runs))
    low, high = interval
    if rows is not None:
        mean = mean[:rows]
        low = low[:rows]
        high = high[:rows]
    if 'BatchSize' in mean_df:
        x = mean_df['BatchSize'].values
    else:
        x = [batchsize] * len(mean)
    x = np.cumsum(x) - x[0]
    plotdf = pd.DataFrame({"Episodes": x, "Mean" : mean, "Low" : low, "High": high})
    plotdf.to_csv(path + '/' + env.lower() + '_' + name.lower() + '_' + key.lower() + '.csv', index=False, header=False)



def load_all(name, nrows=200):
    return [pd.read_csv(file, index_col=False, nrows=nrows) for file in glob.glob("*.csv") if file.startswith(name + '_')]

def compare(env, names, keys=['Perf'], conf=0.95, logdir=None, separate=False, ymin=None, ymax=None, nrows=200, xkey=None, xmax=None):
    for key in keys:
        plt.figure()
        if ymin is not None and ymax is not None:
            plt.ylim(ymin, ymax)
        if xmax is not None:
            plt.xlim(0, xmax)
        if logdir is not None:
            os.chdir(logdir)
        handles = []
        for name in names:
            dfs = load_all(env + '_' + name, nrows=nrows)
            if separate:
                handles+=(plot_all(dfs, key, name, xkey=xkey))
            else:
                handles.append(plot_ci(dfs, key, conf, name, xkey=xkey))
        plt.legend(handles=handles)
        plt.show()