#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 12:29:42 2023

@author: wuy19
"""

import numpy as np
import pandas as pd
from scipy import stats

DataDir = '/isilon/LFMI/VMdrive/YuanHao/AnalysisDirectory/HLTP_fMRI-Prestimulus-Activity/data'
signal_mean = 1
noise_mean = -1
threshold = 1
n_trials = 5000
n_scr = 2500
n_samples = 10000
tmp = np.zeros(n_samples)
tmp[:] = np.nan

sd_range = [3, 2.8, 2.6, 2.4, 2.2, 2]
df_simulation = pd.DataFrame()
# %% Simulate sample obvservers' behavior with varying standard deviations of
# target and non-target distributions

for sd in sd_range:
    data = {'sd': np.tile(sd,len(tmp)), 'HR': tmp, 'MISS': tmp, 'FAR': tmp, 'CR': tmp, 'dprime': tmp, 'c':tmp}
    df_tmp = pd.DataFrame(data)
    for n in range(0,n_samples):
        print(n)
        signal_present = np.ones(n_trials)
        scr = np.random.choice(np.arange(0, n_trials), size=n_scr, replace=False)
        signal_present[scr] = 0
        signal_present = signal_present.astype(dtype=bool)
        signal = np.random.normal(noise_mean, sd, size=signal_present.size)
        signal[signal_present] = np.random.normal(signal_mean, sd, size=signal_present.sum())
        df=pd.DataFrame({"trial": range(len(signal)), "signal_present": signal_present, "signal": signal})
        df['response'] = df.signal > threshold

        # calculate hits, misses, cr, and fa
        hit = df.response[df.signal_present]
        miss = ~df.response[df.signal_present]
        fa = df.response[~df.signal_present]
        cr = ~df.response[~df.signal_present]
        dprime = stats.norm.ppf(hit.mean()) - stats.norm.ppf(fa.mean())
        c = -(stats.norm.ppf(hit.mean()) + stats.norm.ppf(fa.mean()))/2.0

        df_tmp.HR[n] = hit.mean()
        df_tmp.MISS[n] = miss.mean()
        df_tmp.FAR[n] = fa.mean()
        df_tmp.CR[n] = cr.mean()
        df_tmp.dprime[n] = dprime
        df_tmp.c[n] = c
        del hit, miss, fa, cr, dprime, c, df, signal, signal_present, scr
    df_simulation = pd.concat([df_simulation, df_tmp], axis = 0,
                                  ignore_index= True)
    del data, df_tmp
    df_simulation.to_pickle(DataDir + '/SDT_simulation.pkl')