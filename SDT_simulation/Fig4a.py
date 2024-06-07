#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 09:55:03 2023
This code generate Fig 4a using data generated from "SFT_simulation.py" stored
as SDT_simulation.pkl 

@author: wuy19
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DataDir = '/isilon/LFMI/VMdrive/YuanHao/AnalysisDirectory/HLTP_fMRI-Prestimulus-Activity/data'
FigDir = '/isilon/LFMI/VMdrive/YuanHao/AnalysisDirectory/HLTP_fMRI-Prestimulus-Activity/figures'
# %%  Conduct linear regression (OLS) to assess SDT behavioral metrics as a
# function of response variability adn plot the results
  
import statsmodels.api as sm
df_simulation = pd.read_pickle(DataDir + '/SDT_simulation.pkl')
xlabels = ['Hit Rate', 'False Alarm Rate', 'Criterion', 'Sensitiviy']
ylims = [[0.45,0.55],
         [.15,.3],
         [.30,.55],
         [.6,1.1]]
ylabels = [['0.46', '0.48', '0.50', '0.52', '0.54'],
           ['0.15', '0.20', '0.25', '0.30'],
           ['0.3', '0.35', '0.4', '0.45', '0.5'],
           ['0.6', '0.7', '0.8', '0.9', '1.0']]

for i, bhv in enumerate(['HR', 'FAR', 'c', 'dprime']):
    group_mean = df_simulation.groupby('sd')[bhv].mean().values
    std = df_simulation.groupby('sd')[bhv].std(ddof=1).values


    x= df_simulation['sd'].values
    y=df_simulation[bhv].values
    x= sm.add_constant(x)
    model = sm.OLS(y,x).fit()
 
    fig, ax = plt.subplots(1, 1, figsize=(1.5, 1.3))
    ax.plot(np.arange(1.8,3.4,.2), model.params[0] + model.params[1] * np.arange(1.8, 3.4, .2),
        color='gray', linewidth=1)
    ax.errorbar(np.arange(2,3.1, .2), group_mean, yerr = std, capsize = 2, ecolor='gray',
            linestyle='')
    ax.plot(np.arange(2,3.1,.2), group_mean, marker = 'o', linestyle='', color = 'black', markersize = 4,
             markeredgecolor = 'black', markerfacecolor = 'gray')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_position(('outward', 5))
    ax.yaxis.set_ticks_position('left')
    ax.spines['bottom'].set_position(('outward', 5))
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks([2.0, 2.2, 2.4, 2.6, 2.8, 3.0])
    
    yticks = np.array(ylabels[i])
    yticks = yticks.astype(float)
    ax.set_yticks(yticks)
    ax.set_xticklabels(['2','2.2', '2.4', '2.6', '2.8', '3'], fontsize=7)
                       
    ax.set_yticklabels(ylabels[i], fontsize=7)
    plt.xlim([1.9,3.1])
    plt.ylim(ylims[i])
    plt.xlabel('SD of the response distributions (sigma)', fontsize= 7)
    plt.ylabel(xlabels[i], fontsize=7)
    plt.show()
    #fig.savefig(FigDir + '/SDT_simulation_' + bhv + '.svg', 
    #                    bbox_inches = 'tight', dpi = 600, transparent = True)
    #fig.savefig(FigDir + '/SDT_simulation_' + bhv + '.png', 
    #                    bbox_inches = 'tight', dpi = 600, transparent = True)