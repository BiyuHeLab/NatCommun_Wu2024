#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 10:51:42 2020

Prepare data for the main analysis on prestimulus activity's influence on
perceptual behavior. 
Extract data from a specific TR relative to the stimulus onset for each trial
and concatenate them to one signle nifti file.  

@author: podvae01, wuy19
"""
import sys
import os
ProjDir='/isilon/LFMI/VMdrive/YuanHao/HLTP_fMRI_Prestimulus'
sys.path.insert(1, ProjDir)
os.chdir(ProjDir)

import warnings
warnings.filterwarnings("ignore")




import HLTP
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
 
data_dir = '/isilon/LFMI/VMdrive/data/HLTP_fMRI/'
EVs = '/evs/visual_ev.txt'
bhv_df = pd.read_pickle(HLTP.group_result + '/behavior/corrected_bhv_df.pkl')
n_groups = 5
group_percentile = np.arange(0., 100., 100/n_groups) 
n_subj = len(HLTP.subjects)
bhv_vars = ['HR', 'FAR', 'd', 'c']
# %%
grouped_bhv_data = pd.DataFrame(columns = ['subject', 'group', 'HR', 'FAR', 'd', 'c'])

for sub in HLTP.subjects:
    block_N = HLTP.get_block_numbers(sub)
    subj_dir = f"{data_dir}sub{str(sub).zfill(2)}/proc_data/func/"
    bhv_subj = bhv_df[(bhv_df.subject == sub) & (bhv_df.fMRI == True)]

    # get the relative motion from time point n to n+1
    prestim_motion = []
    for block in block_N:
        EV_file = subj_dir + '/block' + str(block) + EVs
        
        # now extract the relative motion parameters derived from 1 TR before the onsets 
        mcf_file = f"{subj_dir}block{block}/block{block}_preproc_unfiltered.feat/mc/prefiltered_func_data_mcf_rel.rms"
        rel_mcf = np.loadtxt(mcf_file)
        
        # Get the stimulus onsets TR within all collected data
        onsets = np.where(np.loadtxt(EV_file)[:len(rel_mcf)])[0]
        prestim_motion.append(rel_mcf[onsets - 1])
        del EV_file, mcf_file, rel_mcf, onsets, block
    prestim_motion = np.concatenate(prestim_motion)

    # Splt the trials in groups based on the magnitude of head motion  
    p_group = np.digitize(prestim_motion, np.percentile(prestim_motion, group_percentile))
      
    # for each trial group, calculate behavioral variables
    for group in np.unique(p_group):
        group_df = bhv_subj.loc[p_group == group]
        bhv_data = HLTP.get_bhv_vars(group_df)
        
        new_row_data = {'subject': sub, 'group': group, 'HR': bhv_data[0],
                       'FAR': bhv_data[1], 'd': bhv_data[2], 'c': bhv_data[3]}
        grouped_bhv_data.loc[len(grouped_bhv_data)] = new_row_data
        del group_df, bhv_data, new_row_data
    del block_N, subj_dir, bhv_subj, prestim_motion, sub   
# %%
FigDir = '/isilon/LFMI/VMdrive/YuanHao/AnalysisDirectory/HLTP_fMRI-Prestimulus-Activity/figures'

for bhv in bhv_vars:
    L = smf.mixedlm(bhv + " ~ group", grouped_bhv_data,
                  groups = grouped_bhv_data["subject"],
                  re_formula = " ~ group").fit()
    print('==================================================================')
    print(bhv)   
    print('==================================================================')
    print('') 
    print(L.summary())

    group_mean = grouped_bhv_data.groupby('group')[bhv].mean().values
    std = grouped_bhv_data.groupby('group')[bhv].std(ddof=1).values
    sem = std / np.sqrt(n_subj)


    fig, ax = plt.subplots(1, 1, figsize=(1.5, 1.3))
    ax.plot(np.arange(0,6,.1), L.params[0] + L.params[1] * np.arange(0, 6, .1),
            color='gray', linewidth=1)
    ax.errorbar(np.arange(1,6), group_mean, yerr = sem, capsize = 2, ecolor='gray',
                linestyle='')
    ax.plot(np.arange(1,6), group_mean, marker = 'o', linestyle='', color = 'black', markersize = 4,
                 markeredgecolor = 'black', markerfacecolor = 'gray')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_position(('outward', 5))
    ax.yaxis.set_ticks_position('left')
    ax.spines['bottom'].set_position(('outward', 5))
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(0,7))
    ax.set_yticks(np.arange(0.4,0.9, .1))
    ax.set_xticklabels(['low', '','', '', '', '', 'high'], fontsize=7)
    #ax.set_yticklabels(['0.48', '0.50', '0.52', '0.54', '0.56'], fontsize=7)
    plt.xlim([0,6])
    plt.ylim([0.38, 0.82])
    plt.xlabel('Relative displacement', fontsize= 7)
    plt.ylabel(bhv, fontsize=7)
    plt.show()
    fig.savefig(f"{FigDir}/{bhv}_by_prestimulus_head_motion.svg", 
                        bbox_inches = 'tight', dpi = 600, transparent = True)
    fig.savefig(f"{FigDir}/{bhv}_by_prestimulus_head_motion.png", 
                        bbox_inches = 'tight', dpi = 600, transparent = True)
    del bhv, L, group_mean, fig, ax, sem, std
