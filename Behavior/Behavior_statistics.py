#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 11:07:49                                          
@author: podvae01, wuy19

Compute group statistical tests for behavioral metrics including hit rate (HR),
false alarm rate (FAR), sensitivity (d'), and criterion (c), and categorization accuracy.

Required input data:
    Mediansplit_df (Behavioral log files) 

Output:
    Source data and statistics for Fig 1D-F and Fig S1

"""

import pandas as pd
import numpy as np
import HLTP
from HLTP import get_bhv_vars
from scipy import stats
from statsmodels.stats.anova import AnovaRM
from scipy.stats import pearsonr

#--------- Load behavioral data in df ------------------------
DataDir = '/isilon/LFMI/VMdrive/YuanHao/AnalysisDirectory/HLTP_fMRI-Prestimulus-Activity/data'
bhv_df = pd.read_pickle(DataDir + "/Mediansplit_df")
n_sub = len(bhv_df.subject.unique())
# %% ##########################################################################
# Descriptive and inferential statistics for HR and FAR (Fig 1D)
###############################################################################

proportion_R = bhv_df.groupby(['real', 'subject'])['R'].mean()
percent_seen_real = 100. * proportion_R.xs(1, level='real').values
percent_seen_scra = 100. * proportion_R.xs(0, level='real').values

avg_seen_real = np.nanmean(percent_seen_real) # mean HR
avg_seen_scra = np.nanmean(percent_seen_scra) # mean FAR
sem_seen_real = (percent_seen_real.std(ddof=1)) / np.sqrt(percent_seen_real.shape[0]) # SEM HR
sem_seen_scra = (percent_seen_scra.std(ddof=1)) / np.sqrt(percent_seen_scra.shape[0]) # SEM FAR 

print("")
print("RECOGNITION RATE FOR REAL IMAGES (Hit Rate) ")
print(f"MEAN: {str(avg_seen_real) :<25} SEM: {str(sem_seen_real)}")
w, p = stats.wilcoxon(x=percent_seen_real-50, alternative="two-sided")
print("")
print("HR AGAINST THRESHOLD LEVEL OF 50% USING WOLCOXON SIGNED RANK TEST")
print(f"W-statistic: {str(w) :<25} pval: {str(p)}")
del w,p

print("")
print("RECOGNITION RATE FOR SCRAMBLED IMAGES (FAR)")
print(f"MEAN: {str(avg_seen_scra) :<25} SEM: {str(sem_seen_scra)}")

w, p = stats.wilcoxon(x=percent_seen_scra, alternative="two-sided")
print("")
print("FAR AGAINST THRESHOLD LEVEL OF 50% USING WOLCOXON SIGNED RANK TEST")
print(f"W-statistic: {str(w) :<25} pval: {str(p)}")
del w,p

w, p = stats.wilcoxon(x=percent_seen_real, y=percent_seen_scra,
                      alternative="greater")
print("")
print("REAL IMAGE VS SCRAMBLED IMAGE RECOGNITION RATE")
print(f"W-statistic: {str(w) :<25} pval: {str(p)}")
del w,p

# save categorization results for figure generation
#proportion_R.to_pickle(DataDir + '/proportion_R.pkl')
del avg_seen_real, avg_seen_scra, percent_seen_real, percent_seen_scra, \
    proportion_R, sem_seen_real, sem_seen_scra
# %% ##########################################################################
# Descriptive and inferential statistics for categorization accuracy (Fig 1E)
############################################################################### 
real_df = bhv_df.groupby(
    ['recognition', 'real', 'subject'])['correct'].mean()
real_df = real_df.reset_index()
real_df = real_df.loc[real_df['recognition'].isin([-1,1])]
real_df = real_df.loc[real_df['real']==1]
real_df['correct'] = real_df['correct']*100

cat_real_R = real_df[real_df['recognition']==1]['correct'].values
cat_real_U = real_df[real_df['recognition']==-1]['correct'].values

# mean and sem: categorization accuracy for recognized real images
avg_cat_real_R = np.nanmean(cat_real_R)
sem_cat_real_R = (cat_real_R.std(ddof=1)) / np.sqrt(cat_real_R.shape[0])
# mean and sem: categorization accuracy for unrecognized real images
avg_cat_real_U = np.nanmean(cat_real_U) 
sem_cat_real_U = (cat_real_U.std(ddof=1)) / np.sqrt(cat_real_U.shape[0])  

print("")
print("CATEGORIZATION ACCURACY FOR RECOGNIZED REAL IMAGES")
print(f"MEAN: {str(avg_cat_real_R) :<25} SEM: {str(sem_cat_real_R)}")
w, p = stats.wilcoxon(x=cat_real_R-25, alternative="greater")
print("")
print("AGAINST 25% USING WOLCOXON SIGNED RANK TEST")
print(f"W-statistic: {str(w) :<25} pval: {str(p)}")
del w,p

print("")
print("CATEGORIZATION ACCURACY FOR UNRECOGNIZED REAL IMAGES")
print(f"MEAN: {str(avg_cat_real_U) :<25} SEM: {str(sem_cat_real_U)}")
w, p = stats.wilcoxon(x=cat_real_U-25, method='approx', alternative="greater")
print("")
print("AGAINST 25% USING WOLCOXON SIGNED RANK TEST")
print(f"W-statistic: {str(w) :<25} pval: {str(p)}")
del w,p
del avg_cat_real_R, avg_cat_real_U, cat_real_R, cat_real_U, sem_cat_real_R, sem_cat_real_U

# For scrambled images, only subjects with <= 5 recognition report were included 
# Find subjects that report scrambled images <= 5 times
n_seen = bhv_df.groupby(['real', 'subject'])['R'].sum()
n_seen_scra = n_seen.xs(0, level='real').values
n_seen_scra_subj = n_seen.xs(0, level='real').index[n_seen_scra <= 5]

# exlude these subjects from scr image analysis
scr_bhv_df = bhv_df.copy()
for s in n_seen_scra_subj:
    scr_bhv_df = scr_bhv_df[scr_bhv_df.subject != s]
scr_df = scr_bhv_df.groupby(
        ['recognition', 'real', 'subject'])['correct'].mean()
scr_df = scr_df.reset_index()
scr_df = scr_df.loc[scr_df['recognition'].isin([-1,1])]
scr_df = scr_df.loc[scr_df['real']==0]
scr_df['correct'] = scr_df['correct']*100
del n_seen, n_seen_scra, n_seen_scra_subj, s, scr_bhv_df

cat_scr_R = scr_df[scr_df['recognition']==1]['correct'].values
cat_scr_U = scr_df[scr_df['recognition']==-1]['correct'].values

# mean and sem: categorization accuracy for recognized scrambled images
avg_cat_scr_R = np.nanmean(cat_scr_R)
sem_cat_scr_R = (cat_scr_R.std(ddof=1)) / np.sqrt(cat_scr_R.shape[0])
# mean and sem: categorization accuracy for unrecognized scrambled images
avg_cat_scr_U = np.nanmean(cat_scr_U)
sem_cat_scr_U = (cat_scr_U.std(ddof=1)) / np.sqrt(cat_scr_U.shape[0])  

print("")
print("CATEGORIZATION ACCURACY FOR RECOGNIZED SCR IMAGES")
print(f"MEAN: {str(avg_cat_scr_R) :<25} SEM: {str(sem_cat_scr_R)}")
w, p = stats.wilcoxon(x=cat_scr_R-25, alternative="greater", method='auto')
print("")
print("TEST AGAINST CHANCE LEVEL OF 25% USING WOLCOXON SIGNED RANK TEST")
print(f"W-statistic: {str(w) :<25} pval: {str(p)}")
del w,p
w, p = stats.wilcoxon(x=cat_scr_R-cat_scr_U, alternative="greater", method='approx')
print("")
print("RECOGNIZED SCR AGAINST UNRECOGNIZED SCR USING WOLCOXON SIGNED RANK TEST")
print(f"W-statistic: {str(w) :<25} pval: {str(p)}")
del w,p


print("")
print("CATEGORIZATION ACCURACY FOR UNRECOGNIZED SCR IMAGES")
print(f"MEAN: {str(avg_cat_scr_U) :<25} SEM: {str(sem_cat_scr_U)}")
#w, p = stats.wilcoxon(x=cat_scr_U-25, method='approx', alternative="greater")
#print("")
#print("TEST AGAINST CHANCE LEVEL OF 25% USING WOLCOXON SIGNED RANK TEST")
#print(f"W-statistic: {str(w) :<25} pval: {str(p)}")
#del w,p

# save categorization results for figure generation
categorization_df = pd.concat([real_df, scr_df], axis=0) 
#categorization_df.to_pickle(DataDir + '/CategorizationReport.pkl')
del avg_cat_scr_R, avg_cat_scr_U, cat_scr_R, cat_scr_U, sem_cat_scr_R, \
    sem_cat_scr_U, real_df, scr_df, categorization_df

# %% ########################################################################## 
# 1x5 ANOVA on HR, FAR, c, and d' Fig S1
###############################################################################

n_subj = 25; n_groups = 5
subject = np.tile(np.arange(n_subj), (n_groups, 1)).T
data_group = np.tile(np.arange(1, 1 + n_groups), (n_subj, 1))

n_block_trials = 72

# matrix with a size of 25 subjects x 6 trial groups (one subject completed
# 18 runs or 432 trials x 8 behavioral variables)
# behavioral variables: rec_rate, HR, FAR, d, c, p_correct, catRT, recRT
all_block_vars = np.zeros( (25, 6, 8) ) + np.nan 
for s_idx, s in enumerate(HLTP.subjects):
    subj_df = bhv_df[bhv_df.subject == s]
    n_trials = len(subj_df)
    n_blocks = n_trials / n_block_trials
    for n in range(0, int(n_blocks)):
        all_block_vars[s_idx, n, :] = get_bhv_vars(
            subj_df[(n * n_block_trials):(n_block_trials * ( n + 1 ))])

# get HR, FAR, d' and c                 
HR = all_block_vars[:, :5, 1]
raw_mean = np.nanmean(HR, axis=1)
inds = np.where(np.isnan(HR))
HR[inds] = np.take(raw_mean, inds[0])

FAR = all_block_vars[:, :5, 2]
raw_mean = np.nanmean(FAR, axis=1)
inds = np.where(np.isnan(FAR))
FAR[inds] = np.take(raw_mean, inds[0])

sensitivity =  all_block_vars[:, :5, 3]
raw_mean = np.nanmean(sensitivity, axis=1)
inds = np.where(np.isnan(sensitivity))
sensitivity[inds] = np.take(raw_mean, inds[0])

criterion = all_block_vars[:, :5, 4]
raw_mean = np.nanmean(criterion, axis=1)
inds = np.where(np.isnan(criterion))
criterion[inds] = np.take(raw_mean, inds[0])

df_HR = pd.DataFrame({'bhv': 'HR',
                   'value': HR.flatten(),
                   'group' : data_group.flatten(),
                   'subject' : subject.flatten()}) 
df_FAR = pd.DataFrame({'bhv': 'FAR',
                   'value': FAR.flatten(),
                   'group' : data_group.flatten(),
                   'subject' : subject.flatten()}) 
df_dprime = pd.DataFrame({'bhv': 'd',
                   'value': sensitivity.flatten(),
                   'group' : data_group.flatten(),
                   'subject' : subject.flatten()}) 
df_c = pd.DataFrame({'bhv': 'c',
                   'value': criterion.flatten(),
                   'group' : data_group.flatten(),
                   'subject' : subject.flatten()}) 
df_sdt = pd.concat([df_HR, df_FAR, df_dprime, df_c], ignore_index=True)
del df_c, df_dprime, df_HR, df_FAR

bhv = ['HR', 'FAR', 'd', 'c']
for i in bhv:
    result = AnovaRM(df_sdt[df_sdt['bhv']==i], 'value', 'subject', within = ['group']).fit()
    print('#############################')
    print('ANOVA RESULTS for ' + i)
    print('#############################') 
    print(result.anova_table)
    print('')
    print('') 
#df_sdt.to_pickle(DataDir + '/bhv_across_times')
del all_block_vars, bhv, criterion, data_group,i, inds, n, n_block_trials, \
    n_blocks, n_groups, n_subj, n_trials, raw_mean, s, s_idx,\
        sensitivity, subj_df, subject, HR, FAR 
# %% #############################################
# COMPUTE c and d' (Fig 1F)
##################################################

df_bhv_vars = []
for s in HLTP.subjects:
    subj_df = bhv_df[bhv_df.subject == s]
    df_bhv_vars.append(get_bhv_vars(subj_df))
df_bhv_vars = np.array(df_bhv_vars)
df_bhv_vars = df_bhv_vars[:, [3,4,5]]
df_bhv_vars = pd.DataFrame(data=df_bhv_vars, columns = ['d', 'c', 'p_correct'])
del subj_df, s

dprime =df_bhv_vars['d'].to_numpy()
criterion = df_bhv_vars['c'].to_numpy()
p_correct = df_bhv_vars['p_correct'].to_numpy()

# mean and sem for each variable
avg_dprime = np.nanmean(dprime)
sem_dprime = (np.nanstd(dprime, ddof=1)) / np.sqrt(dprime.shape[0])
print(stats.wilcoxon(x=dprime, alternative="two-sided", method= 'auto'))

avg_c = np.nanmean(criterion)
sem_c = (np.nanstd(criterion, ddof=1)) / np.sqrt(criterion.shape[0])
print(stats.wilcoxon(x=criterion, alternative="two-sided", method= 'auto'))

##################################################
# Relation between c and p_correct
################################################## 
corr, p = pearsonr(criterion, p_correct)
print(f"Pearson's Correlation': {str(corr) :<25} P-value: {str(p)}")
