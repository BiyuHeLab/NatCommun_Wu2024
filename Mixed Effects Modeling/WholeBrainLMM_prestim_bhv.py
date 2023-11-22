#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 16:02:43 2020

Split trials into 5 groups based on the fMRI activity level before
the stimulus onset (2 - 0 sec relative to the stimulus onsets). 

Fit linear model to predict perceptual behavior based on the trail group.

@author: podvae01, wuy19
"""

import sys
import os 
proj_dir = '/isilon/LFMI/VMdrive/YuanHao/HLTP_fMRI_Prestimulus'
sys.path.insert(1, proj_dir)
os.chdir(proj_dir)

import HLTP
from HLTP import get_bhv_vars
from joblib import Parallel, delayed
import multiprocessing
import pandas as pd
import numpy as np
import nibabel
import nilearn
from statsmodels.stats.anova import AnovaRM    
import statsmodels.formula.api as smf

key = 'resid_Baseline12highres2standadrd'
bhv_df = pd.read_pickle(HLTP.group_result + '/behavior/corrected_bhv_df.pkl')

mask_fname ='/usr/local/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz'
mask_img = nibabel.load(mask_fname)

bhv_vars = ['HR', 'FAR', 'd', 'c']
num_cores = multiprocessing.cpu_count()
# %%
# this function calculate subject's behavioral variable within each group of 
# trials according to prestimulus voxel amplitude

def make_map_of_bhv_by_baseline(sub, key, bhv_df):

    group_percentile = np.arange(0., 100., 20)
    bhv_subj = bhv_df[(bhv_df.subject == sub) & (bhv_df.fMRI == True)]

    # load fmri data
    funcdir = HLTP.data_dir +'/sub' + "%02d" % sub + '/proc_data/func'
    fmri_img_file = funcdir + '/ER_' + key + '.nii.gz'  
    img = nibabel.load(fmri_img_file)
    fmri_data = img.get_fdata()
    brain_mask = fmri_data.sum(axis = -1) != 0
    voxels = np.array(np.where(brain_mask))
    
    #initialize the array of behavioral variables to report
    nan_arr = np.zeros((brain_mask.shape + (5,))) + np.nan
    bhv_data = {}    
    for bhv_var in bhv_vars:
        bhv_data[bhv_var] = nan_arr.copy()

    for v in range(len(voxels[1])):
        # for each voxel, split the trials in groups  
        vdata = fmri_data[voxels[0, v], voxels[1, v],voxels[2, v],:]
        p_group = np.digitize(vdata, np.percentile(vdata, group_percentile))
        
        # for each group, calculate behavioral variables
        for group in np.unique(p_group):
            group_df = bhv_subj.loc[p_group == group]
            bhv_voxel = get_bhv_vars(group_df)
            for ii, vv in enumerate(bhv_vars):
                bhv_data[vv][voxels[0, v], voxels[1, v],voxels[2, v], 
                   group - 1] = bhv_voxel[ii]
    #save the maps
    for vv in bhv_vars:
        bhv_img = nibabel.Nifti1Image(bhv_data[vv], img.affine, img.header)
        nibabel.save(bhv_img, funcdir + '/' + vv + '_by_resid_Baseline.nii.gz')

def missing_values(value_array):
   '''make sure the behavioral values are present in at least two groups 
   for at least 10 subjetcs. The reason for missing values is  
   either missing behavioral response, 
   or imperfect transform to standard brain'''
   present = ~np.isnan(value_array)
   n_subj = (present.sum(axis = 1) > 2).sum()
   if n_subj >= 15:
       return False
   return True

def fit_model_to_bhv_in_voxel(bhv_var, model_type):
    '''fit linear model to predict behavior from prestim in each voxel, '''
    # load all behavior by group data 
    n_subj = 25; n_groups = 5
    subject = np.tile(np.arange(n_subj), (n_groups, 1)).T
    # the group number is defined by voxel magnitude percentile
    data_group = np.tile(np.arange(1, 1 + n_groups), (n_subj, 1)) 
    #hold here the behavioral variables for each voxel and each subject
    group_data = []
    for sub in HLTP.subjects:
        funcdir = HLTP.data_dir +'/sub' + "%02d" % sub + '/proc_data/func'
        img = nibabel.load(funcdir + '/' + bhv_var + 
                           '_by_resid_Baseline.nii.gz')
        img = nilearn.image.smooth_img(img, 3)
        group_data.append(img.get_fdata())
    
    # for each voxel test with mixed linear model
    group_data = np.array(group_data)
    mask_fname ='/usr/local/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz'
    mask_img = nibabel.load(mask_fname)   
    mask = mask_img.get_fdata()

    voxels = np.array(np.where(mask))
    p_vals = np.ones((mask.shape))
    betas = np.zeros((mask.shape))
    for v in range(len(voxels[1])):
        voxel_bhv_var = group_data[:, voxels[0, v], voxels[1, v],voxels[2, v],:] # bhv_var in a given voxel
        n = ~np.isnan(voxel_bhv_var.flatten()) #sort nans
        if missing_values(voxel_bhv_var):
            continue
        df = pd.DataFrame.from_dict({bhv_var:
              voxel_bhv_var.flatten()[n], 
              "subject":subject.flatten()[n],
              "group":data_group.flatten()[n]})
        if model_type == "LMM":    
            try:
                model_Q = smf.mixedlm(bhv_var + "group", data = df,
                        groups = df["subject"], re_formula = "group").fit()        
                p_vals[voxels[0, v], voxels[1, v],voxels[2, v]
                       ] = model_Q.pvalues[1]
                betas[voxels[0, v], voxels[1, v],voxels[2, v]
                      ] = model_Q.params[1]
            except:
                pass
        elif model_type == "ANOVA":
            try:
                model = AnovaRM(df, bhv_var, 'subject', 
                                within = ['group']).fit()
                p_vals[voxels[0, v], voxels[1, v],voxels[2, v]
                       ] = model.anova_table.values[0,-1]
                betas[voxels[0, v], voxels[1, v],voxels[2, v]
                      ] = model.anova_table.values[0,0]
            except:
                pass
        else: 
            print("unknown model type")
            return
        
    HLTP.save([betas, p_vals], HLTP.data_dir + '/group_results/prestim_to_'
              + bhv_var + model_type + '_fwhm5')

Parallel(n_jobs = num_cores)(delayed(fit_model_to_bhv_in_voxel
                                     )(bhv_var, "LMM") 
                             for bhv_var in bhv_vars)            