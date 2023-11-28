#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 11:18:50 2023

@author: wuy19
"""

import sys
import os 
ProjDir = ProjDir = '/isilon/LFMI/VMdrive/YuanHao/AnalysisDirectory/HLTP_fMRI-Prestimulus-Activity'
sys.path.insert(0, ProjDir)
os.chdir(ProjDir)

import HLTP
from HLTP import get_bhv_vars
import pandas as pd
import numpy as np
import nibabel
from nilearn.input_data import NiftiMasker
import pickle 
import statsmodels.formula.api as smf

FigDir = ProjDir + '/figures'
DataDir = ProjDir + '/data'
MaskDir =  '/isilon/LFMI/VMdrive/YuanHao/HLTP_fMRI_Prestimulus/masks'
mask_names  = ['Visual', 'vmPFC', 'CO', 'RSC']
bhv_df = pd.read_pickle(DataDir + "/Mediansplit_df")

# %%
def extract_ROI_prestimulus_activity(subj, roi, key='resid'):
    """ Extract residuals correspondong to the prestimulus period （-2 to 0 sec）
    from each ROI and save it as python dict"""
    
    MaskDir =  '/isilon/LFMI/VMdrive/YuanHao/HLTP_fMRI_Prestimulus/masks'
    ROI_mask = MaskDir +  '/' + roi + '_mask.nii.gz'
    nifti_masker = NiftiMasker(mask_img = ROI_mask)
          
    funcdir = HLTP.data_dir +'/sub' + str(subj).zfill(2)  + '/proc_data/func'
    fmri_img_file = funcdir + '/ER_' + key + '_Baseline12highres2standadrd.nii.gz'  
    img = nibabel.load(fmri_img_file)        
    img_masked = nifti_masker.fit_transform(img)
    return img_masked

def sort_trials_based_on_activity(data_dict, bhv_df, roi, n_subj, n_groups):
    """for each voxel, divide trials based on the magnitude of prestimulus 
    activity into 5 groups and compute the categorization accuracy for each 
    traiil group"""   
    roi_data = data_dict[roi]
    n_voxels = np.size(roi_data[0], 1)  
    group_percentile = np.arange(0., 100., 20)
    bhv_var = np.zeros([len(roi_data), n_voxels, n_groups])
         
    for ind, subj in enumerate(HLTP.subjects):
        bhv_subj = bhv_df[(bhv_df.subject == subj) & (bhv_df.fMRI == True)]     
        subj_data = roi_data[ind]
        for v in range(n_voxels): 
            p_group = np.digitize(subj_data[:,v], np.percentile(subj_data[:,v], group_percentile))
            # for each group, calculate behavioral variables
            for k, group in enumerate(np.unique(p_group)):
                group_df = bhv_subj.loc[p_group == group]
                _, _, _, _, _, bhv_var[ind, v, k], _, _ = get_bhv_vars(group_df)        
    return bhv_var


def bhv_against_ROIActivity(bhv_dict, roi, n_subj, n_groups):
    """Perform LMM to assess ROI prestimlus acitivity's influence on
    categorization accuracy"""
    subject = np.tile(np.arange(n_subj), (n_groups, 1)).T
    data_group = np.tile(np.arange(1, 1 + n_groups), (n_subj, 1))
    
    roi_bhv = bhv_dict[roi]    
    mean_bhv = np.mean(roi_bhv, axis=1) 
    df = pd.DataFrame({'subject':subject.flatten(),
                                "group":data_group.flatten(),
                                'bhv_var': mean_bhv.flatten()})

    L = smf.mixedlm('bhv_var' + " ~ group", df,
                  groups = df["subject"],
                  re_formula = " ~ group").fit()
          
    print('**************************************')
    print(roi)
    print('**************************************')
    print(L.summary())
# %%
data_dict = {}
bhv_dict = {}
n_subj=25; n_groups= 5


for roi in mask_names:
    roi_data = []
    for subj in HLTP.subjects:
        img_masked = extract_ROI_prestimulus_activity(subj, roi)        
        roi_data.append(img_masked)
    data_dict[roi] = roi_data    
with open(DataDir + '/SeedROI_PrestimActivity.pkl', 'wb') as f:
    pickle.dump(data_dict, f)
       
for roi in mask_names:
    bhv_var =  sort_trials_based_on_activity(data_dict, bhv_df, roi, n_subj, n_groups)
    bhv_dict[roi] = bhv_var
with open(DataDir + '/SeedROI_CategorizationAccuracy.pkl', 'wb') as f:
    pickle.dump(bhv_dict, f)   
    
for roi in mask_names:
     _,_, =  bhv_against_ROIActivity(bhv_dict,roi, n_subj, n_groups)               