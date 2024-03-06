#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 17:49:25 2023

-   Compare Category decoding accuracy derived high and low presitmulus activity trials
    against chance level across the entire brain, respectively
-   Create thresholding mask including voxels showing above-decoding accuracy.
-   required inputs:
    Whole-brain decoding accuracy maps from following conditions:
    High visual network prestimulus activity trials
    High CO network prestimulus activity trials
    High vmPFC prestimulus activity trials
    High RSC prestimulus activity trials
    Low visual network prestimulus activity trials
    Low CO network prestimulus activity trials
    Low vmPFC prestimulus activity trials
    Low RSC prestimulus activity trials
-   Output a threshold mask for each of the conditions above.     
 
@author: wuy19
"""
import HLTP
import numpy as np
import scipy
import nibabel
from nilearn.image import index_img, new_img_like, binarize_img, smooth_img, concat_imgs
from nilearn.input_data import NiftiMasker

# function for transforming p-value to z-score
def z_score(pvalue):
    pvalue = np.minimum(np.maximum(pvalue, 1.e-300), 1. - 1.e-16)
    return scipy.stats.norm.isf(pvalue)

ProjDir = "/isilon/LFMI/VMdrive/data/HLTP_fMRI/"
OutDir = HLTP.group_result + "/nilearn_sl/"

# import the standard brain mask
mask_fname ='/usr/local/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz'
mask_img = nibabel.load(mask_fname)


Seeds = ['CO', 'visual', 'vmPFC', 'RSC']

#Iterate across the four seed ROIs 
for seed in Seeds:
    # Load each subject's decoding accuracy maps from high and low 
    # presitmulus activity trials, and smooth them.
    filesH = []; filesL = []; 
    for sub in HLTP.subjects:
        SubDir = ProjDir + "sub" + str(sub).zfill(2) + "/proc_data/func/"
        SL_Dir = SubDir + "MVPA/SL_decoding/"
        filesH.append(SL_Dir + "SL_high_" + seed + "_cat_cross_PE_" + \
                  str(sub).zfill(2) + "_2standard.nii.gz")
        filesL.append(SL_Dir + "SL_low_" + seed + "_cat_cross_PE_" + \
                  str(sub).zfill(2) + "_2standard.nii.gz")
  
    img4H, img4L = concat_imgs(filesH), concat_imgs(filesL)
    img4H_smoothed,img4L_smoothed  = smooth_img(img4H, 3), smooth_img(img4L, 3)
    del img4H, img4L, filesH, filesL
    
    # Exclude voxels outside the standard brain mask and contrast the included
    # voxels against chance level accuracy. outputs: t and p-maps 
    nifti_masker = NiftiMasker(mask_img = mask_img)
    for level in ['high', 'low']:
        if level =="high":           
            fmri_masked = nifti_masker.fit_transform(img4H_smoothed)        
        elif level =='low':
            fmri_masked = nifti_masker.fit_transform(img4L_smoothed)

        t, p = scipy.stats.ttest_1samp(fmri_masked-0.25, popmean = 0, axis = 0,
                               nan_policy = 'omit', alternative='greater')
        t_map = nifti_masker.inverse_transform(t)
        p_map = nifti_masker.inverse_transform(p)
      
        # transform pval map to z-map.
        p_map = p_map.get_fdata()        
        z_map = z_score(p_map) * (mask_img.get_fdata() == 1)
        z_img = new_img_like(mask_img, z_map)  
        z_fname = OutDir + '/SL_cross_' + level + '_' + seed + '_AgainstChance_3fwhm_zstat.nii.gz'
        z_img.to_filename(z_fname)
        
        # generate a mask that includes voxels with above chace-level accuracy at p < 0.05 (z > 1.64)
        # and save it. 
        con_mask = (z_map > 1.64).astype(int)
        con_mask = new_img_like(mask_img, con_mask)  
        con_mask.to_filename(OutDir + '/SL_cross_' + level + '_' + seed + '_ThresholdingMask.nii.gz')