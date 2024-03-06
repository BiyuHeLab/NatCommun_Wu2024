#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 14:51:36 2019

Performing group-level inference on a given ROI prestimulus activity
level's on object decoding accuracy across the whole brain.
    - Voxel-wise contrasting decoding accuracy maps derived from high and low
    prestimulus activity trials.
    - Cluster inference using Gaussian Random Field Theory as implemented in
    FSL CLUSTER
    
Required input data:
    - Subjects' whole-brain decodng accuracy maps in standard space
    
Output:
    - Unthresholded statistical maps
    - Cluster-corrected statistical maps
    - cluster size .txt
    - local maximum .txt
    - cluster index .txt
    
@author: podvae01, wuy19
"""
#######################
#PARAMTER SETTING
#######################
seed = "OTC"
data_type = 'PE'

import HLTP
#import pandas as pd
import numpy as np
import nibabel
from nilearn import image
from nilearn.image import index_img, new_img_like, concat_imgs, math_img, smooth_img
import scipy.stats as stats
from nipype.interfaces import fsl 
from nilearn.input_data import NiftiMasker

def z_score(pvalue):
    pvalue = np.minimum(np.maximum(pvalue, 1.e-300), 1. - 1.e-16)
    return stats.norm.isf(pvalue)

ProjDir = "/isilon/LFMI/VMdrive/data/HLTP_fMRI/"
OutDir = HLTP.group_result + "/nilearn_sl/"
mask_fname ='/usr/local/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz'
mask_img = nibabel.load(mask_fname)
#%% ##########################################################################
# Load subjects' accuracy maps obtained from high/low prestimulus activity
# trials

filesH = []; filesL = []; 
for sub in HLTP.subjects:
    SubDir = ProjDir + "sub" + str(sub).zfill(2) + "/proc_data/func/"
    SL_Dir = SubDir + "MVPA/SL_decoding/"
    filesH.append(SL_Dir + "SL_high_" + seed + "_cat_cross_" + data_type +  "_" + \
                  str(sub).zfill(2) + "_2standard.nii.gz")
    filesL.append(SL_Dir + "SL_low_" + seed + "_cat_cross_" + data_type +  "_" + \
                  str(sub).zfill(2) + "_2standard.nii.gz")
  
img4H, img4L = concat_imgs(filesH), concat_imgs(filesL)

# Compute contrast image between high and low prestimulus condition
img4d_pos = math_img("img1 - img2", img1 = img4H, img2 = img4L)
img4d_pos = smooth_img(img4d_pos, 3)
del filesH, filesL, img4H, img4L

nifti_masker = NiftiMasker(mask_img = mask_img)    # whole-brain mask 
fmri_masked = nifti_masker.fit_transform(img4d_pos)


# Compute t-tests. the unthresholded zstat maps for both positive and negative
# effects are essentially the same. Thet were computed separately for cluster
# inference purpose.

for effect in ['pos', 'neg']:  
    if effect =="pos":
        t, p = stats.ttest_1samp(fmri_masked, popmean = 0, axis = 0,
                               nan_policy = 'omit', alternative='greater')
    else:
        t, p = stats.ttest_1samp(fmri_masked, popmean = 0, axis = 0,
                               nan_policy = 'omit', alternative='less')

    p_map = nifti_masker.inverse_transform(p)   
    p_map = p_map.get_fdata()
 
    z_map = z_map = z_score(p_map) * (mask_img.get_fdata() == 1)
    z_img = image.new_img_like(mask_img, z_map)  
    z_fname = OutDir + '/SL_cross_' + seed + '_PE_imdiff_' + effect + '_nomask_3fwhm_zstat.nii.gz'
    z_img.to_filename(z_fname)
    del z_map, z_img, z_fname
       
    est = fsl.model.SmoothEstimate()
    est.inputs.zstat_file = OutDir + 'SL_cross_' + seed + '_PE_imdiff_' + effect + '_nomask_3fwhm_zstat.nii.gz'
    est.inputs.mask_file = mask_fname
    est.cmdline
    est_out = est.run()
      
    cl = fsl.model.Cluster()
    cl.inputs.threshold = z_score(0.01) # --thresh/-t
    cl.inputs.in_file = OutDir + 'SL_cross_' + seed + '_PE_imdiff_' + effect + '_nomask_3fwhm_zstat.nii.gz'
    cl.inputs.out_threshold_file = OutDir + 'SL_cross_' + seed + '_PE_imdiff_' + effect + '_nomask_3fwhm_CDT01_0025_thresh_zstat.nii.gz' # --othresh
    cl.inputs.out_index_file = OutDir + 'SL_cross_' + seed + '_PE_imdiff_' + effect + '_nomask_3fwhm_CDT01_0025_thresh_zstat_index.nii.gz' #-o / --oindex
    cl.inputs.out_localmax_txt_file = OutDir + 'SL_cross_' + seed + '_PE_imdiff_' + effect + '_nomask_3fwhm_0025_CDT01_stats.txt' # --olmax
    cl.inputs.out_size_file = OutDir + 'SL_cross_' + seed + '_PE_imdiff_' + effect + '_nomask_3fwhm_CDT01_0025_thresh_zstat_size.nii.gz'
    cl.inputs.use_mm = True # --mm
    cl.inputs.dlh = est_out.outputs.dlh#
    cl.inputs.volume = est_out.outputs.volume
    cl.inputs.pthreshold = 0.025   
    cl.inputs.minclustersize = True
    cl.inputs.no_table = False  
    cl.cmdline
    cl_out = cl.run()
    
    
    
    