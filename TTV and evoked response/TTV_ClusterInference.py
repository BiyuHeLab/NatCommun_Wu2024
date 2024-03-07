#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 11 15:22:27 2022
@author: wuy19

Perform whole-brain, group-level comparisons of trial-to-trial variability
(indexed by standard deviation) between peri-stimulus activity (-1, 0, + 1 TR 
relative to stimulus onsets) in high and low prestmulus activity trials.

Perform cluster inference using Gaussian Random Field theory as implemented in
FSL

Required input data:
    Subjects' sd brain maps conditioned by prestiulus activity level in
    specific ROIs
    
Output:
    cluster-corrected stastistical maps
    un-thresholded stastistical maps
    cluster index .txt
    cluster size .txt
    local maximum .txt
"""

ProjDir = '/isilon/LFMI/VMdrive/YuanHao/HLTP_fMRI_Prestimulus'
import sys
sys.path.insert(0, ProjDir)


import HLTP
import nibabel
import numpy as np
import scipy
import nilearn
from nilearn import image
from nilearn.input_data import NiftiMasker
from nipype.interfaces import fsl 
from scipy import stats 


def z_score(pvalue):
    pvalue = np.minimum(np.maximum(pvalue, 1.e-300), 1. - 1.e-16)
    return stats.norm.isf(pvalue)

mask_fname = '/usr/local/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz'
mask_img = nibabel.load(mask_fname)
# %% 
ROI = 'salience_c_mask'
measurement = 'std'
time = 'Response2'

# Locating each subjects' SD maps conditioned by prestimulus activity in a
# specific ROI 
ER_fileH = 'ER_masked_' + ROI +  '_High_' + measurement + '_' + time + '_2standadrd.nii.gz'
ER_fileL = 'ER_masked_' + ROI +  '_Low_' + measurement + '_' + time + '_2standadrd.nii.gz'

# Compute the contrast between high and low presrtimulus activity trials for
# each subject 
imdiff = []
for sub in HLTP.subjects:
    subdir = HLTP.data_dir +'/sub' + "%02d" % sub + '/proc_data/func'
    imdiff.append(nilearn.image.math_img(" img1 - img2", 
                               img1 = subdir + '/' + ER_fileH, 
                               img2 = subdir + '/' + ER_fileL))

analysis_name = 'ER_ttest_' + measurement + '_by_' + ROI + '_prestim_no_mask_' + time
img4d = nilearn.image.concat_imgs(imdiff)
img4d = nilearn.image.smooth_img(img4d, 2)


# do 1 samp t-test on each voxel falling in the brain mask
nifti_masker = NiftiMasker(mask_img = mask_img) 
fmri_masked = nifti_masker.fit_transform(img4d)

t, p = scipy.stats.ttest_1samp(fmri_masked, popmean = 0, axis = 0,
                               nan_policy = 'omit', alternative='greater')
t_map = nifti_masker.inverse_transform(t)
p_map = nifti_masker.inverse_transform(p)

t_fname = HLTP.group_result + '/' + analysis_name + '_2fwhm_tstat.nii.gz'
t_map.to_filename(t_fname)
p_fname = HLTP.group_result + '/' + analysis_name + '_2fwhm_uncorrp.nii.gz'
p_map.to_filename(p_fname)

# transform pval map to z-map
p_map = p_map.get_fdata()
pos_z_map = z_score(p_map) * (mask_img.get_fdata() == 1)


pos_z_img = image.new_img_like(template, pos_z_map)  
pos_z_fname = HLTP.group_result + '/' + analysis_name + '_2fwhm_pos_zstat.nii.gz'
pos_z_img.to_filename(pos_z_fname)

neg_z_map = -1 * pos_z_map
neg_z_img = image.new_img_like(template, neg_z_map)
neg_z_fname = HLTP.group_result + '/' + analysis_name + '_2fwhm_neg_zstat.nii.gz'  
neg_z_img.to_filename(neg_z_fname)

#del img4d, t, p, t_map, p_map, t_fname, p_fname, z_map, z_img  
del nifti_masker, fmri_masked
# %% do cluster extent inference using GRF
est = fsl.model.SmoothEstimate()
est.inputs.zstat_file = pos_z_fname
est.inputs.mask_file = mask_fname
est.cmdline
est_out = est.run()
      

cl = fsl.model.Cluster()
cl.inputs.threshold = z_score(0.01) # --thresh/-t
cl.inputs.in_file = pos_z_fname  # --in/ -i
cl.inputs.out_threshold_file = HLTP.group_result + '/' + analysis_name + '_2fwhm_CDT01_thresh_pos_zstat.nii.gz' # --othresh
cl.inputs.out_index_file = HLTP.group_result + '/' + analysis_name + '_2fwhm_CDT01_thresh_pos_zstat_index.nii.gz' #-o / --oindex
cl.inputs.out_localmax_txt_file = HLTP.group_result + '/' + analysis_name + '_2fwhm_CDT01_pos_stats.txt' # --olmax
cl.inputs.out_size_file = HLTP.group_result + '/' + analysis_name + '_2fwhm_CDT01_thresh_pos_zstat_size.nii.gz'
cl.inputs.use_mm = True # --mm
cl.inputs.dlh = est_out.outputs.dlh#
cl.inputs.volume = est_out.outputs.volume
cl.inputs.pthreshold = 0.05   
cl.inputs.minclustersize = True
cl.inputs.no_table = False  
cl.cmdline
cl_out = cl.run()
 
#del cl, cl_out, est, est_out

cl = fsl.model.Cluster()
cl.inputs.threshold = z_score(0.01) # --thresh/-t
cl.inputs.in_file = neg_z_fname  # --in/ -i
cl.inputs.out_threshold_file = HLTP.group_result + '/' + analysis_name + '_2fwhm_CDT01_thresh_neg_zstat.nii.gz' # --othresh
cl.inputs.out_index_file = HLTP.group_result + '/' + analysis_name + '_2fwhm_CDT01_thresh_neg_zstat_index.nii.gz' #-o / --oindex
cl.inputs.out_localmax_txt_file = HLTP.group_result + '/' + analysis_name + '_2fwhm_CDT01_neg_stats.txt' # --olmax
cl.inputs.out_size_file = HLTP.group_result + '/' + analysis_name + '_2fwhm_CDT01_thresh_neg_zstat_size.nii.gz'
cl.inputs.use_mm = True # --mm
cl.inputs.dlh = est_out.outputs.dlh#
cl.inputs.volume = est_out.outputs.volume
cl.inputs.pthreshold = 0.05   
cl.inputs.minclustersize = True
cl.inputs.no_table = False  
cl.cmdline
cl_out = cl.run()
 
del cl, cl_out, est, est_out

# create an image that show significant z values in both direcitons
bidir_img = nilearn.image.math_img("(img1 + -1*img2) * img3", 
                    img1 = HLTP.group_result + '/' + analysis_name + '_2fwhm_CDT01_thresh_pos_zstat.nii.gz',                                 
                    img2 = HLTP.group_result + '/' + analysis_name + '_2fwhm_CDT01_thresh_neg_zstat.nii.gz',
                    img3 = mask_fname)
fname = HLTP.group_result + '/' + analysis_name + '_CDT01_thresh_bidir_zstat.nii.gz'
bidir_img.to_filename(fname)
