#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 14:39:40 2023
Generate statistical maps (z-stat) from the whole-brain LMM on prestimulus
activity's influence on perceptual behavior (c, d', HR, and FAR)
Output:
unthresholded and cluster corrected zstat map (p < 0.05, CDT: p < 0.01)

@author: wuy19
"""

import sys
sys.path.insert(0, '/isilon/LFMI/VMdrive/YuanHao/HLTP_fMRI_Prestimulus')
import os 
os.chdir('/isilon/LFMI/VMdrive/YuanHao/HLTP_fMRI_Prestimulus')
import HLTP
import numpy as np
import scipy
import nibabel
from nilearn import image
from nipype.interfaces import fsl



def z_score(pvalue):
    pvalue = np.minimum(np.maximum(pvalue, 1.e-300), 1. - 1.e-16)
    return scipy.stats.norm.isf(pvalue)


def cluster_correction_grf(p_val_map, fname, CDT):
    mask_fname ='/usr/local/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz'
    #mask_img = nibabel.load(mask_fname)
 
    #z_val_map = z_score(p_vals)
    #z_val_map = z_val_map * (mask_img.get_fdata() == 1) 
    #zmap_img = image.new_img_like(mask_img, z_val_map)  
    z_fname = fname + '_zstat.nii.gz'
    #zmap_img.to_filename(z_fname)
       
    est = fsl.model.SmoothEstimate()
    est.inputs.zstat_file = z_fname
    est.inputs.mask_file = mask_fname
    est_out = est.run()
      
    cl = fsl.model.Cluster()
    cl.inputs.threshold = z_score(CDT) # --thresh/-t
    cl.inputs.in_file = z_fname  # --in/ -i
    cl.inputs.out_threshold_file = fname + '_CDT01_thresh_zstat.nii.gz' # --othresh
    cl.inputs.out_index_file = fname + '_CDT01_thresh_zstat_index.nii.gz' #-o / --oindex
    cl.inputs.out_localmax_txt_file = fname + '_CDT01_stats.txt' # --olmax
    cl.inputs.out_size_file = fname + '_CDT01_thresh_zstat_size.nii.gz'
    cl.inputs.use_mm = True # --mm
    cl.inputs.dlh = est_out.outputs.dlh#
    cl.inputs.volume = est_out.outputs.volume
    cl.inputs.pthreshold = 0.05   
    cl.inputs.minclustersize = True
    cl.inputs.no_table = False  
    cl.cmdline
    cl_out = cl.run()  


def find_voxels_with_negtive_slopes(betas, CDT, fname):
    mask_fname ='/usr/local/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz'
    z_fname = fname + '_zstat.nii.gz'
    mask_img = nibabel.load(mask_fname)
    mask_map = mask_img.get_fdata()
    pos_neg_map = betas * mask_map
    pos_neg_map[pos_neg_map>0]=1
    pos_neg_map[pos_neg_map<0]=-1
    pos_neg_map[mask_map==0]=0
    pos_neg_img = image.new_img_like(mask_img, pos_neg_map)
    #pos_neg_img.to_filename(OutDir + 'pos_neg_zval.nii.gz')

    #threshold = z_score(0.01)
    zstats_img = nibabel.load(z_fname)
    zstats_map = zstats_img.get_fdata()
    #zstats_map[zstats_map <=threshold]=0
    zstats_img = image.new_img_like(mask_img, zstats_map)
    bidir_img = image.math_img("img1 * img2", 
                    img1 = zstats_img,                                 
                    img2 = pos_neg_img)
    bidir_img.to_filename(fname + 'unthresh_bidir_zstat.nii.gz')


    thresh_img = nibabel.load(fname + '_CDT01_thresh_zstat.nii.gz')
    bidir_thresh_img = image.math_img("img1 * img2", 
                    img1 = thresh_img,                                 
                    img2 = pos_neg_img)
    bidir_thresh_img.to_filename(fname + '_CDT01_thresh_bidir_zstat.nii.gz')




model_type= 'LMM'
CDT = 0.01

for bhv_var in ['d', 'c', 'HR', 'FAR']:
    fname = HLTP.data_dir + '/group_results/prestim_to_bhv/prestim_to_' \
    + bhv_var + model_type + '_fwhm5'
    betas, p_vals = HLTP.load(fname)
    cluster_correction_grf(p_vals, fname, CDT)
    find_voxels_with_negtive_slopes(betas, CDT, fname)







