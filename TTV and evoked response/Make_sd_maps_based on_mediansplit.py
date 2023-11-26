#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue 5/10/2022 13:41:37

This code does the following:
1.) compute the mean prestimulus activiity (residual data) across voxels
    falling in a given ROI.
 
2.) median split trials based on the magnitude of prestimulus activity
    in a given ROIs.
3.) generate mean and SD poststimulus activity maps for each half

examplary output file:
'ER_masked_visual_d_mask_High_Trials_Response1.nii' 	 
  

@author: wuy19
"""
import HLTP
import numpy as np
import nibabel

import sys
sys.path.insert(0, '/isilon/LFMI/VMdrive/YuanHao/HLTP_fMRI_Prestimulus')
import os 
os.chdir('/isilon/LFMI/VMdrive/YuanHao/HLTP_fMRI_Prestimulus')

# %%#############################################################################

def create_poststimulus_mean_and_std_maps_by_prestim_ROI(ROI_name, *args):
    
    mask = MaskDir + ROI_name + ".nii.gz"
    mask = nibabel.load(mask).get_fdata()
    
    # get prestimulus data from ROI
   
    for sub in HLTP.subjects:
        # load subject-specific prestimulus data and compute the mean activity
        # across voxels falling in a given ROI 
        funcdir = HLTP.data_dir + '/sub' + str(sub).zfill(2) + '/proc_data/func'
        prestim_file = funcdir + '/ER_resid_Baseline12highres2standadrd.nii.gz'
        prestim_img = nibabel.load(prestim_file)
        prestim_data = prestim_img.get_fdata()
        
        # mean activity across ROI voxels for each trial
        roi_data = prestim_data[mask ==1].mean(axis = 0)
        
        
        for time in args:
            evoked_file = funcdir + '/ER_masked_' + time + '.nii.gz'
            evoked_img = nibabel.load(evoked_file)
            evoked_data = evoked_img.get_fdata()
                  
            evoked_map1 = []; evoked_map2 = []
            var_map1 = []; var_map2 = []
            #median-split trials beased on ROI prestimulus activity 
            evoked_map1 = evoked_data[:, :, :, roi_data >= np.median(roi_data)]
            evoked_map2 = evoked_data[:, :, :, roi_data <= np.median(roi_data)]
            # compute the mean activity of each split
            mean_map1 = np.nanmean(evoked_map1, axis =-1)
            mean_map2 = np.nanmean(evoked_map2, axis =-1)
            # compute stdev of activity in each split 
            var_map1 = np.std(evoked_data[:, :, :, roi_data >= np.median(roi_data)], axis=-1)         
            var_map2 = np.std(evoked_data[:, :, :, roi_data <= np.median(roi_data)], axis=-1)
             
        
            # generate brain volumes
            new_img = nibabel.Nifti1Image(evoked_map1, evoked_img.affine, evoked_img.header)
            ER_file = 'ER_masked_' + ROI_name + '_High_Trials_' + time
            nibabel.save(new_img, funcdir + '/' + ER_file + '.nii.gz')
        
            new_img = nibabel.Nifti1Image(evoked_map2, evoked_img.affine, evoked_img.header)
            ER_file = 'ER_masked_'  + ROI_name + '_Low_Trials_' + time
            nibabel.save(new_img, funcdir + '/' + ER_file + '.nii.gz')
        
            new_img = nibabel.Nifti1Image(mean_map1, evoked_img.affine, evoked_img.header)
            ER_file = 'ER_masked_' + ROI_name + '_High_mean_' + time
            nibabel.save(new_img, funcdir + '/' + ER_file + '.nii.gz')
            HLTP.transform_to_standard(sub, ER_file)
        
            new_img = nibabel.Nifti1Image(mean_map2, evoked_img.affine, evoked_img.header)
            ER_file = 'ER_masked_' + ROI_name + '_Low_mean_' + time
            nibabel.save(new_img, funcdir + '/' + ER_file + '.nii.gz')
            HLTP.transform_to_standard(sub, ER_file)
        
            new_img = nibabel.Nifti1Image(var_map1, evoked_img.affine, evoked_img.header)
            ER_file = 'ER_masked_' + ROI_name + '_High_std_' + time
            nibabel.save(new_img, funcdir + '/' + ER_file + '.nii.gz')
            HLTP.transform_to_standard(sub, ER_file)
        
            new_img = nibabel.Nifti1Image(var_map2, evoked_img.affine, evoked_img.header)
            ER_file = 'ER_masked_' + ROI_name + '_Low_std_' + time
            nibabel.save(new_img, funcdir + '/' + ER_file + '.nii.gz')
            HLTP.transform_to_standard(sub, ER_file)
            del evoked_file, evoked_img, evoked_data, evoked_map1, evoked_map2, mean_map1, mean_map2, var_map1, var_map2
        del prestim_data, prestim_file, prestim_img, roi_data, funcdir 

MaskDir = "/isilon/LFMI/VMdrive/YuanHao/HLTP_fMRI_Prestimulus/masks/"
ROI_name = "RSC_d_mask"      
create_poststimulus_mean_and_std_maps_by_prestim_ROI(ROI_name, 'Baseline1', 'Response1', 'Response2')        

        
