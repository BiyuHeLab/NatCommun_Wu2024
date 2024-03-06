#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 10:51:42 2020

Prepare data for the main analysis on prestimulus activity's influence on
behavioral metrics. 
Extract data from a specific TR relative to the stimulus onset for each trial
and concatenate them to one signle nifti file.

Output:
    4-D brain maps for each subject, with the first three dimenstions represent
    the dimensions of the 3D brain volume and the 4th dimension indicates the
    number of completed trials.
    Each brain volume contains the residuals estimated from a specific TR
    relative to the stimulus onset.
    e.g., ER_resid_Baseline1 indicates 1 TR before the stimulus onset 
    native, highres T1, and standard space
    
@author: podvae01, wuy19
"""
import sys
import os
ProjDir='/isilon/LFMI/VMdrive/YuanHao/HLTP_fMRI_Prestimulus'
sys.path.insert(1, ProjDir)
os.chdir(ProjDir)

from joblib import Parallel, delayed
import multiprocessing
import HLTP
import numpy as np
import nibabel
from nilearn import image
import pandas as pd
import scipy as sp
 
sourcedir = '/isilon/LFMI/archive/Projects/2021_NatComms_HLTP_fMRI_Max/data'
func_file_name = 'conditions_realscrGLM.feat/stats/res4d.nii.gz'; tag = 'resid'
#func_file_name = 'denoised_data2highres.nii.gz'; tag = 'full' # already smoothed 3 mm 
num_cores = multiprocessing.cpu_count() - 2
events_file = '/evs/visual_ev.txt'
TRtimes = {'Baseline2':-2, 'Baseline1':-1, 
           'Response1':0, 'Response2':1, 'Response3':2}
bhv_df = pd.read_pickle(HLTP.group_result + '/behavior/corrected_bhv_df.pkl')
# %%
def save_event_related_files(sub, funcdir, anatdir, targetdir, events_file, TRtimes, tag):
    
    # Locating the 4D residual brain maps 
    block_N = HLTP.get_block_numbers(sub)
    if tag == "resid":   
        block_files = [(funcdir + '/block' + str(b) + '/' + func_file_name)
                   for b in block_N]
    else: 
        block_files = [(funcdir + '/block' + str(b) + '/block' + str(b) + 
                    '_preproc.feat/' + func_file_name) for b in block_N]
    
    # Extracting simulus presentation triggers:
    events = []
    for block in block_N:
        event_file = funcdir + '/block' + str(block) + events_file
        events.append(np.where(np.loadtxt(event_file))[0]) 
    
    # Using a whole-brain mask to exclude voxels outside the standard brain    
    #if tag == 'full':
    #    mask_file_name = 'divt1pd_brain_2mm_mask.nii'
    #    mask_img = nibabel.load(anatdir + '/' + mask_file_name
    #                        ).get_fdata().astype('bool')
    
    # get data blocks
    # not sure this is efficient, think there was some nilearn builtin func
    block_ER_data = {key: [] for key in TRtimes.keys()}
    for block_n in range(len(block_files)):
        print(block_n)
        nifti_data = nibabel.load(block_files[block_n])
        data = nifti_data.get_fdata()
        if tag == 'resid':
            mask_img = data.mean(axis = -1) != 0

        # detrending is crucial for raw data, not sure matters for residual
        data[mask_img==False] = 0
        data[mask_img, :] = sp.signal.detrend(data[mask_img, :], axis = -1)
        # in case recording stopped early by accident:
        events[block_n] = events[block_n][ events[block_n] < data.shape[3] ]

        for key, t in TRtimes.items():
            block_ER_data[key].append(data[:, :, :, events[block_n] + t])
            
    # Concatenate the trials and save nifti file for each TR 
    img = image.load_img(block_files[0])
    for key in TRtimes.keys():
        ER_data = np.concatenate(block_ER_data[key], axis = 3)
        new_img = nibabel.Nifti1Image(ER_data, img.affine, img.header)
        ER_file = 'ER_' + tag + '_' + key
        nibabel.save(new_img, targetdir + '/' + ER_file + '.nii.gz')
        
        # transform the images to standard for trial-based group analysis
        if tag == 'resid':
            HLTP.transform2highres(sub, ER_file, 'conditionsGLM.feat')
            HLTP.transform_to_standard(sub, ER_file + '2highres')
        else:
            HLTP.transform_to_standard(sub, ER_file)
   

def subj_proc(sub, func_file_name, events_file, TRtimes, bhv_df, tag):
    # Prepare and save peri-stimulus data for each subject
    print("processing subject #", str(sub))
    subjdir = sourcedir +'/sub' + str(sub).zfill(2) + '/proc_data'
    funcdir = subjdir + '/func';   anatdir = subjdir + '/anat'
    targetdir = HLTP.data_dir +'/sub' + str(sub).zfill(2) + '/proc_data/func'
    # Prepare files that include only certain TR time-locked to stimulus onset
    save_event_related_files(sub, funcdir, anatdir, targetdir, events_file, TRtimes, tag)    
    # Calculate the mean images for each behavioral condition
#    save_condition_average(sub, bhv_df, targetdir, TRtimes, tag)
# %%
#for sub in HLTP.subjects:
#    subj_proc(sub, func_file_name, events_file, TRtimes, bhv_df, tag)
    
Parallel(n_jobs = num_cores)(delayed(subj_proc)(
     sub, func_file_name, events_file, TRtimes, bhv_df, tag) 
     for sub in HLTP.subjects) 