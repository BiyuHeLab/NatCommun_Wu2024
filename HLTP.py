#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 15:16:05 2018

@author: podvae01, wuy19
"""
import numpy as np
import random
import re
from nipype.interfaces import fsl 
import pickle
import socket
import scipy
from scipy import stats
# if doesnt work, run the lines below in terminal and restart spyder


data_dir = '/isilon/LFMI/VMdrive/data/HLTP_fMRI'
group_result = data_dir + '/group_results'
bhv_data_file = group_result +'/behavior/bhv_df.pkl'

subjects = [1,  4,  5,  7,  8,  9, 11, 13, 15, 16, 18, 19, 20, 22, 25, 26, 29,
       30, 31, 32, 33, 34, 35, 37, 38]

random.seed(100)

n_tr_in_block = 227 # note few blocks are missing/ended early

category_id = { "face": 1, "house": 2, "object": 3, "animal": 4}
recognition_id = { "R": 1, "U": -1}

#----- Figures paramters ------------------------------------------------------
hfont = {'fontname':'sans-serif'}

#----- Helper functions -------------------------------------------------------

def save(var, file_name):
    outfile = open(file_name, 'wb')          
    pickle.dump(var, outfile)
    outfile.close()
    
def load(file_name):
    outfile = open(file_name, 'rb')          
    var = pickle.load(outfile)
    outfile.close()
    return var

def tfce_correction(in_file, mask_file):
    #in_file is a 4D image
    rand = fsl.Randomise(in_file = in_file, mask = mask_file, tfce = True, 
                         base_name = "1")
    rand.cmdline
'randomise -i allFA.nii -o "randomise" -d design.mat -t design.con -m mask.nii'

def transform2highres(sub, in_file_name, featname):
    subdir = data_dir +'/sub' + "%02d" % sub + '/proc_data/func'
    regdir = '/block2/' + featname +'/reg'
    flt = fsl.FLIRT()
    flt.inputs.in_file = subdir + '/' + in_file_name + '.nii.gz'
    flt.inputs.out_file = subdir + '/' + in_file_name + '2highres' + '.nii.gz'
    if sub == 1:
        flt.inputs.reference = subdir + regdir + '/highres.nii'#TO BE FIXED
    else: 
        flt.inputs.reference = subdir + regdir + '/highres.nii.gz'
    flt.inputs.output_type = 'NIFTI_GZ'
    flt.inputs.apply_xfm = True
    flt.inputs.in_matrix_file = subdir + regdir + '/example_func2highres.mat'
    flt.cmdline
    return flt.run()

def transform_to_standard(sub, in_file_name, featname = 'block2_preproc.feat'):
    subdir = data_dir +'/sub' + "%02d" % sub + '/proc_data/func'
    regdir = '/block2/' + featname + '/reg'
    flt = fsl.FLIRT()
    flt.inputs.in_file = subdir + '/' + in_file_name + '.nii.gz'
    flt.inputs.out_file = subdir + '/' + in_file_name + '2standadard' + '.nii.gz'
    
    if sub == 1:
        flt.inputs.reference = subdir + regdir + '/standard.nii'#TO BE FIXED
    else: 
        flt.inputs.reference = subdir + regdir + '/standard.nii.gz'
    flt.inputs.output_type = 'NIFTI_GZ'
    flt.inputs.apply_xfm = True
    flt.inputs.in_matrix_file = subdir + regdir + '/highres2standard.mat'
    flt.cmdline
    return flt.run()

def get_block_numbers(sub):
    subdir = data_dir + '/sub' + "%02d" % sub
    param_file = subdir + '/sub_params'
    
    lines = []                 
    with open (param_file, 'r+') as in_file:  
        for line in in_file: 
            lines.append(line.rstrip('\n')) 
    for line in lines:
        if line.find("good_blocks") >=0 :
            break;
    return [int(s) for s in re.findall(r'\b\d+\b', line)]


def get_bhv_vars(df):
    rec_rate = (sum(df['recognition']==1)) / len(df)
    n_rec_real = sum(df[df.real == True].recognition == 1)
    n_real = len(df[(df.real == True) & (df.recognition != 0)])
    n_rec_scr = sum(df[df.real == False].recognition == 1)
    n_scr = len(df[(df.real == False) & (df.recognition != 0)])
    p_correct = df.correct.values.mean()   
    catRT =  df.catRT.values.mean()
    recRT =  df.recRT.values.mean()
    HR, FAR, d, c = get_sdt_msr(n_rec_real, n_real, n_rec_scr, n_scr)
    return rec_rate, HR, FAR, d, c, p_correct, catRT, recRT

def get_sdt_msr(n_rec_signal, n_signal, n_rec_noise, n_noise):
    Z = scipy.stats.norm.ppf
    if (n_noise == 0): FAR = np.nan
    else: FAR = max( float(n_rec_noise) / n_noise, 1. / (2 * n_noise) )
    if  n_signal == 0: HR = np.nan
    else: HR = min( float(n_rec_signal) / n_signal, 1 - (1. / (2 * n_signal) ) )
    d = Z(HR)- Z(FAR)
    c = -(Z(HR) + Z(FAR))/2.
    
    # return nans instead of infs
    if np.abs(d) == np.inf: d = np.nan
    if np.abs(c) == np.inf: c = np.nan
    return HR, FAR, d, c    