#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sun Oct 30 22:03:27 2022

Perform Whole-brain searchlight decoding analysis to assess whether the amount
of category-related information in any brain region changes with the
prestimulus activity level of each specific ROI. 

The train data were GLM beta estimates obtained from an independent localizer.
The test data were GLM beta estimates obtained from the main task.

Logistic regression was used as the classification model (c=1)

@author: wuy19
"""

# Determine the seed ROI and from what prestimulus activity condition the data
# were extracted from.     
Seed = 'mPFC'
Prestim = 'high'
###############################################################################


import pandas as pd
import numpy as np
import nibabel
import nilearn.decoding
import time   
import HLTP
from sklearn.model_selection import PredefinedSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

ProjDir = '/isilon/LFMI/VMdrive/data/HLTP_fMRI/'
GLM = '/Pre_' + Seed + '_GLM.feat/stats/'
# Get cope numbers representing recognized face, house, object, and animal
# images in high and low prestimulus activity trials, respectively  
if Prestim == 'high':
    cope_ind = [1,5,9,13] 
elif Prestim == 'low':
     cope_ind = [3,7,11,15]
     

# load the number of trials that went into each condition in each block in each subject  
df = pd.read_pickle('/isilon/LFMI/VMdrive/YuanHao/HLTP_fMRI_Prestimulus/Pre_' + Seed + '_trialnumber.pkl')

# Constrcut a classification model for SL decoding
model = make_pipeline(StandardScaler(), LogisticRegression(C = 1, 
                      multi_class = 'multinomial', solver = 'lbfgs',
                      class_weight='balanced', random_state=42)) 

def decode_category(sub, cope_ind, model, Prestim, Seed): 
    # Locate and load the traing data set (localizer data)  
    train_dir = HLTP.data_dir + '/sub' + "%02d" % sub       
    train_dir = train_dir + '/proc_data/func/loc/mvpa/loc_mvpa_GLM.feat/stats/'
    
    X_train = []           
    for b in range(1,21): #load the 20 PE images as the trainnig set 
        train_img_file = train_dir + 'pe' + str(b) + '_2highres.nii'
        X_train.append(nibabel.load(train_img_file))
        del train_img_file
        #X_train = np.concatenate(X_train)
        # CAUTION!! the indices for the localizer differed from those of the test data 
        #pe1-5: response to animal, 6-10: face, 11-15: house, 16-20: object
        # categories in test data are coded as following: 1: face, 2: house
        # 3: object, 4: animal
        # accordingly, pe1-5 coded as 4, 6-10: 1, 11-15: 2, 16-20: 3
    y_train = np.kron(np.array([4,1,2,3]), np.ones(5)).astype(int)#labels of training set 
    y_train = y_train.tolist()
    
    # Locate and load the test data set (task data)
    subj_df = df.loc[sub]
    subj_df = subj_df.iloc[:, np.subtract(cope_ind, 1)]
    subj_df[subj_df !=0]= 1
    subj_df = subj_df.reset_index()
    SubDir = ProjDir + 'sub' + str(sub).zfill(2) + '/proc_data/func/'   
    mask_img = nibabel.load(ProjDir + 'sub' + str(sub).zfill(2) + \
                            '/proc_data/anat/divt1pd_brain_2mm_mask.nii')
    y_test = []; X_test = [];
    good_blocks = HLTP.get_block_numbers(sub)

    for block in good_blocks:
        BlockDir = SubDir + 'block' + str(block) + GLM
        block_df = subj_df[subj_df['block']==block]
        block_df = block_df.set_index('block')
     
        for label, cope in enumerate(cope_ind):
            test_data = nibabel.load(BlockDir + 'cope' + str(cope) + '_2highres.nii.gz')
            if block_df.iloc[0,label] == 1:             
                #groups.append(block)
                y_test.append(label+1)
                X_test.append(test_data)
            del test_data
        del BlockDir, block_df, label, block, cope
    del subj_df        
             
    test_fold = np.concatenate([-1*np.ones(len(X_train)), np.zeros(len(X_test))])  
    X = X_train + X_test
    y = y_train + y_test 
    del X_train, X_test, y_train, y_test     
    
    # Feed the train and test split into the classification model and run
    # whole-brain searchlight decoding
    ps = PredefinedSplit(test_fold)
    ps.get_n_splits()  
    searchlight_cat = nilearn.decoding.SearchLight(mask_img, 
        radius = 6, n_jobs=-1, verbose=0, 
        estimator = model, cv=ps, scoring = 'balanced_accuracy')   

    # start and end time for a single subject
    start_time = time.time()
    searchlight_cat.fit(X, y)
    end_time = time.time()
    print('Total searchlight duration (including start up time): %.2f' % (end_time - start_time))
      
    fname = 'SL_' + Prestim + '_' + Seed + '_cat_cross_PE_' + str(sub).zfill(2) + '.pkl'   
    HLTP.save(searchlight_cat, '/isilon/LFMI/VMdrive/data/HLTP_fMRI/sub' + \
               str(sub).zfill(2) + '/proc_data/func/MVPA/SL_decoding/' + fname)


for sub in HLTP.subjects:
    print('subject ' + str(sub))
    searchlight_cat = decode_category(sub, cope_ind, model, Prestim, Seed)