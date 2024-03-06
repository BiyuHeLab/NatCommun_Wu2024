# NatCommun_Wu2024

## Brief overview of analysis scripts

HLTP.py - General definitions, directories, and helper functions.

### Behavior
#### Behavior_statistics.py:
- Perform statistical inference on behavioral metrics including hit rate, false alarm rate sensitivty, criterion, and categorization accuracy.
- Output source data and statistics for Fig 1D-F.  

### Mixed Effects Modeling
#### WholeBrainLMM_data_prep.py:
- Extract GLM residuals within a specific TR relative to the stimulus onsets in each trial and concatenate them into a single file.
- Output a 4D .nifti file containing residuals from a specific time point in each trial.   

#### WholeBrainLMM_prestim_bhv.py:
- Categorize trials to _n_ groups based on the prestimulus activity magnitude. This is done for each voxel separately.
- Compute behavioral metrics for each trial group for each voxel and save the results.   
- Fit Linear Mixed-Effects Models to assess prestimulus activity's effect on perceptual behavior across the whole brain.
- Output whole-brain behavioral metric maps for each subject (available at ).
- Ouput LMM coefficient and p-value maps for each behavioral metric. 

#### WholeBrainLMM_make_zstat_img.py:
- Transform the LMM statistics maps to z-stat maps and perform cluster inference using Gaussian Random Field theory as implemented in FSL.
- Output cluster-corrected statistical maps as shown in Fig 2B-C and Fig S3
- Output un-thresholded statistical maps (available at )
  
#### RoiLMM_prestim_CategorizationAccuracy.py:
- Perform LMM to assess how category accuracy change with varying prestimulus activity within different ROIs, respectively.
- Ouput source data for Fig 6

### TTV and evoked response
Make_sd_maps_based_on_mediansplit.py -      mediansplit trials into two halves and compute sd at each voxel location across each trial group.   

FSL_Evoked_by_prestim_GLM_JobList -         Prepare GLMs for TTV comparisons between high and low prestimulus activity trials and submit them as jobs to HPC SLURM scheduler. 

FSL_Evoked_by_prestim_GLM_RunFSLFeat -      Execute 1st level GLM as implemented in FSL FEAT on HPC

FSL_Evoked_by_prestim_GLM_template.fsf -    Template file for 1st level (single run) GLM specification and estimation

FSL_2ndLevel_Evoked_by_prestim_GLM_template.fsf - Template for 2nd level (across runs for a single subject) GLM specification and specification

FSL_Run2ndLevel_GLM_feat -                  Execute 2nd level GLM as implemented in FSL FEAT 

FSL_GroupInference_design.fsf -             GLM specification for group inference for FSL feat

### SDT Simulation
SDT_simulation.py -                         Run simulation on how SDT behavioral metrics change with varying trial-to-trial variability.
                                            Output ...

### Category Decoding
FSL_CategoryDecoding_GLM_JobList -          Prepare GLMs for Category decoding within high and low prestimulus activity trials and submit them as jobs to HPC SLURM scheduler. 
  
FSL_CategoryDecoding_GLM_RunFSLFeat.py -    Execute 1st level GLM as implemented in FSL FEAT on HPC.
                                            Output run-wise beta estimate for each object category, separately for high and low prestimulus activity trials.
  
FSL_CategoryDecoding_GLM_template.fsf -     Template file for 1st level (single run) GLM specification and estimation
  
SL_Decoding_Main.py -                       Perform whole-brain searchlight category decoding for each subject.
                                            Output subject-level decoding accuracy brain maps for high and low prestimulus activity conditions, respectively.    
  
SL_Decoding_Create_Contrast_Mask.py -       Generate threholding masks that include only voxels showing significant decoding above chance used for the group inference
                                            Output thresholding masks for each condition 

SL_Decoding_GroupLevel.py -                 Compare decoding acuracy maps between high and low prestimulus activity conditions at  the group level using cluster inference
                                            Output unthresholded and thresholded statistical maps 


### Data
Source data and unthresholded statistical brain maps 
