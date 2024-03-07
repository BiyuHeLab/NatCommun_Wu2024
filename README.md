# NatCommun_Wu2024

## Brief overview of analysis scripts

HLTP.py - General definitions, directories, and helper functions.
<br>

### Behavior
#### Behavior_statistics.py:
- Performs statistical inference on behavioral metrics (hit rate, false alarm rate sensitivity, criterion, and categorization accuracy).
- Outputs source data and statistics for **Fig 1D-F** (available in the Data folder).
<br>

### Mixed Effects Modeling
#### WholeBrainLMM_data_prep.py:
- Extracts GLM residuals within a specific TR relative to the stimulus onsets in each trial and concatenates them into a single file.
- Outputs 4D .nifti files containing residuals from a specific time point in each trial.   

#### WholeBrainLMM_prestim_bhv.py:
- Categorizes trials to _n_ groups based on the prestimulus activity magnitude. This is done for each voxel separately.
- Computes behavioral metrics for each trial group for each voxel and saves the results.   
- Fits Linear Mixed-Effects Models to assess prestimulus activity's effect on perceptual behavior across the whole brain.
- Outputs whole-brain behavioral metric maps for each subject (available in the Data folder).
- Outputs LMM coefficient and p-value maps for each behavioral metric. 

#### WholeBrainLMM_make_zstat_img.py:
- Transforms the LMM statistics maps to z-stat maps and performs cluster inference using Gaussian Random Field theory as implemented in FSL.
- Outputs cluster-corrected statistical maps as shown in **Fig 2B-C** and **Fig S3** and un-thresholded statistical maps (available in the Data folder).
  
#### RoiLMM_prestim_CategorizationAccuracy.py:
- Performs LMM to assess how category accuracy changes with varying prestimulus activity within different ROIs, respectively.
- Outputs source data for **Fig 6** (available in the Data folder).
<br>

### TTV and evoked response
#### Make_sd_maps_based_on_mediansplit.py:
- Median splits trials into two halves based on the prestimulus activity magnitude in a given ROI
- Computes across-trial standard deviation (SD) of peri-stimulus activity (-1 TR, 0 TR, 1 TR relative to stimulus onset) for each half.
- Outputs peri-stimulus SD maps conditioned by prestimulus activity of specific ROIs, respectively.
### TTV_ClusterInference:
- Performs group-level statistical comparisons between SD derived from high and low prestimulus activity trials.
- Cluster inference performed using Gaussian Rando Field theory as implemented in FSL
- Outputs cluster-corrected statistical maps as displayed in **Fig 3** and **Fig S6** and un-thresholded statistical maps (available in the Data folder).  
<br>

### SDT Simulation
#### SDT_simulation.py:                         
- Simulates how SDT behavioral metrics change with varying trial-to-trial variability.
- Outputs source data for **Fig 4** (available in the Data folder).
<br>

#### FSL_Evoked_by_prestim_GLM_JobList 
- Prepares GLMs for stimulus-evoked responses within high and low prestimulus activity trials and submit them as jobs to HPC SLURM scheduler.
#### FSL_Evoked_by_prestim_GLM_template.fsf
- Template file for 1st level (single run) GLM specification and estimation
#### FSL_Evoked_by_prestim_GLM_RunFSLFeat
- Executes 1st level GLM as implemented in FSL FEAT on HPC
- Outputs 1st-level .feat folders containing beta estimates for stimulus-evoked responses in high and low prestimulus activity trials.
#### FSL_2ndLevel_Evoked_by_prestim_GLM_template.fsf
- Template for 2nd-level GLM specification and specification for stimulus-evoked responses in high and low prestimulus activity trials.
#### FSL_Run2ndLevel_GLM_feat
- Executes 2nd level GLM as implemented in FSL FEAT 
- Outputs 2nd-level .gfeat folders containing across-run averaged estimates for stimulus-evoked responses 
#### FSL_GroupInference_design.fsf
- GLM specification sheet for group inference
- Serves as input file for group inference as implemented in FSL feat FLAME1
- Outputs group-level .gfeat folders containing cluster-corrected statistical maps shown in **Fig S7** and un-thresholded statistical maps (available in the Data folder)  
<br>

### Category Decoding
#### FSL_CategoryDecoding_GLM_JobList
- Prepares GLMs for Category decoding within high and low prestimulus activity trials and submit them as jobs to HPC SLURM scheduler. 
#### FSL_CategoryDecoding_GLM_template.fsf
- Template file for 1st level (single run) GLM specification and estimation  
#### FSL_CategoryDecoding_GLM_RunFSLFeat.py
- Executes 1st level GLM as implemented in FSL FEAT on HPC.
- Outputs run-wise beta estimate for each object category, separately for high and low prestimulus activity trials.
#### SL_Decoding_Main.py
- Performs whole-brain searchlight category decoding for each subject.
- Outputs subject-level decoding accuracy brain maps for high and low prestimulus activity conditions, respectively.    
#### SL_Decoding_Create_Contrast_Mask.py
- Generates thresholding masks that include only voxels showing significant decoding above chance used for the group inference
- Outputs thresholding masks for each condition
#### SL_Decoding_GroupLevel.py
- Compares decoding accuracy maps between high and low prestimulus activity conditions at the group level. Cluster inference performed using Gaussian Random field theory as implemented in FSL 
- Outputs cluster-corrected statistical maps as shown in **Fig 5B** left panel and un-thresholded statistical maps (available in the Data folder).
<br>


## Data
**Source_data_Fig_1.xlsx:** Source data for **Fig 1D-F** and **Fig S1** <br>

**Source_data_Fig_2.xlsx:** Source data for **Fig 2B-C** right panels <br>

**Source_data_Fig_4.xlsx:** Source data for **Fig 4A** <br>

**Source_data_Fig_5.xlsx:** Source data for **Fig 5B** right panel <br>

**Source_data_Fig_6.xlsx:** Source data for **Fig 6** <br>

**Linear Mixed-Effect Modeling (LMM):** Un-thresholded statistical maps for prestimulus activity's effect on behavioral metrics, respectively. Correspond to results shown in **Fig 2B-C** left panel, **Fig S3**
- **/sub#:** Subject-specific folder. Each folder contains four 4-D brain volumes, with each value indicating a behavioral metric calculated from one of the five trial groups based on that voxel's prestimulus activity magnitude, respectively. They serve as the input data for whole-brain LMM. <br>    

**Trial-to-trial Variability:** Un-thresholded statistical maps for trial-to-trial variability at peri-stimulus periods conditioned by the magnitude of prestimulus activity in specific ROIs, respectively. Correspond to results shown in **Fig 3** and **Fig S4**.

**Evoked Response:** Un-thresholded statistical maps for stimulus-evoked responses conditioned by the magnitude of prestimulus activity in specific ROIs, respectively. Related to results shown in **Fig S7**

**Searchlight Category Decoding:**  Un-thresholded statistical maps for decoding accuracy conditioned by the magnitude of prestimulus activity in specific ROIs, respectively. Correspond to results shown in **Fig 5B** left panel.
