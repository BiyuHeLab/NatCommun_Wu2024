# Prestim_fMRI_Wu_2024

## Brief overview of analysis scripts

HLTP.py - General definitions, directories, and helper functions.

### Behavior
Behavior_statistics.py - Behavioral analysis.

### Mixed Effects Modeling
WholeBrainLMM_data_prep.py - Extract BOLD activity within a specific time window relative to the stimulus onsets in each run and concatenate them into a single file.

WholeBrainLMM_prestim_bhv.py - Fit Linear Mixed-Effects Models to assess prestimulus activity's effect on perceptual behavior across the whole brain.

WholeBrainLMM_make_zstat_img.py - Transform the LMM statistics to z-stats map and perform cluster inference.

RoiLMM_prestim_CategorizationAccuracy.py - ROI-based LMM assessing prestimulus activity's influence on category accuracy.

### SDT Simulation
SDT_simulation.py - Simulation on how changes in trial-to-trial variability influence Signal Detection Theory (SDT) behavioral metrics.

### Category Decoding
FSL_CategoryDecoding_GLM_JobList
  
FSL_CategoryDecoding_GLM_RunFSLFeat.py - Run FSL Feat.
  
FSL_CategoryDecoding_GLM_template.fsf - GLM specification template.
  
SL_Decoding_Main.py - Perform whole-brain searchlight category decoding for each subject.
  
SL_Decoding_GroupLevel.py - Cluster inference at the group level.
  
SL_Decoding_Create_Contrast_Mask.py - Generate a mask that includes only voxels showing significant decoding above chance.

### Source Data
Source data are available at [https://nyulangone-my.sharepoint.com/...](https://nyulangone-my.sharepoint.com/:f:/g/personal/yuanhao_wu_nyulangone_org/EvjGYXl2ovZHjW0uGB6vpGQBSfr4VAGVHi2Qk7sE5AGn4g?e=EPpsUS)
