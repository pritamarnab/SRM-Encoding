# **ALIGNING BRAINS INTO A SHARED SPACE IMPROVES THEIR ALIGNMENT TO LARGE LANGUAGE MODELS**

This analysis uses shared response model (SRM) to find a share space across subjects and carries out analysis reported in [paper](https://www.biorxiv.org/content/10.1101/2024.06.04.597448v1.full.pdf) 

Preprocessed data is available [here]( https://zenodo.org/records/14730569).
The full raw dataset is available [here](https://openneuro.org/datasets/ds005574/versions/1.0.0) 

Subjects details including the subject-wise electrode brain area, coordinate and type is in 'subject_info.csv'

The Jupyter Notebook produces all the results in the paper and as each function takes a while for run, we have added the result files for easy plotting.

## Installation

`conda env create -f srm_arnab.yml`

For MAC users, please follow [this](https://github.com/brainiak/brainiak/issues/548) 
