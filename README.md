# Aligning brains into a shared space improves their alignment to large language models

This repository accompanies a manuscript using a shared response model (SRM; Chen et al., 2015) to isolate stimulus-driven components of electrocorticography (ECoG) signals. We show that using SRM to align neural responses to a natural language stimulus improves encoding model performance using the embeddings from a large language model (LLM; Goldstein et al., 2022). If you find this code useful, please cite the corresponding manuscript:

Bhattacharjee, A., Zada, Z., Wang, H., Aubrey, B., Doyle, W., Dugan, P., Friedman, D., Devinsky, O., Flinker, A., Ramadge, P. J., Hasson, U., Goldstein, A.\*, & Nastase, S. A.\* (2024). Aligning brains into a shared space improves their alignment to large language models. *bioRxiv*. https://doi.org/10.1101/2024.06.04.597448

We provide a Jupyter Notebook with code needed to reproduces all of the results in the paper: [`SRM_encoding.ipynb`](https://github.com/snastase/SRM-Encoding/blob/main/SRM_encoding.ipynb). Since each function takes a considerable amount of time to run at scale, we have added intermediate results files to expedite visualization. 

The preprocessed data used in the manuscript and notebook are available on Zenodo: https://zenodo.org/records/14730569

The raw "Podcast" ECoG dataset is available on OpenNeuro: https://openneuro.org/datasets/ds005574/versions/1.0.0

Subject details, including the subject-wise electrode brain area, coordinate, and type are in [`subject_info.csv`](https://github.com/snastase/SRM-Encoding/blob/main/subject_info.csv).

## Installation

To install the environment used for these analyses (on Linux), clone the repo and then use the following command:

`conda env create -f srm_arnab.yml`

For Mac users, please see the detailed instructions for installing BrainIAK here:
https://brainiak.org/docs/installation.html

If you run into any problems, please raise an [issue](https://github.com/pritamarnab/SRM-Encoding/issues) on the GitHub repo.
