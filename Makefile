
# Run all commands in one shell
.ONESHELL:

# NOTE: link data from tigressdata before running any scripts



%-srm: cv = 10
%-srm: srm_k ?= 5
%-srm: lags = $(shell seq -2000 25 2000)

## change the path of the code

## uncomment the analysis to be done, e,g here analysis of a8 is to be run
## if not multiple analyses to be done in a single run, uncomment multiple lines

# %-srm: a1= --encoding_original_neural_data
# %-srm: a2= --encoding_shared_space
# %-srm: a3= --pca_regression
# %-srm: a4= --srm_denoise
# %-srm: a5= --srm_shared_space_generalization
# %-srm: a6= --pca_generalisation_across_subject
# %-srm: a7= --electrode_space_generalization
%-srm: a8= --srm_all_elec
# %-srm: a9= --original_regression_all_elec
# %-srm: a10= --syntactic_feature
# %-srm: a11= --speech_feature


%-srm: CMD = sbatch --job-name=analysis-8 submit.sh
# %-srm: CMD = python


analysis-srm:
	
		$(CMD) /scratch/gpfs/arnab/Encoding/scripts/main.py\
		--cv $(cv) \
		--srm_k $(srm_k) \
		--lags $(lags) \
		$(a1)\
		$(a2)\
		$(a3)\
		$(a4)\
		$(a5)\
		$(a6)\
		$(a7)\
		$(a8)\
		$(a9)\
		$(a10)\
		$(a11)\





		
		



