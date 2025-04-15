
# Run all commands in one shell
.ONESHELL:

# NOTE: link data from tigressdata before running any scripts



%-srm: cv = 10
%-srm: srm_k ?= 5
%-srm: lags = $(shell seq -2000 25 2025)

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


%-layers: cv = 10
%-layers: srm_k ?= 5
%-layers: lags = $(shell seq -2000 25 2025)
%-layers: layer_id = $(shell seq 1 1 48)
%-layers: a12= --different_layer

%-layers: CMD = sbatch --job-name=analysis-$$layer-layers submit.sh
# %-layers: CMD = python


analysis-layers:

		for layer in $(layer_id); do \
	
			$(CMD) /scratch/gpfs/arnab/Encoding/scripts/main.py\
			--cv $(cv) \
			--srm_k $(srm_k) \
			--lags $(lags) \
			--layer_id $$layer \
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
			$(a12);\
		done

%-size: cv = 10
%-size: srm_k ?= 5
%-size: lags = $(shell seq -2000 25 2025)
# %-size: layer_id = $(shell seq 1 1 48)
%-size: a12= --different_size

%-size: model_size = gpt2-large gpt-neox-20b gpt-neo-125M gpt-neo-2.7B gpt-neo-1.3B glove50

%-size: CMD = sbatch --job-name=analysis-$$model-size submit.sh
# %-size: CMD = python

analysis-size:

		for model in $(model_size); do \
	
			$(CMD) /scratch/gpfs/arnab/Encoding/scripts/main.py\
			--cv $(cv) \
			--srm_k $(srm_k) \
			--lags $(lags) \
			--model_size $$model \
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
			$(a12);\
		done







%-encoding: batch_size = 10
%-encoding: hidden_layer_num= 3
%-encoding: sub_id = 0   # 661 717 723 741 742 743 763 798
%-encoding: lags ?= 0 #0 50 100 #
%-encoding: EPOCHS= 1000  ### [54,78,24,15] for [625,676,7170,798] 
%-encoding: train_num= 4500

# %-srm: across_3= --across_subject_with_repacing_srm
# %-srm: select_elec= --selected_elec_id
# %-srm: folds = 10
# %-srm: desired_fold = 0
%-encoding: CMD = sbatch --job-name=deep_enc-hidden_layer-$(hidden_layer_num) submit.sh
# %-encoding: CMD = python 

# %-srm: JOB_NAME = $(subst /,-,$(desired_fold))
# %-srm: CMD = sbatch --job-name=$(production)-$(JOB_NAME)-across submit.sh


deep-encoding:

		$(CMD) /scratch/gpfs/arnab/Encoding/encoding_deep_learning.py\
		--batch_size $(batch_size) \
		--hidden_layer_num $(hidden_layer_num) \
		--sub_id $(sub_id)\
		--lags $(lags)\
		--EPOCHS $(EPOCHS)\
		--train_num $(train_num)\
		
		

# --desired_fold $(desired_fold)


%-select: subjects ?= 7170 # [625,676,7170,798] 
%-select: lags ?= 0  #0 50 100 #
%-select: total_conv_numbers ?= 24  ### [54,78,24,15] for [625,676,7170,798] 
%-select: total_iteration ?= 5000  ### 5000
# %-select: production = 0       ## 1 means production, 0 means comprehension
%-select: CMD = sbatch --job-name=elec-select-$(subjects) submit.sh
# %-select: CMD = python 

elec-select:

		$(CMD) /scratch/gpfs/arnab/model_based_srm/variance_based_elec_selection.py\
		--subject $(subjects) \
		--total_iteration $(total_iteration)\
		--total_conv_numbers $(total_conv_numbers)\
		--lags $(lags)



