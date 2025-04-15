import os
import glob
import pickle
import numpy as np
import pandas as pd  
import csv
from scipy.io import savemat
from scipy.io import loadmat
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from functools import reduce 
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import random
from scipy.stats import ttest_ind, ttest_rel #, permutation_test
from scipy.stats import ttest_1samp
from statsmodels.stats import multitest
from itertools import combinations
from itertools import permutations
from numpy.linalg import inv
from brainiak.funcalign import srm
import argparse
from load_data import *
from utils import *
from config import *



def main():
    """
    Main function to execute the data processing and analysis pipeline.
    """
    args = parse_arguments()
    print(args)
    

    # Set the path for the data
    path = '/scratch/gpfs/arnab/Encoding/'
    os.chdir(path)

    df_all_sub=pd.read_csv('all_subject_sig.csv')
    subjects=np.unique(df_all_sub['subject'])
    lags=args.lags
    cv=args.cv
    srm_k=args.srm_k
    # Load the data
    onsets,word_embeddings=data_create()
    Y_data, elec_num=get_elec_data(subjects, lags, onsets,df_all_sub)

    # Encoding with original neural data
    if args.encoding_original_neural_data:  

        corr_original=[]
        print('Computing original Encoding')

        for qq in range(len(subjects)):
            
            corr_original.append(computing_regression_cv(Y_data[qq,:,:,:], word_embeddings, elec_num[qq],cv))

        path='/scratch/gpfs/arnab/Encoding/result/review_nature_2/'
        filename_mat=path+'encoding_original_neural_data.mat'
        

        savemat(filename_mat,{'corr_original':corr_original,'lags':lags,'subjects':subjects, })

    # Encoding in the Shared Space
    elif args.encoding_shared_space:  

        print('Compute shared space regression')
        srm_corr_cv=np.zeros((len(srm_k),cv,len(lags)))
        for k in range(len(srm_k)):

            srm_corr_cv[k,:,:], shared_space_all_features = srm_regression_cv(Y_data,word_embeddings,elec_num,srm_k[k],cv,subjects)

        path='/scratch/gpfs/arnab/Encoding/result/review_nature_2/'
        filename_mat=path+'encoding_shared_space.mat'
        

        savemat(filename_mat,{'srm_shared_corr_cv':srm_corr_cv,'lags':lags,'subjects':subjects,'shared_space_all_features':shared_space_all_features })

    # PCA Regression
    elif args.pca_regression:  

        print('PCA regression')

        pca_across_sub=pca_across_subject(Y_data,elec_num, subjects,word_embeddings,cv=10,value=5)

        path='/scratch/gpfs/arnab/Encoding/result/review_nature_2/'
        filename_mat=path+'pca_across_sub.mat'
        

        savemat(filename_mat,{'pca_across_subject':pca_across_sub,'lags':lags,'subjects':subjects})

    # Reconstructing Electrode Activity via the Shared Space and doing Encoding
    elif args.srm_denoise:  

        kf = KFold(n_splits=cv)
        p=0
        corr_with_subjects=np.zeros((cv,len(lags),len(subjects)))

        for train_index, test_index in kf.split(word_embeddings):

                print('Fold:',p)
                
                corr_with_subjects[p,:,:]=encoding_with_augmentation(Y_data,word_embeddings,elec_num, train_index, test_index, subjects)
            
                p=p+1

        path='/scratch/gpfs/arnab/Encoding/result/review_nature_2/'
        filename_mat=path+'srm_denoise.mat'
        savemat(filename_mat,{'corr_with_subjects':corr_with_subjects,'lags':lags,'subjects':subjects})

    # SRM Shared Space Generalization
    elif args.srm_shared_space_generalization:  

        kf = KFold(n_splits=cv)
        p=0
        corr_generalization=np.zeros((cv,len(lags),len(subjects)))

        print('Computing srm generalization') 

        for train_index, test_index in kf.split(word_embeddings):

            print('Fold:',p)
            
            corr_generalization[p,:,:]=srm_generalization(Y_data,word_embeddings,elec_num, train_index, test_index,subjects)

            p=p+1

        path='/scratch/gpfs/arnab/Encoding/result/review_nature_2/'
        filename_mat=path+'srm_shared_space_generalization.mat'
        

        savemat(filename_mat,{'srm_generalization':corr_generalization,'lags':lags,'subjects':subjects,})

    # pca generalisation across subject
    elif args.pca_generalisation_across_subject:

        kf = KFold(n_splits=cv)
        p=0
        pca_corr_generalization=np.zeros((cv,len(lags),len(subjects)))

        print('Computing pca generalization') 

        for train_index, test_index in kf.split(word_embeddings):

            print('Fold:',p)
            
            pca_corr_generalization[p,:,:]=pca_generalization_across_subject(Y_data,word_embeddings,elec_num, train_index, test_index,subjects)

            p=p+1

        path='/scratch/gpfs/arnab/Encoding/result/review_nature_2/'
        filename_mat=path+'pca_generalization_across_sub.mat'
        savemat(filename_mat,{'pca_generalization_across_sub':pca_corr_generalization,'lags':lags,'subjects':subjects,})

    # Elctrode Space Generalization via SRM
    elif args.electrode_space_generalization:

        kf = KFold(n_splits=cv)
        p=0
        corr_leave_one_out=np.zeros((cv,len(lags),len(subjects)))

        print('Computing Elec Space Generalization') 

        for train_index, test_index in kf.split(word_embeddings):

            print('Fold:',p)
            
            corr_leave_one_out[p,:,:]=srm_leave_one_out(Y_data,word_embeddings,elec_num, train_index, test_index,subjects)

            p=p+1

        path='/scratch/gpfs/arnab/Encoding/result/review_nature_2/'
        filename_mat=path+'electrode_space_generalization.mat'
        savemat(filename_mat,{'srm_leave_one_out':corr_leave_one_out,'lags':lags,'subjects':subjects,})

    # Localizing Improvement across brain areas
    elif args.srm_all_elec:

        kf = KFold(n_splits=cv)
        p=0
        total_elec=sum([iterator for iterator  in elec_num])
        corr_with_subjects1=np.zeros((cv,len(lags),total_elec))

        print('Computing SRM All Elec')  

        for train_index, test_index in kf.split(word_embeddings):

            print('Fold:',p)
            
            corr_with_subjects1[p,:,:]=srm_regression_all_elec(Y_data,word_embeddings,elec_num, train_index, test_index,subjects)

            p=p+1

        corr_with_subjects_all_elec=corr_with_subjects1[0,:,:]
        for ww in range(1,cv):
            corr_with_subjects_all_elec=corr_with_subjects_all_elec+corr_with_subjects1[ww,:,:]

        corr_with_subjects_all_elec=corr_with_subjects_all_elec/cv

        path='/scratch/gpfs/arnab/Encoding/result/review_nature_2/'
        filename_mat=path+'srm_all_elec.mat'
        savemat(filename_mat,{'srm_regression_all_elec':corr_with_subjects_all_elec,'srm_regression_all_elec_cv':corr_with_subjects1,'lags':lags,'subjects':subjects,})

    # original regression with all electrodes
    elif args.original_regression_all_elec:
        corr_original_all_elec1=[]

        print('Computing Original Regression All Elec')
            
        for qq in range(len(subjects)):
            
            final_corr1,final_corr=original_regression_all_elec(Y_data[qq,:,:,:], word_embeddings, elec_num[qq],cv)

            corr_original_all_elec1.append(final_corr1)
            # a=corr_original.append(original_regression_all_elec(Y_data[qq,:,:,:], word_embeddings, elec_num[qq],cv))
            
            if qq==0:
                corr_original_all_elec_cv=final_corr
            else:
                corr_original_all_elec_cv=np.concatenate((corr_original_all_elec_cv,final_corr),axis=1)


        corr_original_all_elec=np.asarray(corr_original_all_elec1[0])
        for qq in range(1,len(subjects)):
            corr_original_all_elec=np.concatenate((corr_original_all_elec,np.asarray(corr_original_all_elec1[qq])), axis=0)

        path='/scratch/gpfs/arnab/Encoding/result/review_nature_2/'
        filename_mat=path+'original_regression_all_elec.mat'
        savemat(filename_mat,{'corr_original_all_elec':corr_original_all_elec,'corr_original_all_elec_cv':corr_original_all_elec_cv,'lags':lags,'subjects':subjects,})

    elif args.syntactic_feature:

        onsets,word_embeddings=syntactic_embeddings()
        Y_data, elec_num=get_elec_data(subjects, lags, onsets,df_all_sub)

        print('Compute shared space regression')
        srm_corr_cv=np.zeros((len(srm_k),cv,len(lags)))
        for k in range(len(srm_k)):

            srm_corr_cv[k,:,:], shared_space_all_features = srm_regression_cv(Y_data,word_embeddings,elec_num,srm_k[k],cv,subjects)

        path='/scratch/gpfs/arnab/Encoding/result/review_nature_2/'
        filename_mat=path+'encoding_syntactic_emb.mat'
        

        savemat(filename_mat,{'srm_shared_corr_cv':srm_corr_cv,'lags':lags,'subjects':subjects,'shared_space_all_features':shared_space_all_features })

    elif args.speech_feature:

        onsets,word_embeddings=get_speech_embeddings()
        Y_data, elec_num=get_elec_data(subjects, lags, onsets,df_all_sub)

        print('Compute shared space regression')
        srm_corr_cv=np.zeros((len(srm_k),cv,len(lags)))
        for k in range(len(srm_k)):

            srm_corr_cv[k,:,:], shared_space_all_features = srm_regression_cv(Y_data,word_embeddings,elec_num,srm_k[k],cv,subjects)

        path='/scratch/gpfs/arnab/Encoding/result/review_nature_2/'
        filename_mat=path+'encoding_speech_emb.mat'
        

        savemat(filename_mat,{'srm_shared_corr_cv':srm_corr_cv,'lags':lags,'subjects':subjects,'shared_space_all_features':shared_space_all_features })


    elif args.different_layer:

        path='/scratch/gpfs/arnab/Encoding/all_layer/'
        os.chdir(path)

        onsets=np.squeeze(loadmat('All_layer_embeddings_gpt.mat')['onset'])
        word_embeddings2=loadmat('All_layer_embeddings_gpt.mat')['embeddings'][args.layer_id[0],:,:]

        pca = PCA(n_components=50)
        word_embeddings2=word_embeddings2- np.mean(word_embeddings2, axis=0)
        word_embeddings=pca.fit_transform(word_embeddings2)
        word_embeddings = normalize(word_embeddings, axis=1, norm='l2')

        
        Y_data, elec_num=get_elec_data(subjects, lags, onsets,df_all_sub)

        corr_original=[]
        print('Computing original Encoding')

        for qq in range(len(subjects)):
            
            corr_original.append(computing_regression_cv(Y_data[qq,:,:,:], word_embeddings, elec_num[qq],cv))


        print('Compute shared space regression')
        srm_corr_cv=np.zeros((len(srm_k),cv,len(lags)))
        for k in range(len(srm_k)):

            srm_corr_cv[k,:,:], shared_space_all_features = srm_regression_cv(Y_data,word_embeddings,elec_num,srm_k[k],cv,subjects)


        path='/scratch/gpfs/arnab/Encoding/result/review_nature_2/different_layer/'
        filename_mat=path+'result_layer_'+str(args.layer_id[0])+'.mat'
        
        savemat(filename_mat,{'corr_original':corr_original,'srm_shared_corr_cv':srm_corr_cv,'lags':lags,'subjects':subjects, })

    elif args.different_size:

        path='/scratch/gpfs/arnab/Encoding/different_size/'
        os.chdir(path)

        emb_file=args.model_size+'.mat'

        onsets=np.squeeze(loadmat(emb_file)['onset'])
        word_embeddings2=loadmat(emb_file)['embeddings']


        pca = PCA(n_components=50)
        word_embeddings2=word_embeddings2- np.mean(word_embeddings2, axis=0)
        word_embeddings=pca.fit_transform(word_embeddings2)
        word_embeddings = normalize(word_embeddings, axis=1, norm='l2')

        
        Y_data, elec_num=get_elec_data(subjects, lags, onsets,df_all_sub)

        corr_original=[]
        print('Computing original Encoding')

        for qq in range(len(subjects)):
            
            corr_original.append(computing_regression_cv(Y_data[qq,:,:,:], word_embeddings, elec_num[qq],cv))


        print('Compute shared space regression')
        srm_corr_cv=np.zeros((len(srm_k),cv,len(lags)))
        for k in range(len(srm_k)):

            srm_corr_cv[k,:,:], shared_space_all_features = srm_regression_cv(Y_data,word_embeddings,elec_num,srm_k[k],cv,subjects)


        path='/scratch/gpfs/arnab/Encoding/result/review_nature_2/different_size/'

        filename_mat=path+'result_'+emb_file

        savemat(filename_mat,{'corr_original':corr_original,'srm_shared_corr_cv':srm_corr_cv,'lags':lags,'subjects':subjects, })

        



    return None

if __name__ == "__main__":
    main()





