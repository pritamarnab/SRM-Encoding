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
from functools import reduce 
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import random
import scipy.stats
from itertools import combinations
from itertools import permutations
from numpy.linalg import inv
from scipy.stats import ttest_ind, ttest_rel #, permutation_test
from scipy.stats import ttest_1samp
from statsmodels.stats import multitest
from brainiak.utils import utils
from brainiak.funcalign import srm
from brainiak.funcalign import rsrm
import brainiak


def data_create():

    path='/scratch/gpfs/arnab/Encoding/'
    os.chdir(path)

    pickle_file = open("./777_full_gpt2-xl_cnxt_1024_layer_48_embeddings.pkl", "rb")
    objects = []
    
    i=0
    
    while True:
        print('i',i)
        try:
    
            objects.append(pickle.load(pickle_file))
    
        except EOFError:
    
            break
    
    pickle_file.close()
    
    a=objects[0]
    
    #l=5996
    l=5013  #5000
    q=1600
    word_embeddings2=np.zeros((l,q))
    onset=np.zeros((l))
    selected_word=np.zeros((l))
    token_id=np.zeros((l))
    token_idx=np.zeros((l))
    word_index=np.zeros((l))
    is_stop=np.zeros((l))
    selected_word=[]
    pred_prob=np.zeros((l))
    
    p=0
    for i in range(len(a)):
        
              
      #5000 words    
      if (str(a[i]['datum_word'])!='None' and str(a[i]['adjusted_onset'])!='None' and str(a[i]['embeddings'])!='None' and p<l+4 and a[i]['token_idx']==0):
    
        #print(i)
        if p>=4:  #first 4 words had 'None' into it, hence left those
          word_embeddings2[p-4,:]=a[i]['embeddings']
          onset[p-4]=int(a[i]['adjusted_onset'])
          selected_word.append(a[i]['datum_word'])
          token_id[p-4]=a[i]['token_id']
          token_idx[p-4]=a[i]['token_idx']
          #word_index[p-4]=i
          #word_index[p-4]=int(a[i]['index'])
#           pred_prob[p-4]=a[i]['true_pred_prob']  
#           s=0
#           for j in range(179):
#               if a[i]['datum_word']==stops[j]:
#                   s=1
                  
#           is_stop[p-4]=s   
                  
          
          
          
        p=p+1
    
    pca = PCA(n_components=50)
    word_embeddings2=word_embeddings2- np.mean(word_embeddings2, axis=0)
    word_embeddings=pca.fit_transform(word_embeddings2)
    word_embeddings = normalize(word_embeddings, axis=1, norm='l2')
  
   
    return onset,word_embeddings   #, is_stop,pred_prob


def get_elec_data(subjects, lags, onsets,df_all_sub):
   
    electrodes1=[]
    elec_num=[]
    for qq in range(len(subjects)):
        
        subject=subjects[qq]
        #print('subject:',subject)
        s=np.array(df_all_sub.loc[df_all_sub.subject==subject].matfile)
        electrodes1.append(s)
        elec_num.append(len(s))

    Y_data= np.zeros((len(subjects),len(lags),len(onsets),max(elec_num))) #45 is the electrode number here

    path='/scratch/gpfs/arnab/Encoding/'
    os.chdir(path)

    for qq in range(len(subjects)):
    
        subject=subjects[qq]
        print('subject:',subject)
        
        electrodes=electrodes1[qq]
        # elec_num.append(len(electrodes))
        
        if qq==0:
            path=os.path.join(os.pardir,'significant_electrode_podcast',str(subject))
        else:
            path=os.path.join(os.pardir,str(subject))

        os.chdir(path)
        
        ecogs=[]
        for i in electrodes:
            filename='NY'+str(subject)+'_111_Part1_conversation1_electrode_preprocess_file_'+str(i)+'.mat'
            
            e=loadmat(filename)['p1st'].squeeze().astype(np.float32)
            ecogs.append(e)
            
        ecogs = np.asarray(ecogs).T

        window_size=200

        half_window = round((window_size / 1000) * 512 / 2)
        #t = len(brain_signal)
        t=len(ecogs[:,0])
    
        Y= np.zeros((len(onsets), np.shape(ecogs)[1]))
        
            
        for ii in range(len(lags)):
            Y1 = np.zeros((len(onsets), 2 * half_window + 1))
            #print(ii)
            for k in range(np.shape(ecogs)[1]):
        
                brain_signal=ecogs[:,k]
                #from numba import jit, prange
                
                # for lag in prange(len(lags)):
                
                lag_amount = int(lags[ii]/ 1000 * 512)
                
                index_onsets = np.minimum(
                    t - half_window - 1,
                    np.maximum(half_window + 1,
                                np.round_(onsets, 0, onsets) + lag_amount))
                
                # subtracting 1 from starts to account for 0-indexing
                starts = index_onsets - half_window - 1
                stops = index_onsets + half_window
                
                for i, (start, stop) in enumerate(zip(starts, stops)):
                    start=int(start)
                    stop=int(stop)
                    Y1[i, :] = brain_signal[start:stop].reshape(-1)
                    
                                    
                #if subject==717:
                
                Y_data[qq,ii,:,k] = np.mean(Y1, axis=-1)



    for k in range(np.shape(Y_data)[0]):
        for k1 in range(np.shape(Y_data)[1]):
            for k2 in range(elec_num[k]):
                Y_data[k,k1,:,k2]=stats.zscore(Y_data[k,k1,:,k2])

    return Y_data, elec_num


def syntactic_embeddings():

    path_ken1="/scratch/gpfs/kw1166/247/247-pickling/results/podcast/777/pickles/embeddings/"
    model_size='symbolic-lang-new'
    path_ken=path_ken1+model_size+'/full/'

    path_emb=path_ken+'cnxt_0001/'
    os.chdir(path_emb)

    emb_file='layer_00.pkl'

    pickle_file = open(emb_file, "rb")
    objects = []

    i=0

    while True:
        print('i',i)
        try:

            objects.append(pickle.load(pickle_file))

        except EOFError:

            break

    pickle_file.close()

    a=objects[0]

    df_emb=pd.DataFrame(a)

    df3=df_emb.loc[~df_emb.adjusted_onset.isnull()]# & (df.token_idx==0)] # and df.token_idx==1

    onsets=np.unique(np.squeeze(df3.adjusted_onset.values))
    w=df3.embeddings.values

    # word_embeddings2=np.zeros((len(w),np.shape(w[0])[0]))
    # for i in range(len(w)):
    #     word_embeddings2[i,:]=w[i]
        
    # onsets=np.unique(df.adjusted_onset.values)

    word_embeddings2=np.zeros((len(onsets),np.shape(w[0])[0]))

    for k in range(len(onsets)):

        # print(k)

        a=(df3[df3.adjusted_onset==onsets[k]].embeddings.values)
        
        q=a[0]
        for j in range(1,len(a)):

            q=q+np.asarray(a[j])

        word_embeddings2[k,:]=np.asarray(q)/len(a)

        # break
        
        del a
        del q  
        pca = PCA(n_components=50)
        word_embeddings2=word_embeddings2- np.mean(word_embeddings2, axis=0)
        word_embeddings=pca.fit_transform(word_embeddings2)
        word_embeddings = normalize(word_embeddings, axis=1, norm='l2')

    return onsets, word_embeddings

def get_speech_embeddings():

    path_ken1="/scratch/gpfs/kw1166/247/247-pickling/results/podcast/777/pickles/embeddings/"
    model_size='symbolic-speech-new'
    path_ken=path_ken1+model_size+'/full/'

    path_emb=path_ken+'cnxt_0001/'
    os.chdir(path_emb)

    emb_file='layer_00.pkl'

    pickle_file = open(emb_file, "rb")
    objects = []

    i=0

    while True:
        print('i',i)
        try:

            objects.append(pickle.load(pickle_file))

        except EOFError:

            break

    pickle_file.close()

    a=objects[0]

    df=pd.DataFrame(a)
    df=df.loc[~df.adjusted_onset.isnull()]# & (df.token_idx==0)] # and df.token_idx==1

    onsets=np.unique(df.adjusted_onset.values)

    word_embeddings2=np.zeros((len(onsets),60))

    for k in range(len(onsets)):

        # print(k)

        a=(df[df.adjusted_onset==onsets[k]].embeddings.values)
        q=a[0]
        for j in range(1,len(a)):

            q=q+np.asarray(a[j])

        word_embeddings2[k,:]=np.asarray(q)/len(a)

        # break
        
        del a
        del q  

    pca = PCA(n_components=50)
    word_embeddings2=word_embeddings2- np.mean(word_embeddings2, axis=0)
    word_embeddings=pca.fit_transform(word_embeddings2)
    word_embeddings = normalize(word_embeddings, axis=1, norm='l2')

    return onsets, word_embeddings