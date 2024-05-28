# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 16:52:23 2023

@author: arnab
"""

import os
import pickle
import numpy as np
import pandas as pd  
import csv
import random
from scipy.io import savemat
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from functools import reduce 
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.linear_model import Ridge,Lasso
# from sklearn.ensemble import RandomForestRegressor

from nltk.corpus import stopwords
stops = stopwords.words('english')

from brainiak.isc import isc
from brainiak.fcma.util import compute_correlation
import brainiak.funcalign.srm

df_all_sub=pd.read_csv('all_subject_sig.csv')
subjects=np.unique(df_all_sub['subject'])

#subjects=[661,717,723,798]
# lags=[1000] #[-400,-350,-300,-250,-200-150,-100,-50,0,50,100,150,200,250,300,350,400]
lags=list(range(-2000,2000,25))
#lags=[-4000]
#lags=[-2000,100]
srm_k=[5] #K value for SRM
#np.mean(corr_with_subjects[:,:,0])
shift_ms=300 # this is a dummy parameter, not used
cv=10
#elec_number=31
filename_mat='elec_space_generalization_across_subject.mat'

# reconstruction_area='IFG'
# 'srm_result_all_sub_without_augmentation_cv_10.mat'

# embeddings_model='bert'  #'gpt2' #different_layer  #different_size
embeddings_model='gpt2'
layer_id=3
model_size='symbolic-lang' #'gpt2-medium' 'gpt2-small' 'gpt2-large' 'gpt-neox-20b' 'gpt-neo-125M' 'gpt-neo-2.7B' 'gpt-neo-1.3B' 'glove50' symbolic-lang

compute_srm_ridge=False 

compute_regression_cv=False
compute_srm=False
compute_srm_cv=False
compute_encoding_with_augmentation=False
compute_pca_regression=False
compute_pca_across_subject=False
compute_encoding_with_augmentation_per_elec=False
denoising=False
compute_regression_per_elec=False
content_function_split=False 
content=False 
function=False
compute_srm_regression_all_elec=False 
compute_original_regression_all_elec=False  
compute_srm_leave_one_out=True
compute_shared_split_regression=False
compute_shuffling_electrode=False
compute_srm_generalization=False

srm_reconstruction_sub=False 
srm_reconstruction_area=False 

compute_pca_generalization_across_subject=False 

def data_create(shift_ms):

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
    pred_prob=np.zeros((l))
    selected_word=[]
    
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
          pred_prob[p-4]=a[i]['true_pred_prob']  
          s=0
          for j in range(179):
              if a[i]['datum_word']==stops[j]:
                  s=1
                  
          is_stop[p-4]=s  
                  
          
          
          
        p=p+1
    
    pca = PCA(n_components=50)
    word_embeddings2=word_embeddings2- np.mean(word_embeddings2, axis=0)
    word_embeddings=pca.fit_transform(word_embeddings2)
    word_embeddings = normalize(word_embeddings, axis=1, norm='l2')
    
   
    return onset,word_embeddings, is_stop
if embeddings_model=='gpt2':

    [onsets,word_embeddings,is_stop]= data_create(shift_ms)

elif embeddings_model=='bert':

    path='/scratch/gpfs/arnab/Encoding/bert'
    os.chdir(path)
    onsets=np.squeeze(loadmat('bert_embeddings.mat')['onset'])
    word_embeddings=loadmat('bert_embeddings.mat')['embeddings']


    path='/scratch/gpfs/arnab/Encoding/'
    os.chdir(path)

elif embeddings_model=='different_layer':

    path='/scratch/gpfs/arnab/Encoding/all_layer/'
    os.chdir(path)

    onsets=np.squeeze(loadmat('All_layer_embeddings_gpt.mat')['onset'])
    word_embeddings2=loadmat('All_layer_embeddings_gpt.mat')['embeddings'][layer_id,:,:]

    pca = PCA(n_components=50)
    word_embeddings2=word_embeddings2- np.mean(word_embeddings2, axis=0)
    word_embeddings=pca.fit_transform(word_embeddings2)
    word_embeddings = normalize(word_embeddings, axis=1, norm='l2')

    filename_mat='srm_layer_'+str(layer_id)+'.mat'

    print(filename_mat)

    path='/scratch/gpfs/arnab/Encoding/'
    os.chdir(path)

elif embeddings_model=='different_size':

    path='/scratch/gpfs/arnab/Encoding/different_size/'
    os.chdir(path)

    emb_file=model_size+'.mat'

    onsets=np.squeeze(loadmat(emb_file)['onset'])
    word_embeddings2=loadmat(emb_file)['embeddings']


    pca = PCA(n_components=50)
    word_embeddings2=word_embeddings2- np.mean(word_embeddings2, axis=0)
    word_embeddings=pca.fit_transform(word_embeddings2)
    word_embeddings = normalize(word_embeddings, axis=1, norm='l2')

    filename_mat='original_'+emb_file

    print(filename_mat)

    path='/scratch/gpfs/arnab/Encoding/'
    os.chdir(path)




if content_function_split:

    if content:
        index=np.where(is_stop==0)
        onsets=onsets[index]
        word_embeddings=np.squeeze(word_embeddings[index,:])
    if function:
        index=np.where(is_stop==1)
        onsets=onsets[index]
        word_embeddings=np.squeeze(word_embeddings[index,:])



electrodes1=[]
elec_num=[]
for qq in range(len(subjects)):
    
    subject=subjects[qq]
    #print('subject:',subject)
    s=np.array(df_all_sub.loc[df_all_sub.subject==subject].matfile)
    electrodes1.append(s)
    elec_num.append(len(s))



Y_data= np.zeros((len(subjects),len(lags),len(onsets),max(elec_num))) #45 is the electrode number here
# Y_742= np.zeros((len(lags),len(onsets),43))
# Y_798= np.zeros((len(lags),len(onsets),43))


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
            # if subject==742:
            #     Y_742[ii,:,k] = np.mean(Y1, axis=-1)
            # if subject==798:
            #     Y_798[ii,:,k] = np.mean(Y1, axis=-1)
        

#breakpoint()


## with cv

def computing_regression_cv(data1, word_embeddings,elec_num,cv):
    
    
    
    lag=np.shape(data1)[0]
    
    final_corr=np.zeros((cv,lag))
    
    kf = KFold(n_splits=cv)
    p=0
    
    for train_index, test_index in kf.split(word_embeddings):
        
        corr_lin2=[]
        
        X_train,X_test = word_embeddings[train_index,:],word_embeddings[test_index,:]
    
        # X_train= word_embeddings[0:4000,:]  #working on the last fold
        # X_test= word_embeddings[4000:5013,:]
    
        X_train -= np.mean(X_train, axis=0)
        X_test -= np.mean(X_train, axis=0)
        
        for i in range(lag):
            
            corr_lin1=[]
            
            data=data1[i,:,:]
            data=data[:,:elec_num]
        
            for k in range(np.shape(data)[1]):
                
                label=data[:,k]  
                Y_train,Y_test = label[train_index],label[test_index]
                
                # Y_train=data[0:4000,k]  
                # Y_test=data[4000:5013,k]
                
                Y_train -= np.mean(Y_train, axis=0)
                Y_test -= np.mean(Y_train, axis=0)
              
                
                
                #We fit the Linear regression to our train set
                clf_linear=LinearRegression()
                clf_linear.fit(X_train,Y_train)
                
                
                prediction_linear=clf_linear.predict(X_test)
                
                corr_lin1.append(np.corrcoef(Y_test,prediction_linear)[0,1])
                
                del clf_linear
                
            corr_lin2.append(np.mean(corr_lin1))
            
        final_corr[p,:]=np.asarray(corr_lin2)
        p=p+1
            
    return final_corr

corr_original=[]



if compute_regression_cv:
    print('Computing original Regression CV')

    for qq in range(len(subjects)):
        
        corr_original.append(computing_regression_cv(Y_data[qq,:,:,:], word_embeddings, elec_num[qq],cv))
        
        
#breakpoint()        

#     corr_717_cv=computing_regression_cv(Y_717, word_embeddings)
#     corr_798_cv=computing_regression_cv(Y_798, word_embeddings)
#     corr_742_cv=computing_regression_cv(Y_742, word_embeddings)

# else:
#     corr_717_cv=0
#     corr_798_cv=0
#     corr_742_cv=0
    

## srm

srm_corr=np.zeros((len(srm_k),len(lags)))


def srm_regression(Y_data,word,elec_num,features):
    
    lag=np.shape(Y_data)[1]
    corr_srm=[]
    
    n_iter = 1000
    for i in range(lag):
        
        train_data=[]
        test_data=[]
        
        for qq in range(len(subjects)):
            
            data=Y_data[qq,i,:,:]
            data=data[:,:elec_num[qq]]
            
            
            train1=data[0:4000,:].T
            test1=data[4000:5013,:].T
            
            train_data.append(train1)
            test_data.append(test1)
                
                    
                
    
        # data1=data11[i,:,:]
        # data2=data22[i,:,:]
        # data3=data33[i,:,:]
        
        # train_717=data1[0:4000,:].T
        # test_717=data1[4000:5013,:].T
        
        # train_742=data2[0:4000,:].T
        # test_742=data2[4000:5013,:].T
        
        # train_798=data3[0:4000,:].T
        # test_798=data3[4000:5013,:].T
        
        # train_data=[]
        # train_data.append(train_717)
        # train_data.append(train_742)
        # train_data.append(train_798)
        
        # test_data=[]
        # test_data.append(test_717)
        # test_data.append(test_742)
        # test_data.append(test_798)
        
          # How many iterations of fitting will you perform
    
        # Create the SRM object
        srm = brainiak.funcalign.srm.SRM(n_iter=n_iter, features=features)
        
        srm.fit(train_data)
        
        shared_test = srm.transform(test_data)
        shared_train = srm.transform(train_data)
        
        s_avg_train=(shared_train[0]+shared_train[1]+shared_train[2])/3
        s_avg_test=(shared_test[0]+shared_test[1]+shared_test[2])/3
        
        X_train= word[0:4000,:]
        X_test= word[4000:5013,:]
        
        X_train -= np.mean(X_train, axis=0)
        X_test -= np.mean(X_train, axis=0)
        
        srm_train=s_avg_train.T
        srm_test=s_avg_test.T
        
        c=[]       
        for k in range(features):
            
            
            Y_train=srm_train[:,k]
            Y_test=srm_test[:,k]
            
            Y_train -= np.mean(Y_train, axis=0)
            Y_test -= np.mean(Y_train, axis=0)
            
            clf_linear=LinearRegression()
            clf_linear.fit(X_train,Y_train)
            prediction_linear=clf_linear.predict(X_test)
            
            c.append(np.corrcoef(Y_test,prediction_linear)[0,1])
            del clf_linear
        # print(np.mean(c))    
        corr_srm.append(np.mean(c))
        
    return corr_srm
        



if compute_srm:    
    print('Computing SRM')
    for k in range(len(srm_k)):
    
        srm_corr[k,:]= srm_regression(Y_data,word_embeddings,elec_num, srm_k[k])
    
#print(srm_corr)    
        
def encoding_with_augmentation(Y_data, word,elec_num, train_index,test_index):
    
    X_train= word[train_index,:]
    X_test= word[test_index,:]
    X1=np.tile(X_test,(len(subjects)-1,1))
    #X2=np.tile(X_train,(len(subjects)-1,1))
    #X1=X_test
    #X=np.concatenate((X_train, X2, ), axis=0)
    #X=np.concatenate((X_train, X1, ), axis=0)
    X=X_train
    X -= np.mean(X, axis=0)
    X_test2 =X_test- np.mean(X, axis=0)
    
    lag=np.shape(Y_data)[1]
    n_iter = 100
    
    # corr_with_742=[]
    # corr_with_798=[]
    final_corr=[]
    for i in range(lag):
        
        train_data=[]
        test_data=[]
        
        for qq in range(len(subjects)):
            
            data=Y_data[qq,i,:,:]
            data=data[:,:elec_num[qq]]
            
            
            train1=data[train_index,:].T
            test1=data[test_index,:].T
            
            train_data.append(train1)
            test_data.append(test1)
    
       
        
        srm = brainiak.funcalign.srm.SRM(n_iter=n_iter, features=5)
        
        srm.fit(train_data)
        
        shared_test = srm.transform(test_data)
        shared_train = srm.transform(train_data)
        
                
        corr_with_subjects=[]
        
        for qq in range(len(subjects)):

            #print('subject:',subjects[qq])
            
            w0=srm.w_[qq]
            
            subject_elec=elec_num[qq]
            
            self_train=w0.dot(shared_train[qq])
            Y_train=self_train.T
            self_test=w0.dot(shared_test[qq])
            Y_test=self_test.T

            # Y_train=(train_data[qq]).T
            # Y_test=(test_data[qq]).T

            ##breakpoint()

            aa=np.arange(len(subjects))
            aa=np.delete(aa,qq)
            
            Y_aug_train=[]
            Y_aug_test=[]
            
            for index in aa:

            #index=aa[5]
        
                #print(subjects[index])    
                signal_test=w0.dot(shared_test[index])
                signal_test=signal_test.T

                signal_train=w0.dot(shared_train[index])
                signal_train=signal_train.T

                Y_aug_train.append(signal_train)             
                Y_aug_test.append(signal_test)
                del signal_test
                del signal_train
                
            

            Y_aug_train=np.array(Y_aug_train)
            Y_aug_train=np.reshape(Y_aug_train,(np.shape(Y_aug_train)[0]*np.shape(Y_aug_train)[1],np.shape(Y_aug_train)[2]))
           
            Y_aug_test=np.array(Y_aug_test)
            Y_aug_test=np.reshape(Y_aug_test,(np.shape(Y_aug_test)[0]*np.shape(Y_aug_test)[1],np.shape(Y_aug_test)[2]))
            
            
            corr1_with_717=[]
            for k in range(subject_elec):
                
                
                Y=Y_train[:,k]

                #Y=np.concatenate((Y_train[:,k], Y_aug_train[:,k],), axis=0)
                #Y=np.concatenate((Y_train[:,k], Y_aug_test[:,k],), axis=0)
                
                Y -= np.mean(Y, axis=0)
                Y_test2 =Y_test[:,k]- np.mean(Y, axis=0)
                
                                
                clf_linear2=LinearRegression()
                clf_linear2.fit(X,Y)
                prediction_linear2=clf_linear2.predict(X_test2)
                corr1_with_717.append(np.corrcoef(Y_test2,prediction_linear2)[0,1])
                del clf_linear2
                
                
            corr_with_subjects.append(np.mean(corr1_with_717))

            #breakpoint()

            del Y_aug_test
            del Y_aug_train
            del Y_train
            del Y_test
            
        ww=np.array(corr_with_subjects)
        
        final_corr.append(ww)        
    return final_corr
            

   

kf = KFold(n_splits=cv)
p=0
corr_with_subjects=np.zeros((cv,len(lags),len(subjects)))

if compute_encoding_with_augmentation:
    print('Computing Augmentation (denoising/augmentation)') 

    for train_index, test_index in kf.split(word_embeddings):

        print('Fold:',p)
        
        corr_with_subjects[p,:,:]=encoding_with_augmentation(Y_data,word_embeddings,elec_num, train_index, test_index)
    
        p=p+1


#srm with cv
#breakpoint()

def srm_regression_cv(Y_data,word,elec_num,features,cv):
    
    
    lag=np.shape(Y_data)[1]
    n_iter = 100
    final_corr=np.zeros((cv,lag))
    kf = KFold(n_splits=cv)
    p=0
    
    for train_index, test_index in kf.split(word):
        corr_srm=[]
    
        for i in range(lag):
            
            train_data=[]
            test_data=[]
            
            for qq in range(len(subjects)):
                
                data=Y_data[qq,i,:,:]
                data=data[:,:elec_num[qq]]
                
                
                train1=data[train_index,:].T
                test1=data[test_index,:].T
                
                train_data.append(train1)
                test_data.append(test1)
            
            
        
            # data1=data11[i,:,:]
            # data2=data22[i,:,:]
            # data3=data33[i,:,:]
            
            # train_717=data1[train_index,:].T
            # test_717=data1[test_index,:].T
            
            # train_742=data2[train_index,:].T
            # test_742=data2[test_index,:].T
            
            # train_798=data3[train_index,:].T
            # test_798=data3[test_index,:].T
            
            # train_data=[]
            # train_data.append(train_717)
            # train_data.append(train_742)
            # train_data.append(train_798)
            
            # test_data=[]
            # test_data.append(test_717)
            # test_data.append(test_742)
            # test_data.append(test_798)
            
              # How many iterations of fitting will you perform
        
            # Create the SRM object
            srm = brainiak.funcalign.srm.SRM(n_iter=n_iter, features=features)
            
            srm.fit(train_data)
            
            shared_test = srm.transform(test_data)
            shared_train = srm.transform(train_data)
            
            s_avg_train=shared_train[0]
            s_avg_test=shared_test[0]
            for ii in range(1,len(subjects)):
                s_avg_train=s_avg_train+shared_train[ii]
                s_avg_test=s_avg_test+shared_test[ii]

            s_avg_train=s_avg_train/len(subjects)
            s_avg_test=s_avg_test/len(subjects)

            X_train= word[train_index,:]
            X_test= word[test_index,:]
            
            X_train -= np.mean(X_train, axis=0)
            X_test -= np.mean(X_train, axis=0)
            
            srm_train=s_avg_train.T
            srm_test=s_avg_test.T
            
            c=[]       
            for k in range(features):
                
                
                Y_train=srm_train[:,k]
                Y_test=srm_test[:,k]
                
                Y_train -= np.mean(Y_train, axis=0)
                Y_test -= np.mean(Y_train, axis=0)
                
                clf_linear=LinearRegression()
                clf_linear.fit(X_train,Y_train)
                prediction_linear=clf_linear.predict(X_test)
                
                c.append(np.corrcoef(Y_test,prediction_linear)[0,1])
                del clf_linear
            # print(np.mean(c))    
            corr_srm.append(np.mean(c))
            
        final_corr[p,:]=corr_srm
        p=p+1
            
    return final_corr

srm_corr_cv=np.zeros((len(srm_k),cv,len(lags)))

if compute_srm_cv:

    print('Compute shared space regression')

    for k in range(len(srm_k)):
    
        srm_corr_cv[k,:,:]= srm_regression_cv(Y_data,word_embeddings,elec_num,srm_k[k],cv)
    

#breakpoint()

def pca_data_prep(data1,train_index,test_index,value):
    
    a1=data1[train_index,:]
    # a2=data2[train_index,:]
    # a3=data3[train_index,:]
    
    b1=data1[test_index,:]
    # b2=data2[test_index,:]
    # b3=data3[test_index,:]
    
    a1=a1-(np.mean(a1,axis=0))
    # a2=a2-(np.mean(a2,axis=0))
    # a3=a3-(np.mean(a3,axis=0))
    
    b1=b1-(np.mean(a1,axis=0))
    # b2=b2-(np.mean(a2,axis=0))
    # b3=b3-(np.mean(a3,axis=0))
    
    pca = PCA(n_components=value)

    train_717_pca = pca.fit_transform(a1)
    test_717_pca = pca.transform(b1)
    # train_742_pca = pca.fit_transform(a2)
    # test_742_pca = pca.transform(b2)
    # train_798_pca = pca.fit_transform(a3)
    # test_798_pca = pca.transform(b3)
    
    
    # train_pca=(train_717_pca+train_742_pca+train_798_pca)/3
    # test_pca=(test_717_pca+test_742_pca+test_798_pca)/3
    
    return train_717_pca,test_717_pca



    
def pca_regression_cv(Y_data, word_embeddings,cv=10,value=5):
    
    lag=np.shape(Y_data)[1]
    
    final_corr=np.zeros((cv,lag))
    
    kf = KFold(n_splits=cv)
    p=0
    
    for train_index, test_index in kf.split(word_embeddings):
        
        corr_lin2=[]
        
        X_train,X_test = word_embeddings[train_index,:],word_embeddings[test_index,:]
    
        # X_train= word_embeddings[0:4000,:]  #working on the last fold
        # X_test= word_embeddings[4000:5013,:]
    
        X_train -= np.mean(X_train, axis=0)
        X_test -= np.mean(X_train, axis=0)
        
        for i in range(lag):
            
            corr_lin1=[]
            a1,b1=pca_data_prep(Y_data[0,i,:,:elec_num[0]],train_index,test_index,value)
            for qq in range(1,len(subjects)):
                a11,b11=pca_data_prep(Y_data[qq,i,:,:elec_num[qq]],train_index,test_index,value)
                a1=a1+a11
                b1=b1+b11
            data_train=a1/len(subjects)
            data_test=b1/len(subjects)

            #data=data1[i,:,:]
            # [data_train,data_test]=pca_data_prep(data1[i,:,:],data2[i,:,:],data3[i,:,:],train_index,test_index,value)
        
            for k in range(np.shape(data_train)[1]):
                
                Y_train=data_train[:,k]  
                Y_test=data_test[:,k]  
                #Y_train,Y_test = label[train_index],label[test_index]
                
                # Y_train=data[0:4000,k]  
                # Y_test=data[4000:5013,k]
                
                Y_train -= np.mean(Y_train, axis=0)
                Y_test -= np.mean(Y_train, axis=0)
              
                
                
                #We fit the Linear regression to our train set
                clf_linear=LinearRegression()
                clf_linear.fit(X_train,Y_train)
                
                
                prediction_linear=clf_linear.predict(X_test)
                
                corr_lin1.append(np.corrcoef(Y_test,prediction_linear)[0,1])
                
                del clf_linear
                
            corr_lin2.append(np.mean(corr_lin1))
            
        final_corr[p,:]=np.asarray(corr_lin2)
        p=p+1
            
    return final_corr

if compute_pca_regression:
    print('PCA regression')

    pca_cv=pca_regression_cv(Y_data, word_embeddings,cv=10,value=5)
    
else:
    pca_cv=0


# pca across subject

def pca_across_subject(Y_data, word_embeddings,cv=10,value=5):
    
    lag=np.shape(Y_data)[1]
    
    final_corr=np.zeros((cv,lag))
    
    kf = KFold(n_splits=cv)
    p=0
    
    for train_index, test_index in kf.split(word_embeddings):
        
        corr_lin2=[]
        
        X_train,X_test = word_embeddings[train_index,:],word_embeddings[test_index,:]
    
        # X_train= word_embeddings[0:4000,:]  #working on the last fold
        # X_test= word_embeddings[4000:5013,:]
    
        X_train -= np.mean(X_train, axis=0)
        X_test -= np.mean(X_train, axis=0)
        
        for i in range(lag):
            
            corr_lin1=[]
            # a1,b1=pca_data_prep(Y_data[0,i,:,:elec_num[0]],train_index,test_index,value)
            a1=Y_data[0,i,:,:elec_num[0]]
            data_train1=a1[train_index,:]
            data_test1=a1[test_index,:]


            for qq in range(1,len(subjects)):
                del a1
                # a11,b11=pca_data_prep(Y_data[qq,i,:,:elec_num[qq]],train_index,test_index,value)
                a1=Y_data[qq,i,:,:elec_num[qq]]
                temp_train=a1[train_index,:]
                temp_test=a1[test_index,:]
                
                data_train1=np.concatenate((data_train1,temp_train),axis=1)
                data_test1=np.concatenate((data_test1,temp_test),axis=1)

            data_train1=data_train1-(np.mean(data_train1,axis=0))
            data_test1=data_test1-(np.mean(data_train1,axis=0))

            pca = PCA(n_components=value)

            data_train = pca.fit_transform(data_train1)
            data_test = pca.transform(data_test1)

            #data=data1[i,:,:]
            # [data_train,data_test]=pca_data_prep(data1[i,:,:],data2[i,:,:],data3[i,:,:],train_index,test_index,value)
        
            for k in range(np.shape(data_train)[1]):
                
                Y_train=data_train[:,k]  
                Y_test=data_test[:,k]  
                #Y_train,Y_test = label[train_index],label[test_index]
                
                # Y_train=data[0:4000,k]  
                # Y_test=data[4000:5013,k]
                
                Y_train -= np.mean(Y_train, axis=0)
                Y_test -= np.mean(Y_train, axis=0)
              
                
                
                #We fit the Linear regression to our train set
                clf_linear=LinearRegression()
                clf_linear.fit(X_train,Y_train)
                
                
                prediction_linear=clf_linear.predict(X_test)
                
                corr_lin1.append(np.corrcoef(Y_test,prediction_linear)[0,1])
                
                del clf_linear
                
            corr_lin2.append(np.mean(corr_lin1))
            
        final_corr[p,:]=np.asarray(corr_lin2)
        p=p+1
            
    return final_corr

if compute_pca_across_subject:
    print('PCA across subject')

    pca_across_sub=pca_across_subject(Y_data, word_embeddings,cv=10,value=5)
    
else:
    pca_across_sub=0

# breakpoint()

def computing_regression_per_elec(data1, word_embeddings,elec_num,cv=10):
    
    lag=np.shape(data1)[0]
    
    final_corr=np.zeros((cv,elec_num))
    
    kf = KFold(n_splits=cv)
    p=0
    
    for train_index, test_index in kf.split(word_embeddings):
        
        corr_lin2=[]
        
        X_train,X_test = word_embeddings[train_index,:],word_embeddings[test_index,:]
    
        # X_train= word_embeddings[0:4000,:]  #working on the last fold
        # X_test= word_embeddings[4000:5013,:]
    
        X_train -= np.mean(X_train, axis=0)
        X_test -= np.mean(X_train, axis=0)
        
        for i in range(lag):
            
            corr_lin1=[]
            
            data=data1[i,:,:]
            data=data[:,:elec_num]
        
            for k in range(np.shape(data)[1]):
                
                label=data[:,k]  
                Y_train,Y_test = label[train_index],label[test_index]
                
                # Y_train=data[0:4000,k]  
                # Y_test=data[4000:5013,k]
                
                Y_train -= np.mean(Y_train, axis=0)
                Y_test -= np.mean(Y_train, axis=0)
              
                
                
                #We fit the Linear regression to our train set
                clf_linear=LinearRegression()
                clf_linear.fit(X_train,Y_train)
                
                
                prediction_linear=clf_linear.predict(X_test)
                
                corr_lin1.append(np.corrcoef(Y_test,prediction_linear)[0,1])
                
                del clf_linear
                
            corr_lin2.append(np.asarray(corr_lin1))
        
            
        corr_lin2=np.asarray(corr_lin2)
        corr_lin2=np.max(corr_lin2,axis=0)
        
        final_corr[p,:]=np.asarray(corr_lin2)
        p=p+1
    a=np.max(final_corr,axis=0)       
    return a

if compute_regression_per_elec:

    print('computing regression per elec')
    corr_original_per_elec=[]

    for qq in range(len(subjects)):
        corr_original_per_elec.append(computing_regression_per_elec(Y_data[qq,:,:,:], word_embeddings, elec_num[qq],cv))

else:
    corr_original_per_elec=[]


def encoding_with_augmentation_per_elec(Y_data, word,elec_num, train_index,test_index):
    
    X_train= word[train_index,:]
    X_test= word[test_index,:]
    X1=np.tile(X_test,(len(subjects)-1,1))
    #X2=np.tile(X_train,(len(subjects)-1,1))
    #X1=X_test
    #X=np.concatenate((X_train, X2, ), axis=0)
    #X=np.concatenate((X_train, X1, ), axis=0)
    if denoising:
        X=X_train
    else:
        X=np.concatenate((X_train, X1, ), axis=0)

    X -= np.mean(X, axis=0)
    X_test2 =X_test- np.mean(X, axis=0)
    
    lag=np.shape(Y_data)[1]
    n_iter = 100
    data_collect=np.zeros((len(subjects),lag,max(elec_num)))
    # corr_with_742=[]
    # corr_with_798=[]
    final_corr=[]
    for i in range(lag):
        
        train_data=[]
        test_data=[]
        
        for qq in range(len(subjects)):
            
            data=Y_data[qq,i,:,:]
            data=data[:,:elec_num[qq]]
            
            
            train1=data[train_index,:].T
            test1=data[test_index,:].T
            
            train_data.append(train1)
            test_data.append(test1)
    
       
        
        srm = brainiak.funcalign.srm.SRM(n_iter=n_iter, features=5)
        
        srm.fit(train_data)
        
        shared_test = srm.transform(test_data)
        shared_train = srm.transform(train_data)
        
                
        corr_with_subjects=[]
        
        for qq in range(len(subjects)):

            #print('subject:',subjects[qq])
            
            w0=srm.w_[qq]
            
            subject_elec=elec_num[qq]
            
            self_train=w0.dot(shared_train[qq])
            Y_train=self_train.T
            self_test=w0.dot(shared_test[qq])
            Y_test=self_test.T

            # Y_train=(train_data[qq]).T
            # Y_test=(test_data[qq]).T

            ##breakpoint()

            aa=np.arange(len(subjects))
            aa=np.delete(aa,qq)
            
            Y_aug_train=[]
            Y_aug_test=[]
            
            for index in aa:

            #index=aa[5]
        
                #print(subjects[index])    
                signal_test=w0.dot(shared_test[index])
                signal_test=signal_test.T

                signal_train=w0.dot(shared_train[index])
                signal_train=signal_train.T

                Y_aug_train.append(signal_train)             
                Y_aug_test.append(signal_test)
                del signal_test
                del signal_train
                
            

            Y_aug_train=np.array(Y_aug_train)
            Y_aug_train=np.reshape(Y_aug_train,(np.shape(Y_aug_train)[0]*np.shape(Y_aug_train)[1],np.shape(Y_aug_train)[2]))
           
            Y_aug_test=np.array(Y_aug_test)
            Y_aug_test=np.reshape(Y_aug_test,(np.shape(Y_aug_test)[0]*np.shape(Y_aug_test)[1],np.shape(Y_aug_test)[2]))
            
            
            corr1_with_717=[]
            for k in range(subject_elec):
                
                if denoising:
                    Y=Y_train[:,k]
                else:
                    Y=np.concatenate((Y_train[:,k], Y_aug_test[:,k],), axis=0)


                #Y=np.concatenate((Y_train[:,k], Y_aug_train[:,k],), axis=0)
                #Y=np.concatenate((Y_train[:,k], Y_aug_test[:,k],), axis=0)
                
                Y -= np.mean(Y, axis=0)
                Y_test2 =Y_test[:,k]- np.mean(Y, axis=0)
                
                                
                clf_linear2=LinearRegression()
                clf_linear2.fit(X,Y)
                prediction_linear2=clf_linear2.predict(X_test2)
                corr1_with_717.append(np.corrcoef(Y_test2,prediction_linear2)[0,1])
                del clf_linear2
                
                
            corr_with_subjects.append(np.mean(corr1_with_717))

            data_collect[qq,i,:subject_elec]=np.asarray(corr1_with_717)

            #breakpoint()

            del Y_aug_test
            del Y_aug_train
            del Y_train
            del Y_test
            
        # ww=np.array(corr_with_subjects)
        
        # final_corr.append(ww)        

    c=np.zeros((len(subjects),max(elec_num)))
    for qq in range(len(subjects)):

        d=data_collect[qq,:,:]
        c[qq,:]=(np.max(np.asarray(d),axis=0))


    return c

corr_with_subjects_per_elec=[]
c=np.zeros((cv,len(subjects),max(elec_num)))

if compute_encoding_with_augmentation_per_elec:

    print('compute encoding with augmentation per elec')

    kf = KFold(n_splits=cv)
    p=0
    for train_index, test_index in kf.split(word_embeddings):

        print('Fold:',p)
        
        c[p,:,:]=encoding_with_augmentation_per_elec(Y_data,word_embeddings,elec_num, train_index, test_index)
    
        p=p+1

    
    for qq in range(len(subjects)):
        d=c[:,qq,:elec_num[qq]]
        corr_with_subjects_per_elec.append(np.max(np.asarray(d),axis=0))
        del d


# getting result of all elec

def original_regression_all_elec(data1, word_embeddings,elec_num,cv):
    
    lag=np.shape(data1)[0]
    
    final_corr=np.zeros((cv,elec_num,lag))

    final_corr1=np.zeros((elec_num,lag))
    
    kf = KFold(n_splits=cv)
    p=0
    
    for train_index, test_index in kf.split(word_embeddings):
        
        corr_lin2=[]
        
        X_train,X_test = word_embeddings[train_index,:],word_embeddings[test_index,:]
    
        # X_train= word_embeddings[0:4000,:]  #working on the last fold
        # X_test= word_embeddings[4000:5013,:]
    
        X_train -= np.mean(X_train, axis=0)
        X_test -= np.mean(X_train, axis=0)
        
        for i in range(lag):
            
            corr_lin1=[]
            
            data=data1[i,:,:]
            data=data[:,:elec_num]
        
            for k in range(np.shape(data)[1]):
                
                label=data[:,k]  
                Y_train,Y_test = label[train_index],label[test_index]
                
                # Y_train=data[0:4000,k]  
                # Y_test=data[4000:5013,k]
                
                Y_train -= np.mean(Y_train, axis=0)
                Y_test -= np.mean(Y_train, axis=0)
              
                
                
                #We fit the Linear regression to our train set
                clf_linear=LinearRegression()
                clf_linear.fit(X_train,Y_train)
                
                
                prediction_linear=clf_linear.predict(X_test)
                
                corr_lin1.append(np.corrcoef(Y_test,prediction_linear)[0,1])
                
                del clf_linear
                
            corr_lin2.append(corr_lin1)
            
        final_corr[p,:,:]=np.asarray(corr_lin2).T
        p=p+1

    for w in range(cv):
        final_corr1=final_corr[w,:,:]+final_corr1
    final_corr1=final_corr1/cv



            
    return final_corr1

corr_original_all_elec1=[]


if compute_original_regression_all_elec:

    print('Computing Regression All Elec')
    
    for qq in range(len(subjects)):
        
        corr_original_all_elec1.append(original_regression_all_elec(Y_data[qq,:,:,:], word_embeddings, elec_num[qq],cv))
        # a=corr_original.append(original_regression_all_elec(Y_data[qq,:,:,:], word_embeddings, elec_num[qq],cv))
        

    corr_original_all_elec=np.asarray(corr_original_all_elec1[0])
    for qq in range(1,len(subjects)):
        corr_original_all_elec=np.concatenate((corr_original_all_elec,np.asarray(corr_original_all_elec1[qq])), axis=0)

else:
    corr_original_all_elec=0


#srm all elec

def srm_regression_all_elec(Y_data, word,elec_num, train_index,test_index):
    
    X_train= word[train_index,:]
    X_test= word[test_index,:]
    X1=np.tile(X_test,(len(subjects)-1,1))
    #X2=np.tile(X_train,(len(subjects)-1,1))
    #X1=X_test
    #X=np.concatenate((X_train, X2, ), axis=0)
    #X=np.concatenate((X_train, X1, ), axis=0)
    X=X_train
    X -= np.mean(X, axis=0)
    X_test2 =X_test- np.mean(X, axis=0)
    
    lag=np.shape(Y_data)[1]
    n_iter = 100
    
    # corr_with_742=[]
    # corr_with_798=[]
    final_corr=[]
    for i in range(lag):
        
        train_data=[]
        test_data=[]
        
        for qq in range(len(subjects)):
            
            data=Y_data[qq,i,:,:]
            data=data[:,:elec_num[qq]]
            
            
            train1=data[train_index,:].T
            test1=data[test_index,:].T
            
            train_data.append(train1)
            test_data.append(test1)
    
       
        
        srm = brainiak.funcalign.srm.SRM(n_iter=n_iter, features=5)
        
        srm.fit(train_data)
        
        shared_test = srm.transform(test_data)
        shared_train = srm.transform(train_data)
        
                
        corr_with_subjects=[]
        
        for qq in range(len(subjects)):

            #print('subject:',subjects[qq])
            
            w0=srm.w_[qq]
            
            subject_elec=elec_num[qq]
            
            self_train=w0.dot(shared_train[qq])
            Y_train=self_train.T
            self_test=w0.dot(shared_test[qq])
            Y_test=self_test.T

            # Y_train=(train_data[qq]).T
            # Y_test=(test_data[qq]).T

            # ##breakpoint()

            # aa=np.arange(len(subjects))
            # aa=np.delete(aa,qq)
            
            # Y_aug_train=[]
            # Y_aug_test=[]
            
            # for index in aa:

            # #index=aa[5]
        
            #     #print(subjects[index])    
            #     signal_test=w0.dot(shared_test[index])
            #     signal_test=signal_test.T

            #     signal_train=w0.dot(shared_train[index])
            #     signal_train=signal_train.T

            #     Y_aug_train.append(signal_train)             
            #     Y_aug_test.append(signal_test)
            #     del signal_test
            #     del signal_train
                
            

            # Y_aug_train=np.array(Y_aug_train)
            # Y_aug_train=np.reshape(Y_aug_train,(np.shape(Y_aug_train)[0]*np.shape(Y_aug_train)[1],np.shape(Y_aug_train)[2]))
           
            # Y_aug_test=np.array(Y_aug_test)
            # Y_aug_test=np.reshape(Y_aug_test,(np.shape(Y_aug_test)[0]*np.shape(Y_aug_test)[1],np.shape(Y_aug_test)[2]))
            
            
            corr1_with_717=[]
            for k in range(subject_elec):
                
                
                Y=Y_train[:,k]

                #Y=np.concatenate((Y_train[:,k], Y_aug_train[:,k],), axis=0)
                #Y=np.concatenate((Y_train[:,k], Y_aug_test[:,k],), axis=0)
                
                Y -= np.mean(Y, axis=0)
                Y_test2 =Y_test[:,k]- np.mean(Y, axis=0)
                
                                
                clf_linear2=LinearRegression()
                clf_linear2.fit(X,Y)
                prediction_linear2=clf_linear2.predict(X_test2)
                corr1_with_717.append(np.corrcoef(Y_test2,prediction_linear2)[0,1])
                corr_with_subjects.append(np.corrcoef(Y_test2,prediction_linear2)[0,1])
                del clf_linear2
                
                
            # corr_with_subjects.append(corr1_with_717)

            #breakpoint()

            # del Y_aug_test
            # del Y_aug_train
            del Y_train
            del Y_test
            
        ww=np.array(corr_with_subjects)
        
        final_corr.append(ww)        
    return final_corr
  

kf = KFold(n_splits=cv)
p=0
total_elec=sum([iterator for iterator  in elec_num])
corr_with_subjects1=np.zeros((cv,len(lags),total_elec))

if compute_srm_regression_all_elec:

    print('Computing SRM All Elec')  

    for train_index, test_index in kf.split(word_embeddings):

        print('Fold:',p)
        
        corr_with_subjects1[p,:,:]=srm_regression_all_elec(Y_data,word_embeddings,elec_num, train_index, test_index)
    
        p=p+1

    corr_with_subjects_all_elec=corr_with_subjects1[0]
    for ww in range(1,cv):
        corr_with_subjects_all_elec=corr_with_subjects_all_elec+corr_with_subjects1[ww,:,:]

    corr_with_subjects_all_elec=corr_with_subjects_all_elec/cv

else:
    corr_with_subjects_all_elec=0


def srm_leave_one_out(Y_data, word,elec_num, train_index,test_index):

    X_train= word[train_index,:]
    X_test= word[test_index,:]
    # X1=np.tile(X_test,(len(subjects)-1,1))
    #X2=np.tile(X_train,(len(subjects)-1,1))
    #X1=X_test
    #X=np.concatenate((X_train, X2, ), axis=0)
    #X=np.concatenate((X_train, X1, ), axis=0)
    X=X_train
    X -= np.mean(X, axis=0)
    X_test2 =X_test- np.mean(X, axis=0)
    
    lag=np.shape(Y_data)[1]
    n_iter = 100
    
    # corr_with_742=[]
    # corr_with_798=[]
    final_corr=[]
    for i in range(lag):
        
        corr_with_subjects=[]
        
        for qq1 in range(len(subjects)):

            sub_now=subjects[qq1]
            train_data=[]
            test_data=[]
            subject_elec=elec_num[qq1]

            for qq in range(len(subjects)):
    
                if subjects[qq]!=sub_now:

                    data=Y_data[qq,i,:,:]
                    data=data[:,:elec_num[qq]]


                    train1=data[train_index,:].T
                    test1=data[test_index,:].T

                    train_data.append(train1)
                    test_data.append(test1)
                    
                else:
                    data=Y_data[qq,i,:,:]
                    data=data[:,:elec_num[qq]]


                    train1=data[train_index,:].T
                    test1=data[test_index,:].T

                    self_train=train1
                    self_test=test1

            srm = brainiak.funcalign.srm.SRM(n_iter=n_iter, features=5)
            srm.fit(train_data)
            
            w_new=srm.transform_subject(self_train)
            #previous
            # q1=w_new.T.dot(self_train)
            # Y_train=(w_new.dot(q1)).T
            Y_train=(w_new.dot(srm.s_)).T
            
            q2=w_new.T.dot(self_test)
            Y_test=(w_new.dot(q2)).T
            # transform_test.append(srm1.w_[k].dot(q2))
            # transform_test.append(self_test)

            corr1_with_717=[]
            for k in range(subject_elec):
                
                
                Y=Y_train[:,k]

                #Y=np.concatenate((Y_train[:,k], Y_aug_train[:,k],), axis=0)
                #Y=np.concatenate((Y_train[:,k], Y_aug_test[:,k],), axis=0)
                
                Y -= np.mean(Y, axis=0)
                Y_test2 =Y_test[:,k]- np.mean(Y, axis=0)
                
                                
                clf_linear2=LinearRegression()
                clf_linear2.fit(X,Y)
                prediction_linear2=clf_linear2.predict(X_test2)
                corr1_with_717.append(np.corrcoef(Y_test2,prediction_linear2)[0,1])
                del clf_linear2
                
                
            corr_with_subjects.append(np.mean(corr1_with_717))

            #breakpoint()

            del Y_train
            del Y_test
            
        ww=np.array(corr_with_subjects)
            
        final_corr.append(ww)     
    

    return final_corr


kf = KFold(n_splits=cv)
p=0
corr_leave_one_out=np.zeros((cv,len(lags),len(subjects)))

if compute_srm_leave_one_out:
    print('Computing srm leave one out') 

    for train_index, test_index in kf.split(word_embeddings):

        print('Fold:',p)
        
        corr_leave_one_out[p,:,:]=srm_leave_one_out(Y_data,word_embeddings,elec_num, train_index, test_index)
    
        p=p+1


def shared_space_split(Y_data, word,elec_num, features,cv):
   
    lag=np.shape(Y_data)[1]
    n_iter = 100
    final_corr=np.zeros((cv,lag))
    kf = KFold(n_splits=cv)
    p=0
    
    for train_index, test_index in kf.split(word):
        corr_srm=[]

        for i in range(lag):

            
            train_data_first_half=[]
            test_data_first_half=[]
            train_data_second_half=[]
            test_data_second_half=[]
            
            for qq in range(int(len(subjects)/2)):

                data=Y_data[qq,i,:,:]
                data=data[:,:elec_num[qq]]

                train1=data[train_index,:].T
                test1=data[test_index,:].T

                train_data_first_half.append(train1)
                test_data_first_half.append(test1)

            for qq in range(int(len(subjects)/2),len(subjects)):

                data=Y_data[qq,i,:,:]
                data=data[:,:elec_num[qq]]

                train1=data[train_index,:].T
                test1=data[test_index,:].T

                train_data_second_half.append(train1)
                test_data_second_half.append(test1)
                    
                
            srm1 = brainiak.funcalign.srm.SRM(n_iter=n_iter, features=features)
            srm1.fit(train_data_first_half)
            srm2 = brainiak.funcalign.srm.SRM(n_iter=n_iter, features=features)
            srm2.fit(train_data_second_half)
            
            
            shared_test1 = srm1.transform(test_data_first_half)
            shared_train1 = srm1.transform(train_data_first_half)
            shared_test2 = srm2.transform(test_data_second_half)
            shared_train2 = srm2.transform(train_data_second_half)
            
            s_avg_train1=shared_train1[0]
            s_avg_test1=shared_test1[0]
            s_avg_train2=shared_train2[0]
            s_avg_test2=shared_test2[0]
            
            for ii in range(1,int(len(subjects)/2)):
                s_avg_train1=s_avg_train1+shared_train1[ii]
                s_avg_test1=s_avg_test1+shared_test1[ii]

                s_avg_train2=s_avg_train2+shared_train2[ii]
                s_avg_test2=s_avg_test2+shared_test2[ii]

            s_avg_train1=s_avg_train1/(int(len(subjects)/2))
            s_avg_test1=s_avg_test1/(int(len(subjects)/2))

            s_avg_train2=s_avg_train2/(int(len(subjects)/2))
            s_avg_test2=s_avg_test2/(int(len(subjects)/2))

            X_train= word[train_index,:]
            X_test= word[test_index,:]
            
            X_train -= np.mean(X_train, axis=0)
            X_test -= np.mean(X_train, axis=0)
            
            srm_train=s_avg_train1.T
            srm_test=s_avg_test2.T
            
            c=[]       
            for k in range(features):
                
                
                Y_train=srm_train[:,k]
                Y_test=srm_test[:,k]
                
                Y_train -= np.mean(Y_train, axis=0)
                Y_test -= np.mean(Y_train, axis=0)
                
                clf_linear=LinearRegression()
                clf_linear.fit(X_train,Y_train)
                prediction_linear=clf_linear.predict(X_test)
                
                c.append(np.corrcoef(Y_test,prediction_linear)[0,1])
                del clf_linear
            # print(np.mean(c))    
            corr_srm.append(np.mean(c))

        final_corr[p,:]=corr_srm
        p=p+1

    return final_corr

            
shared_split_cv=np.zeros((cv,len(lags)))

if compute_shared_split_regression:

    print('Compute shared split regression')

    features=5

    shared_split_cv= shared_space_split(Y_data,word_embeddings,elec_num,features,cv)


# shuffling electrodes

def shuffling_electrode(Y_data, elec_num):

    Y_data_shuffled= np.zeros((np.shape(Y_data)[0],np.shape(Y_data)[1],np.shape(Y_data)[2],np.shape(Y_data)[3])) #45 is the electrode number here

    for i in range(np.shape(Y_data)[1]):

        for qq in range(np.shape(Y_data)[0]):
            
            for k in range(elec_num[qq]):
                
                random_subject=random.randint(0, len(subjects)-1)
                
                w=elec_num[random_subject]
                
                random_elec=random.randint(0, w-1)
                
                a=Y_data[random_subject,i,:,random_elec]
                
                Y_data_shuffled[qq,i,:,k]=a


    return Y_data_shuffled


if compute_shuffling_electrode:

    kf = KFold(n_splits=cv)
    p=0
    corr_with_subjects=np.zeros((cv,len(lags),len(subjects)))
    Y_data_shuffled=shuffling_electrode(Y_data, elec_num)

    for train_index, test_index in kf.split(word_embeddings):

            print('Fold:',p)
            
            corr_with_subjects[p,:,:]=encoding_with_augmentation(Y_data_shuffled,word_embeddings,elec_num, train_index, test_index)
        
            p=p+1


    corr_original=[]

    
    print('Computing original Regression CV')

    for qq in range(len(subjects)):
        
        corr_original.append(computing_regression_cv(Y_data_shuffled[qq,:,:,:], word_embeddings, elec_num[qq],cv))


def srm_generalization(Y_data, word,elec_num, train_index,test_index):

    X_train= word[train_index,:]
    X_test= word[test_index,:]
    # X1=np.tile(X_test,(len(subjects)-1,1))
    #X2=np.tile(X_train,(len(subjects)-1,1))
    #X1=X_test
    #X=np.concatenate((X_train, X2, ), axis=0)
    #X=np.concatenate((X_train, X1, ), axis=0)
    X=X_train
    X -= np.mean(X, axis=0)
    X_test2 =X_test- np.mean(X, axis=0)
    
    lag=np.shape(Y_data)[1]
    n_iter = 100
    
    # corr_with_742=[]
    # corr_with_798=[]
    final_corr=[]
    for i in range(lag):
        
        corr_with_subjects=[]
        
        for qq1 in range(len(subjects)):

            sub_now=subjects[qq1]
            train_data=[]
            test_data=[]
            subject_elec=elec_num[qq1]

            for qq in range(len(subjects)):
    
                if subjects[qq]!=sub_now:

                    data=Y_data[qq,i,:,:]
                    data=data[:,:elec_num[qq]]


                    train1=data[train_index,:].T
                    test1=data[test_index,:].T

                    train_data.append(train1)
                    test_data.append(test1)
                    
                else:
                    data=Y_data[qq,i,:,:]
                    data=data[:,:elec_num[qq]]


                    train1=data[train_index,:].T
                    test1=data[test_index,:].T

                    self_train=train1
                    self_test=test1

            features=5
            srm = brainiak.funcalign.srm.SRM(n_iter=n_iter, features=5)
            srm.fit(train_data)

            shared_train=srm.transform(train_data)
            shared_test=srm.transform(test_data)

            s_avg_train=shared_train[0]
            s_avg_test=shared_test[0]
            
            for ii in range(1,(len(subjects)-1)):
                s_avg_train=s_avg_train+shared_train[ii]
                s_avg_test=s_avg_test+shared_test[ii]
                

            s_avg_train=s_avg_train/(int(len(subjects)-1))
            s_avg_test=s_avg_test/(int(len(subjects)-1))

            Y_train1=s_avg_train.T
            # Y_train2=s_avg_test.T
            
            # transform_train=[]
            # transform_test=[]
            w_new=srm.transform_subject(self_train)
            # q1=w_new.T.dot(self_train)
            # Y_train=(w_new.dot(q1)).T

            # transform_train=(converted_train1[k])
            # transform_train.append(train_data[k])

            q2=w_new.T.dot(self_test)
            Y_test=q2.T
            # Y_test=(w_new.dot(q2)).T
            # transform_test.append(srm1.w_[k].dot(q2))
            # transform_test.append(self_test)

            corr1_with_717=[]
            for k in range(features):
                
                
                Y1=Y_train1[:,k]
                # Y2=Y_train2[:,k]

                #Y=np.concatenate((Y_train[:,k], Y_aug_train[:,k],), axis=0)
                #Y=np.concatenate((Y_train[:,k], Y_aug_test[:,k],), axis=0)
                
                Y1 -= np.mean(Y1, axis=0)
                # Y2 -= np.mean(Y2, axis=0)
                Y_test1 =Y_test[:,k]- np.mean(Y1, axis=0)
                # Y_test2 =Y_test[:,k]- np.mean(Y2, axis=0)
                
                                
                clf_linear2=LinearRegression()
                clf_linear2.fit(X,Y1)
                prediction_linear2=clf_linear2.predict(X_test2)
                corr1_with_717.append(np.corrcoef(Y_test1,prediction_linear2)[0,1])
                del clf_linear2
                
                
            corr_with_subjects.append(np.mean(corr1_with_717))

            #breakpoint()

            del Y_train1
            del Y_test1
            
        ww=np.array(corr_with_subjects)
            
        final_corr.append(ww)     
    

    return final_corr


kf = KFold(n_splits=cv)
p=0
corr_generalization=np.zeros((cv,len(lags),len(subjects)))

if compute_srm_generalization:
    print('Computing srm generalization') 

    for train_index, test_index in kf.split(word_embeddings):

        print('Fold:',p)
        
        corr_generalization[p,:,:]=srm_generalization(Y_data,word_embeddings,elec_num, train_index, test_index)
    
        p=p+1


## mat file create for ridge

def srm_ridge(Y_data,word,elec_num,features,cv):
    
    
    lag=np.shape(Y_data)[1]
    n_iter = 100
    final_corr=np.zeros((cv,lag))
    kf = KFold(n_splits=cv)
    p=0
    
    for train_index, test_index in kf.split(word):
        corr_srm=[]
    
        for i in range(lag):
            
            train_data=[]
            test_data=[]
            
            for qq in range(len(subjects)):
                
                data=Y_data[qq,i,:,:]
                data=data[:,:elec_num[qq]]
                
                
                train1=data[train_index,:].T
                test1=data[test_index,:].T
                
                train_data.append(train1)
                test_data.append(test1)
            
            
        
            # data1=data11[i,:,:]
            # data2=data22[i,:,:]
            # data3=data33[i,:,:]
            
            # train_717=data1[train_index,:].T
            # test_717=data1[test_index,:].T
            
            # train_742=data2[train_index,:].T
            # test_742=data2[test_index,:].T
            
            # train_798=data3[train_index,:].T
            # test_798=data3[test_index,:].T
            
            # train_data=[]
            # train_data.append(train_717)
            # train_data.append(train_742)
            # train_data.append(train_798)
            
            # test_data=[]
            # test_data.append(test_717)
            # test_data.append(test_742)
            # test_data.append(test_798)
            
              # How many iterations of fitting will you perform
        
            # Create the SRM object
            srm = brainiak.funcalign.srm.SRM(n_iter=n_iter, features=features)
            
            srm.fit(train_data)
            
            shared_test = srm.transform(test_data)
            shared_train = srm.transform(train_data)
            
            s_avg_train=shared_train[0]
            s_avg_test=shared_test[0]
            for ii in range(1,len(subjects)):
                s_avg_train=s_avg_train+shared_train[ii]
                s_avg_test=s_avg_test+shared_test[ii]

            s_avg_train=s_avg_train/len(subjects)
            s_avg_test=s_avg_test/len(subjects)

            X_train= word[train_index,:]
            X_test= word[test_index,:]
            
            X_train -= np.mean(X_train, axis=0)
            X_test -= np.mean(X_train, axis=0)
            
            srm_train=s_avg_train.T
            srm_test=s_avg_test.T


            path='/scratch/gpfs/arnab/Encoding/mat_file_ridge/'
            os.chdir(path)

            filename_ridge='srm_fold_'+str(p)+'_'+str(i)+'.mat'

            savemat(filename_ridge,{'srm_train':srm_train,'srm_test':srm_test,'X_train':X_train,'X_test':X_test})
            
        #     c=[]       
        #     for k in range(features):
                
                
        #         Y_train=srm_train[:,k]
        #         Y_test=srm_test[:,k]
                
        #         Y_train -= np.mean(Y_train, axis=0)
        #         Y_test -= np.mean(Y_train, axis=0)
                
        #         clf_linear=LinearRegression()
        #         clf_linear.fit(X_train,Y_train)
        #         prediction_linear=clf_linear.predict(X_test)
                
        #         c.append(np.corrcoef(Y_test,prediction_linear)[0,1])
        #         del clf_linear
        #     # print(np.mean(c))    
        #     corr_srm.append(np.mean(c))
            
        # final_corr[p,:]=corr_srm
        p=p+1
            
    return final_corr

srm_corr_cv=np.zeros((len(srm_k),cv,len(lags)))

if compute_srm_ridge:

    print('Compute shared space regression')

    for k in range(len(srm_k)):
    
        a= srm_ridge(Y_data,word_embeddings,elec_num,srm_k[k],cv)


## new work

def srm_reconstruction(Y_data,elec_num, train_index,test_index):

    lag=np.shape(Y_data)[1]
    n_iter = 100
    
    # corr_with_742=[]
    # corr_with_798=[]
    final_corr=[]
    final_corr1=np.zeros((np.shape(Y_data)[1],np.shape(Y_data)[0]))
#     final_std=[]
    for i in range(lag):
        
        corr_with_subjects=[]
        
        for qq1 in range(len(subjects)):

            sub_now=subjects[qq1]
            train_data=[]
            test_data=[]
            subject_elec=elec_num[qq1]

            for qq in range(len(subjects)):
    
                if subjects[qq]!=sub_now:

                    data=Y_data[qq,i,:,:]
                    data=data[:,:elec_num[qq]]


                    train1=data[train_index,:].T
                    test1=data[test_index,:].T

                    train_data.append(train1)
                    test_data.append(test1)
                    
                else:
                    data=Y_data[qq,i,:,:]
                    data=data[:,:elec_num[qq]]


                    train1=data[train_index,:].T
                    test1=data[test_index,:].T

                    self_train=train1
                    self_test=test1

            srm = brainiak.funcalign.srm.SRM(n_iter=n_iter, features=5)
            srm.fit(train_data)
            
            # transform_train=[]
            # transform_test=[]
            
            shared_test_srm= srm.transform(test_data)
            
            w_new=srm.transform_subject(self_train)
            q1=w_new.T.dot(self_train)
            Y_train=(w_new.dot(q1)).T
            # transform_train=(converted_train1[k])
            # transform_train.append(train_data[k])
            
            s_test=shared_test_srm[0]
            for kk in range(1,len(subjects)-1):

                s_test=s_test+shared_test_srm[kk]
                
            s_test=s_test/(len(subjects)-1)
            
            reconstructed_data=w_new.dot(s_test)
            
#             print(np.shape(reconstructed_data))
#             print(np.shape(self_test))
            
            corr=[]
            for elec in range(np.shape(reconstructed_data)[0]):
                
                corr.append(np.corrcoef(reconstructed_data[elec,:],self_test[elec,:])[0,1])
                
            final_corr.append(np.mean(corr))
#             final_std.append(np.std(corr)/sqrt(np.shape(reconstructed_data)[0]))
            
        final_corr1[i,:]=final_corr
        final_corr=[]
            
#         ww=np.array(corr_with_subjects)
            
#         final_corr.append(ww)     
        
    

    return final_corr1


def srm_reconstruction_sub_matrix(Y_data,elec_num, train_index,test_index):

    lag=np.shape(Y_data)[1]
    n_iter = 100
    
    # corr_with_742=[]
    # corr_with_798=[]
    final_corr=[]
    final_corr1=np.zeros((np.shape(Y_data)[0],np.shape(Y_data)[1],np.shape(Y_data)[1]))
#     final_std=[]
    for i in range(lag):
        
        corr_with_subjects=[]
        
        for qq1 in range(len(subjects)):

            sub_now=subjects[qq1]
            train_data=[]
            test_data=[]
            subject_elec=elec_num[qq1]

            for qq in range(len(subjects)):
    
                if subjects[qq]!=sub_now:

                    data=Y_data[qq,i,:,:]
                    data=data[:,:elec_num[qq]]


                    train1=data[train_index,:].T
                    test1=data[test_index,:].T

                    train_data.append(train1)
                    test_data.append(test1)
                    
                else:
                    data=Y_data[qq,i,:,:]
                    data=data[:,:elec_num[qq]]


                    train1=data[train_index,:].T
                    test1=data[test_index,:].T

                    self_train=train1
                    self_test=test1

            srm = brainiak.funcalign.srm.SRM(n_iter=n_iter, features=5)
            srm.fit(train_data)
            
            # transform_train=[]
            # transform_test=[]
            
            shared_test_srm= srm.transform(test_data)
            
            w_new=srm.transform_subject(self_train)
            q1=w_new.T.dot(self_train)
            Y_train=(w_new.dot(q1)).T
            # transform_train=(converted_train1[k])
            # transform_train.append(train_data[k])
            
            s_test=shared_test_srm[0]
            for kk in range(1,len(subjects)-1):

                s_test=s_test+shared_test_srm[kk]
                
            s_test=s_test/(len(subjects)-1)
            
            reconstructed_data=w_new.dot(s_test)
            
#             print(np.shape(reconstructed_data))
#             print(np.shape(self_test))
            corr_all_lag=[]
            for kk in range(lag):

                data=Y_data[qq1,kk,:,:]
                data=data[:,:subject_elec]
                test=data[test_index,:].T

            
                corr=[]
                for elec in range(np.shape(reconstructed_data)[0]):
                    
                    corr.append(np.corrcoef(reconstructed_data[elec,:],test[elec,:])[0,1])
            
                corr_all_lag.append(np.mean(corr))    

            final_corr1[qq1,i,:]=corr_all_lag
            #final_corr.append(np.mean(corr))
#             final_std.append(np.std(corr)/sqrt(np.shape(reconstructed_data)[0]))
            
        # final_corr1[i,:]=final_corr
        # final_corr=[]
            
#         ww=np.array(corr_with_subjects)
            
#         final_corr.append(ww)     
        
    

    return final_corr1


kf = KFold(n_splits=cv)
p=0
# corr_reconstruction_sub=np.zeros((cv,len(lags),len(subjects))) ## for srm_reconstruction

corr_reconstruction_sub=np.zeros((cv,len(subjects),len(lags),len(lags))) ## for srm_reconstruction_sub_matrix
# corr_std=np.zeros((cv,len(lags),len(subjects)))

if srm_reconstruction_sub:

    print('Compute SRM reconstruction sub')

    for train_index, test_index in kf.split(word_embeddings):

#         print('Fold:',p)
        
        # corr_reconstruction_sub[p,:,:]=srm_reconstruction(Y_data,elec_num, train_index, test_index)

        corr_reconstruction_sub[p,:,:,:]=srm_reconstruction_sub_matrix(Y_data,elec_num, train_index, test_index) ## for srm_reconstruction_sub_matrix
           
        p=p+1

corr_reconstruction_area=np.zeros((cv,len(lags),len(subjects)))

if srm_reconstruction_area:   

    area='STG'      

    print('Compute SRM reconstruction area')
    print(area)

    path='/scratch/gpfs/arnab/Encoding/'
    os.chdir(path)
    df_all_sub=pd.read_csv('all_subject_sig_with_area.csv')
    subjects=np.unique(df_all_sub['subject'])
    # lags=[0]
    shift_ms=300 # this is a dummy parameter, not used
    cv=10


    electrodes1=[]
    elec_num=[]
    new_subjects=[]

    
    for qq in range(len(subjects)):
        
        subject=subjects[qq]
        #print('subject:',subject)
        s=np.array(df_all_sub.loc[(df_all_sub.subject==subject) & (df_all_sub.area==area)].matfile)
        
        if len(s)>5:
            new_subjects.append(subject)
            electrodes1.append(s)
            elec_num.append(len(s))
        
    # new_subjects=subjects[np.where(np.asarray(elec_num)>10)]
    # electrodes1=np.asarray(electrodes1)
    # electrodes2=electrodes1[np.where(np.asarray(elec_num)>10)]

    subjects=np.asarray(new_subjects)

    print(elec_num)
    print(subjects)

    Y_data= np.zeros((len(new_subjects),len(lags),len(onsets),max(elec_num))) #45 is the electrode number here

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

    kf = KFold(n_splits=cv)
    p=0
    corr_reconstruction_area=np.zeros((cv,len(lags),len(subjects)))

    for train_index, test_index in kf.split(word_embeddings):
       
        corr_reconstruction_area[p,:,:]=srm_reconstruction(Y_data,elec_num, train_index, test_index)
    
        p=p+1


## pca generalisation across subjecy
## take n-1 subject, do pca and build encoding model, project the left-out subject
## and test your encoding model
        
def pca_procustes(train,shared_train):

    A=np.matmul(train, (shared_train.T))

    U, S, V = np.linalg.svd(A, full_matrices=False)

    #                 W.append(np.matmul(U,V.T))
    #                 W.append(np.matmul(U,V))
    W=(np.matmul(U,V))
    
    return W



def pca_generalization_across_subject(Y_data, word,elec_num, train_index,test_index):

    X_train= word[train_index,:]
    X_test= word[test_index,:]
   
    X=X_train
    X -= np.mean(X, axis=0)
    X_test2 =X_test- np.mean(X, axis=0)
    
    lag=np.shape(Y_data)[1]
    
    
    # corr_with_742=[]
    # corr_with_798=[]
    final_corr=[]
    for i in range(lag):
        
        corr_with_subjects=[]
        
        for qq1 in range(len(subjects)):

            pw=0

            sub_now=subjects[qq1]
            train_data=[]
            test_data=[]
            subject_elec=elec_num[qq1]

            for qq in range(len(subjects)):
    
                if subjects[qq]!=sub_now:

                    data=Y_data[qq,i,:,:]
                    data=data[:,:elec_num[qq]]


                    temp_train=data[train_index,:]
                    # test1=data[test_index,:]

                    # temp_test=a1[test_index,:]

                    if pw==0:
                        data_train1=temp_train
                        pw=1

                    else:    
                
                        data_train1=np.concatenate((data_train1,temp_train),axis=1)
                    # data_test1=np.concatenate((data_test1,temp_test),axis=1)

                    # train_data.append(train1)
                    # test_data.append(test1)
                    
                else:
                    data=Y_data[qq,i,:,:]
                    data=data[:,:elec_num[qq]]


                    train1=data[train_index,:]
                    test1=data[test_index,:]

                    self_train=train1
                    self_test=test1

            value=5
            pca = PCA(n_components=value)

            Y_train1 = pca.fit_transform(data_train1)
            aaa=np.zeros((np.shape(self_test)[0],np.shape(data_train1)[1]-np.shape(self_test)[1]))
            data_test=np.concatenate((self_test,aaa),axis=1)

            W=pca_procustes(self_train.T,Y_train1.T)

            Y_test=(np.matmul(W.T, self_test.T)).T

            # Y_test = pca.transform(data_test)

            # breakpoint() 

            corr1_with_717=[]
            for k in range(value):
                                
                Y1=Y_train1[:,k]
                
                Y1 -= np.mean(Y1, axis=0)
                Y_test1 =Y_test[:,k]- np.mean(Y1, axis=0)
                
                                
                clf_linear2=LinearRegression()
                clf_linear2.fit(X,Y1)
                prediction_linear2=clf_linear2.predict(X_test2)
                corr1_with_717.append(np.corrcoef(Y_test1,prediction_linear2)[0,1])
                del clf_linear2
                
                
            corr_with_subjects.append(np.mean(corr1_with_717))

            #breakpoint()

            del Y_train1
            del Y_test1
            del data_train1
            del data_test
            
        ww=np.array(corr_with_subjects)
            
        final_corr.append(ww)     
    

    return final_corr


kf = KFold(n_splits=cv)
p=0
pca_corr_generalization=np.zeros((cv,len(lags),len(subjects)))

if compute_pca_generalization_across_subject:
    print('Computing pca generalization') 

    for train_index, test_index in kf.split(word_embeddings):

        print('Fold:',p)
        
        pca_corr_generalization[p,:,:]=pca_generalization_across_subject(Y_data,word_embeddings,elec_num, train_index, test_index)
    
        p=p+1

# # path=os.path.join(os.pardir)
# # os.chdir(path)

path='/scratch/gpfs/arnab/Encoding/result/'
os.chdir(path)

savemat(filename_mat,{'corr_original':corr_original,'lags':lags,'subjects':subjects, 'corr_with_subjects':corr_with_subjects,'srm_shared_corr_cv':srm_corr_cv,'pca_cv':pca_cv,'pca_across_subject':pca_across_sub,'corr_original_per_elec':corr_original_per_elec,'corr_with_subjects_per_elec':corr_with_subjects_per_elec,'srm_regression_all_elec':corr_with_subjects_all_elec,'corr_original_all_elec':corr_original_all_elec,'srm_leave_one_out':corr_leave_one_out,'shared_split_regression':shared_split_cv,'srm_generalization':corr_generalization,'srm_reconstruction_sub':corr_reconstruction_sub,'srm_reconstruction_area':corr_reconstruction_area, 'pca_generalization_across_sub':pca_corr_generalization})





