import os
import pickle
import numpy as np
import pandas as pd  
import csv
from scipy.io import savemat
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from functools import reduce 

from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.linear_model import Ridge,Lasso
# from sklearn.ensemble import RandomForestRegressor

# from nltk.corpus import stopwords
# stops = stopwords.words('english')

from brainiak.isc import isc
from brainiak.fcma.util import compute_correlation
import brainiak.funcalign.srm

subjects=[717,742,798]
lags=[-400,-350,-300,-250,200-150,-100,-50,0,50,100,150,200,250,300,350,400]
srm_k=[15,20,25,30,35,45] #K value for SRM

shift_ms=300 # this is a dummy parameter, not used


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
                  
          
          
          
        p=p+1
    
    pca = PCA(n_components=50)
    word_embeddings2=word_embeddings2- np.mean(word_embeddings2, axis=0)
    word_embeddings=pca.fit_transform(word_embeddings2)
    word_embeddings = normalize(word_embeddings, axis=1, norm='l2')
    
   
    return onset,word_embeddings

[onsets,word_embeddings]= data_create(shift_ms)

Y_717= np.zeros((len(lags),len(onsets),45)) #45 is the electrode number here
Y_742= np.zeros((len(lags),len(onsets),45))
Y_798= np.zeros((len(lags),len(onsets),45))


for subject in subjects:
    
    print('subject:',subject)
    
    if subject==717:
        
        # path1='C:\Princeton\Research\Hyper_alignment\significant electrode files\Patient '
        # path=path1+ str(subject)+'\All significant electrode'
        # os.chdir(path)
        
        path=os.path.join(os.pardir,'significant_electrode_podcast','717')
        os.chdir(path)
        
        #print(path)
            
        ifg_significant=[4,9,10,18,27,66,71,74,75,78,79,80,82,86,87,88,95,108]
        ifg_non_significant=[17]
        stg_significant=[36,37,38,39,46,47,112,113,114,116,117,119,120,121,122,126]
        stg_non_significant=[40,45]
        aMTG_significant=[44]
        precentral_significant=[85]
        precentral_non_significant=[21]
        other_significant=[158,179,173,174,175,176]
        parietal_significant=[32]
        TP_non_significant=[41,42,43,49,50]
        aMTG_non_significant=[128]
        MFG_non_significant=[1,65,68,69,70,153,161,162,70,153]
        postcg_non_significant=[8,14,15,22,30,91,100,101,102,103,110,111,16,23,104,90,8,14,15]
        
        #electrodes=[ifg_significant,ifg_non_significant,stg_significant,stg_non_significant,aMTG_significant,precentral_significant,precentral_non_significant,TP_non_significant]
        
        #electrodes=[4,9,10,18,27,66,71,74,75,78,79,80,82,86,87,88,95,108,17,36,37,38,39,46,47,112,113,114,116,117,119,120,121,122,126,40,45,44,85,21,41,42,43,49,50]
        
        electrodes=[ifg_significant,ifg_non_significant,stg_significant,stg_non_significant,aMTG_significant,precentral_significant,precentral_non_significant,TP_non_significant]#,MFG_non_significant,postcg_non_significant]
        # electrodes=[ifg_significant,ifg_non_significant,stg_significant,stg_non_significant,aMTG_significant,precentral_significant,precentral_non_significant,TP_non_significant]#,MFG_non_significant,postcg_non_significant]
        electrodes = reduce(lambda a, b: a+b, electrodes)


    if subject==798:
        
        # path='C:\Princeton\Research\Hyper_alignment\significant electrode files\Patient 798\\all'
        # os.chdir(path)
        
        path=os.path.join(os.pardir,'798')
        os.chdir(path)
        
        ifg_significant=[12,17,18,19,20,26,27,77,79,83,84,86,87,88,92,93,94,95,96]
        stg_significant=[56]
        #stg_non_significant=[116,117,124,125,128,41,34,42,43,44,45,46,52,53,54,55,55]  #original
        stg_non_significant=[116,43,52,55,31]
        aMTG_non_significant=[51]
        precentral_significant=[16]
        precentral_non_significant=[21]
        other_significant=[190,191,154]
        TP_significant=[131]
        TP_non_significant=[132,49,50,57]
        AG_significant=[140,142]
        MFG_significant=[10,5,164]
        premotor_significant=[15]
        MFG_non_significant=[3,4,66,67]
        postcg_non_significant=[24,30,31,37,38,91,102,103,104,105,110,111,112,113,114,118,119,120,24]
        #electrodes=[ifg_significant,ifg_non_significant,stg_significant,stg_non_significant,aMTG_significant,precentral_significant,precentral_non_significant,TP_non_significant]

        #electrodes=[12,17,18,19,20,26,27,77,79,83,84,86,87,88,92,93,94,95,96,56,116,117,124,125,128,41,34,42,43,44,45,46,52,53,54,55,55,51,16,21,131,132,49,50,57]

        # electrodes=[ifg_significant,stg_significant,stg_non_significant,aMTG_non_significant,precentral_significant,precentral_non_significant,TP_significant,TP_non_significant]#,MFG_non_significant,postcg_non_significant]
        #electrodes=[postcg_non_significant,MFG_non_significant]
        electrodes=[ifg_significant,stg_significant,stg_non_significant, MFG_non_significant,precentral_significant,precentral_non_significant,TP_significant,other_significant,AG_significant,MFG_significant,premotor_significant,TP_non_significant]#,MFG_non_significant,postcg_non_significant]
        electrodes = reduce(lambda a, b: a+b, electrodes)

    if subject==742:
        
        # path='C:\Princeton\Research\Hyper_alignment\significant electrode files\Patient 742\significant electrodes from masterlist'
        # os.chdir(path)
        
        path=os.path.join(os.pardir,'742')
        os.chdir(path)
        
        ifg_significant=[34,35,42,43,44,45,53,117,118,124,128]
        ifg_non_significant=[116,26,36,51,52,59,60,116]
        stg_significant=[101,102,103,104,110,111,113,119,120,121,125,15,22,23,30,31,37,7]
        aMTG_significant=[24]
        precentral_significant=[108,75]
        other_significant=[61,165,159]
        TP_significant=[54,55,56,63,64]
        MFG_non_significant=[25,33,41,49,50,57,58,106,25,33]
        postcg_non_significant=[2,3,11,12,19,20,21,29,66,73,76,82,83,84,85,88,89,99,100]
            


        #electrodes=[34,35,42,43,44,45,53,117,118,124,128,116,26,36,51,52,59,60,116,101,102,103,104,110,111,113,119,120,121,125,15,22,23,30,31,37,7,24,108,75,54,55,56,63,64]

        electrodes=[ifg_significant,ifg_non_significant,stg_significant,aMTG_significant,precentral_significant,TP_significant]#,MFG_non_significant,postcg_non_significant]
        electrodes= reduce(lambda a, b: a+b, electrodes)


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
                
                                
            if subject==717:
               
                Y_717[ii,:,k] = np.mean(Y1, axis=-1)
            if subject==742:
                Y_742[ii,:,k] = np.mean(Y1, axis=-1)
            if subject==798:
                Y_798[ii,:,k] = np.mean(Y1, axis=-1)
        

def computing_regression(data1, word_embeddings):
    
    
    corr_lin2=[]
    lag=np.shape(data1)[0]
    
    X_train= word_embeddings[0:4000,:]  #working on the last fold
    X_test= word_embeddings[4000:5013,:]

    X_train -= np.mean(X_train, axis=0)
    X_test -= np.mean(X_train, axis=0)
    
    for i in range(lag):
        
        corr_lin1=[]
        
        data=data1[i,:,:]
    
        for k in range(np.shape(data)[1]):
            
            
            Y_train=data[0:4000,k]  
            Y_test=data[4000:5013,k]
            
            Y_train -= np.mean(Y_train, axis=0)
            Y_test -= np.mean(Y_train, axis=0)
          
            
            
            #We fit the Linear regression to our train set
            clf_linear=LinearRegression()
            clf_linear.fit(X_train,Y_train)
            
            
            prediction_linear=clf_linear.predict(X_test)
            
            corr_lin1.append(np.corrcoef(Y_test,prediction_linear)[0,1])
            
            del clf_linear
            
        corr_lin2.append(np.mean(corr_lin1))
            
    return corr_lin2

print('Computing Regression')
            
corr_717=computing_regression(Y_717, word_embeddings)
corr_798=computing_regression(Y_798, word_embeddings)
corr_742=computing_regression(Y_742, word_embeddings)

print(corr_717)




srm_corr=np.zeros((len(srm_k),len(lags)))


def srm_regression(data11,data22,data33,word,features):
    
    corr_srm=[]
    lag=np.shape(data11)[0]
    n_iter = 1000
    for i in range(lag):
    
        data1=data11[i,:,:]
        data2=data22[i,:,:]
        data3=data33[i,:,:]
        
        train_717=data1[0:4000,:].T
        test_717=data1[4000:5013,:].T
        
        train_742=data2[0:4000,:].T
        test_742=data2[4000:5013,:].T
        
        train_798=data3[0:4000,:].T
        test_798=data3[4000:5013,:].T
        
        train_data=[]
        train_data.append(train_717)
        train_data.append(train_742)
        train_data.append(train_798)
        
        test_data=[]
        test_data.append(test_717)
        test_data.append(test_742)
        test_data.append(test_798)
        
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
        

print('Computing SRM')
    
for k in range(len(srm_k)):

    srm_corr[k,:]= srm_regression(Y_717,Y_742,Y_798,word_embeddings,srm_k[k])
    
print(srm_corr)    
        
def encoding_with_augmentation(data11, data22, data33, word):
    
    X_train= word[0:4000,:]
    X_test= word[4000:5013,:]
    X=np.concatenate((X_train, X_test, ), axis=0)
    X -= np.mean(X, axis=0)
    X_test2 =X_test- np.mean(X, axis=0)
    
    lag=np.shape(data11)[0]
    n_iter = 1000
    corr_with_717=[]
    corr_with_742=[]
    corr_with_798=[]
    for i in range(lag):
    
        data1=data11[i,:,:]
        data2=data22[i,:,:]
        data3=data33[i,:,:]
        
        train_717=data1[0:4000,:].T
        test_717=data1[4000:5013,:].T
        
        train_742=data2[0:4000,:].T
        test_742=data2[4000:5013,:].T
        
        train_798=data3[0:4000,:].T
        test_798=data3[4000:5013,:].T
        
        train_data=[]
        train_data.append(train_717)
        train_data.append(train_742)
        train_data.append(train_798)
        
        test_data=[]
        test_data.append(test_717)
        test_data.append(test_742)
        test_data.append(test_798)
        
        srm = brainiak.funcalign.srm.SRM(n_iter=n_iter, features=5)
        
        srm.fit(train_data)
        
        shared_test = srm.transform(test_data)
        shared_train = srm.transform(train_data)
        
        # for subject 717 augmented with 742
        
        w0 = srm.w_[0]  # Weights for subject 0
        signal_srm0 = w0.dot(shared_test[0])  # Reconstructed test signal for subject 0
        signal_srm1 = w0.dot(shared_test[1])  # Reconstructed test signal for subject 1 in subject 0 space
        signal_srm2 = w0.dot(shared_test[2])  # Reconstructed test signal for subject 2 in subject 0 space
        
        signal_srm0_train = w0.dot(shared_train[0])  # Reconstructed train signal for subject 0
        signal_srm1_train = w0.dot(shared_train[1])  # Reconstructed train signal for subject 1 in subject 0 space
        signal_srm2_train = w0.dot(shared_train[2])  # Reconstructed train signal for subject 2 in subject 0 space
        
       
        corr1_with_717=[]
        for k in range(np.shape(train_data[0])[0]):
          Y_train=signal_srm0_train.T[:,k]
         
        
          Y_aug1=signal_srm1.T[:,k]
          Y_aug2=signal_srm2.T[:,k]
        
          Y=np.concatenate((Y_train, Y_aug1,), axis=0)
        
          Y_test=signal_srm0.T[:,k]
          
          Y -= np.mean(Y, axis=0)
          Y_test2 =Y_test- np.mean(Y, axis=0)
        
          clf_linear2=LinearRegression()
        
          clf_linear2.fit(X,Y)
          prediction_linear2=clf_linear2.predict(X_test2)
          corr1_with_717.append(np.corrcoef(Y_test2,prediction_linear2)[0,1])
          
        corr_with_717.append(np.mean(corr1_with_717))
        
        # for subject 742 augmented with 798
        
        w0 = srm.w_[1]  # Weights for subject 1
        signal_srm0 = w0.dot(shared_test[0])  
        signal_srm1 = w0.dot(shared_test[1])  
        signal_srm2 = w0.dot(shared_test[2])  
        
        signal_srm0_train = w0.dot(shared_train[0])  
        signal_srm1_train = w0.dot(shared_train[1])  
        signal_srm2_train = w0.dot(shared_train[2])  
        
       
        corr1_with_742=[]
        for k in range(np.shape(train_data[0])[0]):
          Y_train=signal_srm1_train.T[:,k]
         
        
          Y_aug1=signal_srm0.T[:,k]
          Y_aug2=signal_srm2.T[:,k]
        
          Y=np.concatenate((Y_train, Y_aug2,), axis=0)
        
          Y_test=signal_srm1.T[:,k]
          
          Y -= np.mean(Y, axis=0)
          Y_test2 =Y_test- np.mean(Y, axis=0)
        
          clf_linear2=LinearRegression()
        
          clf_linear2.fit(X,Y)
          prediction_linear2=clf_linear2.predict(X_test2)
          corr1_with_742.append(np.corrcoef(Y_test2,prediction_linear2)[0,1])
          
        corr_with_742.append(np.mean(corr1_with_717))
        
        # for subject 798 augmented with 717
        
        w0 = srm.w_[2]  # Weights for subject 1
        signal_srm0 = w0.dot(shared_test[0])  
        signal_srm1 = w0.dot(shared_test[1])  
        signal_srm2 = w0.dot(shared_test[2])  
        
        signal_srm0_train = w0.dot(shared_train[0]) 
        signal_srm1_train = w0.dot(shared_train[1]) 
        signal_srm2_train = w0.dot(shared_train[2]) 
        
       
        corr1_with_798=[]
        for k in range(np.shape(train_data[0])[0]):
          Y_train=signal_srm2_train.T[:,k]
         
        
          Y_aug1=signal_srm0.T[:,k]
          Y_aug2=signal_srm1.T[:,k]
        
          Y=np.concatenate((Y_train, Y_aug1,), axis=0)
        
          Y_test=signal_srm2.T[:,k]
          
          Y -= np.mean(Y, axis=0)
          Y_test2 =Y_test- np.mean(Y, axis=0)
        
          clf_linear2=LinearRegression()
        
          clf_linear2.fit(X,Y)
          prediction_linear2=clf_linear2.predict(X_test2)
          corr1_with_798.append(np.corrcoef(Y_test2,prediction_linear2)[0,1])
          
        corr_with_798.append(np.mean(corr1_with_717))
        
        
    return corr_with_717,corr_with_742,corr_with_798
            

print('Computing Augmentation')    
[corr_with_717,corr_with_742,corr_with_798]=encoding_with_augmentation(Y_717,Y_742,Y_798,word_embeddings)

print(corr_with_717)    

path=os.path.join(os.pardir)
os.chdir(path)

savemat('All_result.mat',{'corr_717':corr_717,'corr_742':corr_742,'corr_798':corr_798,'lags':lags,'srm_corr':srm_corr, 'corr_with_717':corr_with_717,'corr_with_742':corr_with_742,'corr_with_798':corr_with_798,})





