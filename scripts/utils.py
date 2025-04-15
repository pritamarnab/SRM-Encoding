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
from brainiak.utils import utils
from brainiak.funcalign import srm
import brainiak


## functions for plotting

def fdr(pvals):
    _, pcor, _, _ = multitest.multipletests(
        pvals, method="fdr_bh", is_sorted=False
    )
    return pcor

def get_sig_lags1( lags, a,b, threshold=0.01):
    #sig_lags = {}
    
    # df_prob = a#df_key[df_key.index.get_level_values("label") == label1]
    # df_improb =b# df_key[df_key.index.get_level_values("label") == label2]
    # df_prob.sort_values([("electrode")], ascending=True, inplace=True)
    # df_improb.sort_values([("electrode")], ascending=True, inplace=True)

    ts = []
    rs = []
    for df_col in np.arange(0,a.shape[1]):
        # r = ttest_ind(a[:,df_col], b[:,df_col],alternative="two-sided")
        r = ttest_rel(a[:,df_col], b[:,df_col],alternative="two-sided") #ttest_rel
        ts.append(r[0])
        rs.append(r[1])
    rs = fdr(rs)

    # threshold = 0.05
    sig_lags = [lags[idx] for (idx, r) in enumerate(rs) if ( ts[idx] > 0 and  r < threshold)]
    #sig_lags[f"{key}_{label2}"] = [lags[idx] for (idx, r) in enumerate(rs) if (ts[idx] < 0 and r < threshold)]
    
    return sig_lags


def plot_data_prep(a):
    
    mu=np.mean(a,axis=0)
    
    q=np.std(a, axis=0)

    err=q/np.sqrt(np.shape(a)[0])
    
    a1=mu-err
    a2=mu+err
    
    return a1,a2,mu

## functions for analysis

def computing_regression_cv(data1, word_embeddings,elec_num,cv):
    
    lag=np.shape(data1)[0]
    
    final_corr=np.zeros((cv,lag))
    
    kf = KFold(n_splits=cv)
    p=0
    
    for train_index, test_index in kf.split(word_embeddings):
        
        corr_lin2=[]
        
        X_train,X_test = word_embeddings[train_index,:],word_embeddings[test_index,:]
    
        X_train -= np.mean(X_train, axis=0)
        X_test -= np.mean(X_train, axis=0)
        
        for i in range(lag):
            
            corr_lin1=[]
            
            data=data1[i,:,:]
            data=data[:,:elec_num]
        
            for k in range(np.shape(data)[1]):
                
                label=data[:,k]  
                Y_train,Y_test = label[train_index],label[test_index]
                
       
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

def srm_regression_cv(Y_data,word,elec_num,features,cv,subjects):
    
    
    lag=np.shape(Y_data)[1]
    n_iter = 100
    final_corr=np.zeros((cv,lag))
    final_corr_all_features=np.zeros((cv,lag,features))

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
            final_corr_all_features[p,i,:]=c 
            
        final_corr[p,:]=corr_srm
        p=p+1
            
    return final_corr,final_corr_all_features

def pca_data_prep(data1,train_index,test_index,value):
    
    a1=data1[train_index,:] 
    b1=data1[test_index,:]
    a1=a1-(np.mean(a1,axis=0))
    b1=b1-(np.mean(a1,axis=0))
   
    
    pca = PCA(n_components=value)

    train_717_pca = pca.fit_transform(a1)
    test_717_pca = pca.transform(b1)
    
    return train_717_pca,test_717_pca

def pca_across_subject(Y_data, elec_num, subjects, word_embeddings,cv=10,value=5):
    
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

def encoding_with_augmentation(Y_data, word,elec_num, train_index,test_index, subjects):
    
    X_train= word[train_index,:]
    X_test= word[test_index,:]
    
    X=X_train
    X -= np.mean(X, axis=0)
    X_test2 =X_test- np.mean(X, axis=0)
    
    lag=np.shape(Y_data)[1]
    n_iter = 100
    
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

            corr1_with_717=[]
            for k in range(subject_elec):
                
                Y=Y_train[:,k]

                Y -= np.mean(Y, axis=0)
                Y_test2 =Y_test[:,k]- np.mean(Y, axis=0)
                
                                
                clf_linear2=LinearRegression()
                clf_linear2.fit(X,Y)
                prediction_linear2=clf_linear2.predict(X_test2)
                corr1_with_717.append(np.corrcoef(Y_test2,prediction_linear2)[0,1])
                del clf_linear2
                
                
            corr_with_subjects.append(np.mean(corr1_with_717))

            del Y_train
            del Y_test
            
        ww=np.array(corr_with_subjects)
        
        final_corr.append(ww)        
    return final_corr

def srm_generalization(Y_data, word,elec_num, train_index,test_index, subjects):

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

def pca_procustes(train,shared_train):

    A=np.matmul(train, (shared_train.T))

    U, S, V = np.linalg.svd(A, full_matrices=False)

    #                 W.append(np.matmul(U,V.T))
    #                 W.append(np.matmul(U,V))
    W=(np.matmul(U,V))
    
    return W



def pca_generalization_across_subject(Y_data, word,elec_num, train_index,test_index,subjects):

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


def srm_leave_one_out(Y_data, word,elec_num, train_index,test_index, subjects):

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

def srm_regression_all_elec(Y_data, word,elec_num, train_index,test_index,subjects):
    
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



            
    return final_corr1, final_corr

def plot_area(original_all_elec,srm_all_elec, index,name):
    
    original=original_all_elec[index,:]
    srm=srm_all_elec[index,:]
    
    w=get_sig_lags1( lags, srm,original)
    
    original_elec_area=np.zeros((3,len(lags)))
    srm_elec_area=np.zeros((3,len(lags)))
    
    [a1,a2,mu]=plot_data_prep(original)
    original_elec_area[0,:]=a1
    original_elec_area[1,:]=a2
    original_elec_area[2,:]=mu
    
    [a1,a2,mu]=plot_data_prep(srm)
    srm_elec_area[0,:]=a1
    srm_elec_area[1,:]=a2
    srm_elec_area[2,:]=mu
    
    plt.figure(figsize=(3.5, 2.7))
    
    plt.fill_between(lags, original_elec_area[0,:], original_elec_area[1,:], alpha=0.2)
    plt.plot(lags, original_elec_area[2,:], linewidth=3.5, label = "original")  #, color=c1, ls=ls, lw = lw)
    
    plt.fill_between(lags, srm_elec_area[0,:], srm_elec_area[1,:], alpha=0.2)
    plt.plot(lags, srm_elec_area[2,:], linewidth=3.5, label = "SRM")  #, color=c1, ls=ls, lw = lw)
    
    plt.scatter(w, # (x)
                    np.full(len(w), 0.001), # (y)
                    color='red')

    plt.axvline(0, ls="dashed", alpha=0.3, c="grey")
    plt.legend()
    plt.xlabel('lags (ms)')
    plt.ylabel('Correlation')
    plt.title(name)
    plt.ylim([0,0.4])
    plt.tight_layout()
    # plt.savefig('/scratch/gpfs/arnab/Encoding/aSTG.png',dpi=600)

def shared_info_across_subjects(Y_data,elec_num, train_index,test_index,subjects):

    lag=np.shape(Y_data)[1]
    n_iter = 20
    
    # corr_with_742=[]
    # corr_with_798=[]
    final_corr=[]
#     final_std=[]
    for i in range(lag):
        
        corr_with_subjects=[]
        
        for qq1 in range(len(subjects)):

            sub_now=subjects[qq1]
            # print(sub_now)
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
            
            
            
#         ww=np.array(corr_with_subjects)
            
#         final_corr.append(ww)     
        
    

    return final_corr