# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 13:56:14 2023

@author: arnab
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 10:32:16 2023

@author: arnab
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  
from scipy.io import savemat
from scipy.io import loadmat
from scipy.stats import ttest_ind, ttest_rel #, permutation_test
from statsmodels.stats import multitest
# savemat(filename,{'corr_717':corr_717,'corr_742':corr_742,'corr_798':corr_798,'lags':lags,'srm_corr':srm_corr, 'corr_with_717':corr_with_717,'corr_with_742':corr_with_742,'corr_with_798':corr_with_798,})

filename='All_result_all.mat'
filename1='result_all_sub_compute_augmentation.mat'
filename2='result_all_sub_compute_regression.mat'


filename3='result_all_sub_no_augmentation_selected_4.mat'
filename4='result_all_sub_with_augmentation_selected_4.mat'

# filename5='result_all_sub_no_augmentation.mat'
filename5='srm_result_all_sub_without_augmentation_cv_10.mat'
filename6='result_all_sub_augmentation_with_one_subject.mat'

filename7='rsrm_result_all_sub_without_augmentation_cv_5.mat'
filename8='rsrm_result_all_sub_with_augmentation_cv_5.mat'

filename9='corr_original_function.mat'
filename10='corr_srm_function.mat'

filename11='corr_original_content.mat'
filename12='corr_srm_content.mat'

filename13='pca_regression_k_5.mat'
# filename14='shared_space_regression_k_5.mat'
filename14='shared_space_regression_new.mat'

filename15='original_regression_all_elec.mat'
filename16='srm_denoising_regression_all_elec.mat'

filename17='elec_space_generalization_across_subject.mat'  #'srm_leave_one_out.mat'
filename18='shared_split_regression.mat'

filename19='shuffling_electrode.mat'
filename20='srm_generalization.mat'

filename21='behavior_srm.mat'

filename22='bert_original_regression.mat'
filename23='bert_srm_denoise.mat'

filename24='pca_across_subject.mat'

path='C:\Princeton\Research\Hyper_alignment\encoding\\All_subject'
os.chdir(path)

def fdr(pvals):
    _, pcor, _, _ = multitest.multipletests(
        pvals, method="fdr_bh", is_sorted=False
    )
    return pcor

def get_sig_lags1( lags, a,b):
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

    threshold = 0.01
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



df_all_sub=pd.read_csv('all_subject_sig.csv')
subjects=np.unique(df_all_sub['subject'])
electrodes1=[]
elec_num=[]
for qq in range(len(subjects)):
    
    subject=subjects[qq]
    #print('subject:',subject)
    s=np.array(df_all_sub.loc[df_all_sub.subject==subject].matfile)
    electrodes1.append(s)
    elec_num.append(len(s))


#corr_717=np.squeeze(loadmat(filename)['corr_717'])
#corr_742=np.squeeze(loadmat(filename)['corr_742'])
#corr_798=np.squeeze(loadmat(filename)['corr_798'])


lags=np.squeeze(loadmat(filename2)['lags'])

# subjects=[661,717,723,798]


corr_original=np.squeeze(loadmat(filename2)['corr_original'])
corr_with_subjects=loadmat(filename1)['corr_with_subjects']  #srm

# corr_with_subjects_no_aug=loadmat(filename3)['corr_with_subjects']
# corr_with_subjects_with_aug=loadmat(filename4)['corr_with_subjects']

corr_with_subjects_no_aug=loadmat(filename5)['corr_with_subjects'] #srm
corr_with_subjects_with_aug=loadmat(filename6)['corr_with_subjects']

corr_with_subjects_rsrm=loadmat(filename8)['corr_with_subjects']  #rsrm
corr_with_subjects_no_aug_rsrm=loadmat(filename7)['corr_with_subjects'] #rsrm


corr_original_function=np.squeeze(loadmat(filename9)['corr_original'])
corr_srm_function=np.squeeze(loadmat(filename10)['corr_with_subjects'])

corr_original_content=np.squeeze(loadmat(filename11)['corr_original'])
corr_srm_content=np.squeeze(loadmat(filename12)['corr_with_subjects'])

pca_across_sub1=np.squeeze(loadmat(filename24)['pca_across_subject'])
pca_regression1=np.squeeze(loadmat(filename13)['pca_cv'])
shared_space_regression1=np.squeeze(loadmat(filename14)['srm_shared_corr_cv'])

original_all_elec=np.squeeze(loadmat(filename15)['corr_original_all_elec'])
srm_all_elec=np.squeeze(loadmat(filename16)['srm_regression_all_elec']).T

srm_leave_one_out1=np.squeeze(loadmat(filename17)['srm_leave_one_out'])
shared_split_regression1=np.squeeze(loadmat(filename18)['shared_split_regression'])


shuffled_elec_original=np.squeeze(loadmat(filename19)['corr_original'])
shuffled_elec_srm=np.squeeze(loadmat(filename19)['corr_with_subjects'])

srm_generalize1=np.squeeze(loadmat(filename20)['srm_generalization'])

# behavior_original=loadmat(filename21)['corr_original']
# behavior_srm=loadmat(filename21)['corr_srm']

improvement=loadmat(filename21)['improvement']

bert_corr_original=np.squeeze(loadmat(filename22)['corr_original'])
bert_corr_with_subjects_no_aug=loadmat(filename23)['corr_with_subjects'] #srm


# improvement=[]
# for q in range(4):
#     a=np.max(behavior_original[q,2,:])
#     b=np.max(behavior_srm[q,2,:])
        
#     index_org=np.where(behavior_original[q,2,:]==a)
#     index_srm=np.where(behavior_srm[q,2,:]==b)
    
#     a11=behavior_original[q,1,index_org]
#     a22=behavior_original[q,2,index_org]
    
#     b11=behavior_srm[q,1,index_org]
#     b22=behavior_srm[q,2,index_org]
    
#     improvement.append([(b-a)/a, (b22-a11)/a11, (b11-a22)/a22,])
    
# print(improvement)
# plt.plot(behavior_original[2,2,:])
# plt.plot(behavior_srm[2,2,:])


augmented_data=np.zeros((8,10,160))
cv=10
for c in range(cv):
    
    a=corr_with_subjects[c,:,:]
    
    for qq in range(len(subjects)):
    
        augmented_data[qq,c,:]=a[:,qq]


# aa=corr_with_subjects[:,:,0]
# [a1,a2,mu]=plot_data_prep(aa)


original_regression=np.zeros((len(subjects),3,len(lags)))
augmented_regression_srm=np.zeros((len(subjects),3,len(lags)))
augmented_regression_rsrm=np.zeros((len(subjects),3,len(lags)))

bert_original_regression=np.zeros((len(subjects),3,len(lags)))
bert_no_aug_srm=np.zeros((len(subjects),3,len(lags)))


no_aug_srm=np.zeros((len(subjects),3,len(lags)))
no_aug_rsrm=np.zeros((len(subjects),3,len(lags)))
with_aug=np.zeros((len(subjects),3,len(lags)))

original_function=np.zeros((len(subjects),3,len(lags)))
srm_function=np.zeros((len(subjects),3,len(lags)))

original_content=np.zeros((len(subjects),3,len(lags)))
srm_content=np.zeros((len(subjects),3,len(lags)))

pca_across_sub=np.zeros((3,len(lags)))
pca_regression=np.zeros((3,len(lags)))
shared_space_regression=np.zeros((3,len(lags)))
shared_split_regression=np.zeros((3,len(lags)))

srm_leave_one_out=np.zeros((len(subjects),3,len(lags)))

shuffled_original_regression=np.zeros((len(subjects),3,len(lags)))
shuffled_srm_regression=np.zeros((len(subjects),3,len(lags)))

srm_shared_space_generalization=np.zeros((len(subjects),3,len(lags)))


for qq in range(len(subjects)):
    
    aa=corr_original[qq,:,:]    
    [a1,a2,mu]=plot_data_prep(aa)
    
    original_regression[qq,0,:]=a1
    original_regression[qq,1,:]=a2
    original_regression[qq,2,:]=mu
    
    del aa
    

    aa=corr_with_subjects_no_aug[:,:,qq]
    [a1,a2,mu]=plot_data_prep(aa)
    
    no_aug_srm[qq,0,:]=a1
    no_aug_srm[qq,1,:]=a2
    no_aug_srm[qq,2,:]=mu    
    
    del aa
    
    aa=corr_with_subjects[:,:,qq]
    [a1,a2,mu]=plot_data_prep(aa)
    
    augmented_regression_srm[qq,0,:]=a1
    augmented_regression_srm[qq,1,:]=a2
    augmented_regression_srm[qq,2,:]=mu
    
    del aa
    
    aa=corr_with_subjects_rsrm[:,:,qq]
    [a1,a2,mu]=plot_data_prep(aa)
    
    augmented_regression_rsrm[qq,0,:]=a1
    augmented_regression_rsrm[qq,1,:]=a2
    augmented_regression_rsrm[qq,2,:]=mu
    
    del aa
    
    aa=corr_with_subjects_no_aug_rsrm[:,:,qq]
    [a1,a2,mu]=plot_data_prep(aa)
    
    no_aug_rsrm[qq,0,:]=a1
    no_aug_rsrm[qq,1,:]=a2
    no_aug_rsrm[qq,2,:]=mu
    
    del aa
    
    aa=corr_original_function[qq,:,:]    
    [a1,a2,mu]=plot_data_prep(aa)
    
    original_function[qq,0,:]=a1
    original_function[qq,1,:]=a2
    original_function[qq,2,:]=mu
    
    del aa
    
    aa=corr_srm_function[:,:,qq]
    [a1,a2,mu]=plot_data_prep(aa)
    
    srm_function[qq,0,:]=a1
    srm_function[qq,1,:]=a2
    srm_function[qq,2,:]=mu    
    
    del aa
    
    aa=corr_original_content[qq,:,:]    
    [a1,a2,mu]=plot_data_prep(aa)
    
    original_content[qq,0,:]=a1
    original_content[qq,1,:]=a2
    original_content[qq,2,:]=mu
    
    del aa
    
    aa=corr_srm_content[:,:,qq]
    [a1,a2,mu]=plot_data_prep(aa)
    
    srm_content[qq,0,:]=a1
    srm_content[qq,1,:]=a2
    srm_content[qq,2,:]=mu    
    
    del aa
    
    aa=srm_leave_one_out1[:,:,qq]
    [a1,a2,mu]=plot_data_prep(aa)
    
    srm_leave_one_out[qq,0,:]=a1
    srm_leave_one_out[qq,1,:]=a2
    srm_leave_one_out[qq,2,:]=mu    
    
    del aa
    
    aa=shuffled_elec_original[qq,:,:]    
    [a1,a2,mu]=plot_data_prep(aa)
    
    shuffled_original_regression[qq,0,:]=a1
    shuffled_original_regression[qq,1,:]=a2
    shuffled_original_regression[qq,2,:]=mu
    
    del aa
    

    aa=shuffled_elec_srm[:,:,qq]
    [a1,a2,mu]=plot_data_prep(aa)
    
    shuffled_srm_regression[qq,0,:]=a1
    shuffled_srm_regression[qq,1,:]=a2
    shuffled_srm_regression[qq,2,:]=mu    
    
    del aa
    
    aa=srm_generalize1[:,:,qq]
    [a1,a2,mu]=plot_data_prep(aa)
    
    srm_shared_space_generalization[qq,0,:]=a1
    srm_shared_space_generalization[qq,1,:]=a2
    srm_shared_space_generalization[qq,2,:]=mu    
    
    del aa
    
    aa=bert_corr_original[qq,:,:]    
    [a1,a2,mu]=plot_data_prep(aa)
    
    bert_original_regression[qq,0,:]=a1
    bert_original_regression[qq,1,:]=a2
    bert_original_regression[qq,2,:]=mu
    
    del aa
    

    aa=bert_corr_with_subjects_no_aug[:,:,qq]
    [a1,a2,mu]=plot_data_prep(aa)
    
    bert_no_aug_srm[qq,0,:]=a1
    bert_no_aug_srm[qq,1,:]=a2
    bert_no_aug_srm[qq,2,:]=mu    
    
    del aa

    
       
    

[a1,a2,mu]=plot_data_prep(pca_regression1)
pca_regression[0,:]=a1
pca_regression[1,:]=a2
pca_regression[2,:]=mu

[a1,a2,mu]=plot_data_prep(pca_across_sub1)
pca_across_sub[0,:]=a1
pca_across_sub[1,:]=a2
pca_across_sub[2,:]=mu

[a1,a2,mu]=plot_data_prep(shared_space_regression1)
shared_space_regression[0,:]=a1
shared_space_regression[1,:]=a2
shared_space_regression[2,:]=mu

w=get_sig_lags1( lags, shared_space_regression1,pca_across_sub1)

plt.figure(figsize=(4.8,3.5))

plt.fill_between(lags, shared_space_regression[0,:], shared_space_regression[1,:], alpha=0.2, color='darkorange')
plt.plot(lags, shared_space_regression[2,:], linewidth=3.5, label = "SRM", color='darkorange')  #, color=c1, ls=ls, lw = lw)

plt.fill_between(lags, pca_across_sub[0,:], pca_across_sub[1,:], alpha=0.2)
plt.plot(lags, pca_across_sub[2,:], linewidth=3.5, label = "PCA", color='green')  #, color=c1, ls=ls, lw = lw)

plt.axvline(0, ls="dashed", alpha=0.3, c="k")

plt.scatter(w, # (x)
                np.full(len(w), 0.005), # (y)
                color='red')
plt.ylim([0,0.4])

name='Shared Space Encoding vs PCA control'

plt.legend()
plt.xlabel('lags (ms)')
plt.ylabel('Encoding Performance (r)')
plt.title(name)
plt.tight_layout()
filename=name+'.png'

path1='C:\Princeton\Research\Hyper_alignment\encoding\All_subject\plots\\'
os.chdir(path1)
plt.savefig(filename,dpi=600)



[a1,a2,mu]=plot_data_prep(shared_split_regression1)
shared_split_regression[0,:]=a1
shared_split_regression[1,:]=a2
shared_split_regression[2,:]=mu

plt.fill_between(lags, shared_split_regression[0,:], shared_split_regression[1,:], alpha=0.2)
plt.plot(lags, shared_split_regression[2,:], linewidth=3.5, label = "PCA")  #, color=c1, ls=ls, lw = lw)
plt.axvline(0, ls="dashed", alpha=0.3, c="k")


# shared space generalize vs all shared space encoding

a=np.mean(srm_shared_space_generalization,axis=0)
a2_original=np.mean(original_regression,axis=0)

#elec space srm generalization

a3=np.mean(srm_leave_one_out,axis=0)


path='C:\Princeton\Research\Hyper_alignment\encoding\\All_subject'
os.chdir(path)

ww=loadmat('pca_generalization_across_subject.mat')['pca_generalization_across_sub']
pca=np.zeros((3,160))
pca1=np.zeros((8,160))

for k in range(8):

    [a1,a2,mu]=plot_data_prep(ww[:,:,k])
    pca1[k,:]=mu
    
[a1,a2,mu]=plot_data_prep(pca1)
pca[0,:]=a1
pca[1,:]=a2
pca[2,:]=mu   


plt.figure(figsize=(4.8,3.5))

# w=get_sig_lags1( lags, srm_shared_space_generalization[:,2,:],original_regression[:,2,:])

# plt.fill_between(lags, shared_space_regression[0,:], shared_space_regression[1,:], alpha=0.2)
# plt.plot(lags, shared_space_regression[2,:], linewidth=3.5, label = "Shared space all data")  #, color=c1, ls=ls, lw = lw)


# plt.fill_between(lags, a2_original[0,:], a2_original[1,:], alpha=0.2)
# plt.plot(lags, a2_original[2,:], linewidth=3.5, label = "original encoding")   #, color=c1, ls=ls, lw = lw)

w=get_sig_lags1( lags, srm_shared_space_generalization[:,2,:],original_regression[:,2,:])

plt.fill_between(lags, a[0,:], a[1,:], alpha=0.2, color='darkorange')
plt.plot(lags, a[2,:], linewidth=3.5, label = "SRM generalization", color='darkorange')  #, color=c1, ls=ls, lw = lw)
plt.axvline(0, ls="dashed", alpha=0.3, c="k")


plt.fill_between(lags, pca[0,:], pca[1,:], alpha=0.2, color='green')
plt.plot(lags, pca[2,:], linewidth=3.5, label = "PCA generalization", color='green')  #, color=c1, ls=ls, lw = lw)

plt.scatter(w, # (x)
                np.full(len(w), 0.005), # (y)
                color='red')

name_t='Shared Space Generalization'

plt.legend( prop={'size':9})
plt.xlabel('lags (ms)')
plt.ylabel('Encoding Performance (r)')
plt.title(name_t)
plt.ylim([0,0.25])
plt.tight_layout()
filename=name_t+'.png'

path1='C:\Princeton\Research\Hyper_alignment\encoding\All_subject\plots\\'
os.chdir(path1)
plt.savefig(filename,dpi=600)

# plt.scatter(w, np.full(len(w), 0.001),color='red')



plt.figure(figsize=(4.8,3.5))

w=get_sig_lags1( lags, srm_leave_one_out[:,2,:],original_regression[:,2,:])

plt.fill_between(lags, a2_original[0,:], a2_original[1,:], alpha=0.2)
plt.plot(lags, a2_original[2,:], linewidth=3.5, label = "original encoding")   #, color=c1, ls=ls, lw = lw)

plt.fill_between(lags, a3[0,:], a3[1,:], alpha=0.2, color='darkorange')
plt.plot(lags, a3[2,:], linewidth=3.5, label = "SRM generalization", color='darkorange')  #, color=c1, ls=ls, lw = lw)
plt.axvline(0, ls="dashed", alpha=0.3, c="k")

plt.scatter(w, # (x)
                np.full(len(w), 0.005), # (y)
                color='red')


name='Electrode Space SRM Generalization'

plt.legend( prop={'size':9})
plt.xlabel('lags (ms)')
plt.ylabel('Encoding Performance (r)')
plt.title(name)
plt.ylim([0,0.25])
plt.tight_layout()
filename=name+'.png'

path1='C:\Princeton\Research\Hyper_alignment\encoding\All_subject\plots\\'
os.chdir(path1)
plt.savefig(filename,dpi=600)

q=[]
q1=[]
q2=[]
for i in range(len(subjects)):
    q1.append(max(no_aug_srm[i,2,:]))
    q2.append(max(original_regression[i,2,:]))
    
    q.append((max(no_aug_srm[i,2,:])-max(original_regression[i,2,:]))/max(original_regression[i,2,:]))
    
print((q2))



name='Shared Space Generalization_pca_generalization_added'

# plt.legend()
# plt.xlabel('lags (ms)')
# plt.ylabel('Correlation')
# plt.title(name)

filename=name+'.png'

path1='C:\Princeton\Research\Hyper_alignment\encoding\All_subject\plots\\'
os.chdir(path1)
plt.savefig(filename,dpi=600)

# color=['blue','green','red','crimson','brown','maroon','darkred','gray']
qq=3  

for qq in range(len(subjects)):
    
    a=corr_original[qq,:,:]
    b=corr_with_subjects_no_aug[:,:,qq]
    # # b=srm_leave_one_out1[:,:,qq]
    w=get_sig_lags1( lags, b,a)
    

    # plt.figure(figsize=(3.5, 2.7))
    
    
    plt.fill_between(lags, original_regression[qq,0,:], original_regression[qq,1,:], alpha=0.2)
    plt.plot(lags, original_regression[qq,2,:], linewidth=3.5, label = 'S'+str(qq+1)) #"original")  #, color=c1, ls=ls, lw = lw)
    
    # plt.fill_between(lags, augmented_regression_srm[qq,0,:], augmented_regression_srm[qq,1,:], alpha=0.2)
    # plt.plot(lags, augmented_regression_srm[qq,2,:], label = "augmented_srm")  #, color=c1, ls=ls, lw = lw)  
    
    # plt.fill_between(lags, no_aug_rsrm[qq,0,:], no_aug_rsrm[qq,1,:], alpha=0.2)
    # plt.plot(lags, no_aug_rsrm[qq,2,:], label = "rsrm")  #, color=c1, ls=ls, lw = lw)
    
    # plt.fill_between(lags, no_aug_srm[qq,0,:], no_aug_srm[qq,1,:], alpha=0.2)
    # plt.plot(lags, no_aug_srm[qq,2,:], linewidth=3.5,label = "SRM",color='darkorange',)  #, color=c1, ls=ls, lw = lw)
    
    
    # plt.fill_between(lags, original_content[qq,0,:], original_content[qq,1,:], alpha=0.2)
    # plt.plot(lags, original_content[qq,2,:], linewidth=3.5, label = "original")  #, color=c1, ls=ls, lw = lw)
    
    # plt.fill_between(lags, srm_content[qq,0,:], srm_content[qq,1,:], alpha=0.2)
    # plt.plot(lags, srm_content[qq,2,:], linewidth=3.5, label = "SRM")  #, color=c1, ls=ls, lw = lw)
    
    # plt.fill_between(lags, original_function[qq,0,:], original_function[qq,1,:], alpha=0.2)
    # plt.plot(lags, original_function[qq,2,:], linewidth=3.5, label = "original")  #, color=c1, ls=ls, lw = lw)
    
    # plt.fill_between(lags, srm_function[qq,0,:], srm_function[qq,1,:], alpha=0.2)
    # plt.plot(lags, srm_function[qq,2,:], linewidth=3.5, label = "SRM")  #, color=c1, ls=ls, lw = lw)
    
    # plt.fill_between(lags, srm_leave_one_out[qq,0,:], srm_leave_one_out[qq,1,:], alpha=0.2)
    # plt.plot(lags, srm_leave_one_out[qq,2,:], linewidth=3.5, label = "SRM-denoised")  #, color=c1, ls=ls, lw = lw)
    
    # plt.fill_between(lags, srm_shared_space_generalization[qq,0,:], srm_shared_space_generalization[qq,1,:], alpha=0.2)
    # plt.plot(lags, srm_shared_space_generalization[qq,2,:], linewidth=3.5, label = "SRM")  #, color=c1, ls=ls, lw = lw)
    
    ###bert
    # plt.fill_between(lags, bert_original_regression[qq,0,:], bert_original_regression[qq,1,:], alpha=0.2)
    # plt.plot(lags, bert_original_regression[qq,2,:], linewidth=3.5, label = "original")  #, color=c1, ls=ls, lw = lw)
    
    # plt.fill_between(lags, bert_no_aug_srm[qq,0,:], bert_no_aug_srm[qq,1,:], alpha=0.2)
    # plt.plot(lags, bert_no_aug_srm[qq,2,:], linewidth=3.5,label = "srm-denoised")  #, color=c1, ls=ls, lw = lw)
    
    
    # plt.scatter(w, # (x)
    #                 np.full(len(w), 0.001), # (y)
    #                 color='red')
    
    name2='Encoding_'+str(subjects[qq]) #+'('+str(elec_num[qq])+')'
    name='Encoding'+' '+'S'+str(qq+1)+'('+str(elec_num[qq])+')'
    
    
    
    
    plt.ylim(0, 0.32)
    plt.xlabel('lags (ms)')
    plt.ylabel('Encoding Performance (r)')
    plt.title('Encoding All Subjects')
       
    plt.tight_layout()
    
    # name='Encoding Content Words'+' '+str(subjects[qq])+' '+'('+str(elec_num[qq])+')'
    # plt.ylim(0, 0.45)
    plt.axvline(0, ls="dashed", alpha=0.3, c="grey")
    plt.legend()
    plt.ylim([0,0.4])
   
    
    
    # print(subjects[qq])
    # print(max(no_aug_srm[qq,2,:]))
    # print(max(no_aug_rsrm[qq,2,:]))
    
    filename=name2+'_new2.png'
    
    # filename='all_sub.png'
    
    path1='C:\Princeton\Research\Hyper_alignment\encoding\All_subject\\plots\\'
    os.chdir(path1)
    plt.savefig(filename,dpi=600)    
    

# [a1,a2,mu]=plot_data_prep(pca_regression1)
# pca_regression[0,:]=a1
# pca_regression[1,:]=a2
# pca_regression[2,:]=mu

# [a1,a2,mu]=plot_data_prep(shared_space_regression1)
# shared_space_regression[0,:]=a1
# shared_space_regression[1,:]=a2
# shared_space_regression[2,:]=mu

# plt.fill_between(lags, pca_regression[0,:], pca_regression[1,:], alpha=0.2)
# plt.plot(lags, pca_regression[2,:], linewidth=3.5, label = "PCA")  #, color=c1, ls=ls, lw = lw)

# plt.fill_between(lags, shared_space_regression[0,:], shared_space_regression[1,:], alpha=0.2)
# plt.plot(lags, shared_space_regression[2,:], linewidth=3.5, label = "SRM")  #, color=c1, ls=ls, lw = lw)

# # name='Shared Space Encoding vs PCA'
# name= 'Shared Space Encoding all'
# plt.legend()
# plt.xlabel('lags (ms)')
# plt.ylabel('Correlation')
# plt.title(name)

# filename=name+'.png'

# path1='C:\Princeton\Research\Hyper_alignment\encoding\All_subject\plots\\'
# os.chdir(path1)
# plt.savefig(filename,dpi=600)



    
#shuffling

a=original_regression[:,0,:]
mean_original_regression=np.max(a,axis=0)

a=shuffled_srm_regression[:,0,:]
shuffle_srm=np.max(a,axis=0)
a=no_aug_srm[:,0,:]
srm_denoise=np.max(a,axis=0)
plt.plot(lags, srm_denoise, linewidth=3.5, label = "SRM")  #, color=c1, ls=ls, lw = lw)
plt.plot(lags, shuffle_srm, linewidth=3.5, label = "Shuffle SRM")  #, color=c1, ls=ls, lw = lw)
plt.plot(lags, mean_original_regression, linewidth=3.5, label = "original")  #, color=c1, ls=ls, lw = lw)

plt.legend()
plt.xlabel('lags (ms)')
plt.ylabel('Correlation')
plt.title(name)

# area wise plotting
area='IFG'
area2='rSTG'
index=np.squeeze(np.where(df_all_sub['area2']==area2))

# index=np.squeeze(np.where(df_all_sub['area2']==area2))

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


name=area2+' '+'('+str(len(index))+')'

plt.axvline(0, ls="dashed", alpha=0.3, c="grey")
plt.legend()
plt.xlabel('lags (ms)')
plt.ylabel('Correlation')
plt.title(name)
plt.ylim([0,0.4])
plt.tight_layout()
filename=area2+'.png'

path1='C:\Princeton\Research\Hyper_alignment\encoding\All_subject\plots\\area_wise'
os.chdir(path1)
plt.savefig(filename,dpi=600)  

#area wise improvement plotting
#area_all=['IFG','STG','TP','AG','aMTG','parietal','premotor','MFG','pmtg','precentral']
area_all=['IFG','aSTG','mSTG','TP','cSTG']#,'AG','aMTG','parietal','premotor','MFG','pmtg','precentral']

index_lag=np.squeeze(np.where((lags>=-2000) & (lags<=2000)))

for k in range(len(area_all)):
    area=area_all[k]
    
    index=np.squeeze(np.where(df_all_sub['area']==area))
    
    if area=='mSTG':
        index=np.squeeze(np.where(df_all_sub['area2']==area))
    elif area=='cSTG':
        index=np.squeeze(np.where(df_all_sub['area2']==area))
    elif area=='aSTG':
            index=np.squeeze(np.where(df_all_sub['area2']=='rSTG'))
        
    

    original=original_all_elec[index,:]
    srm=srm_all_elec[index,:]

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
    
    plt.plot(lags[index_lag], srm_elec_area[2,:][index_lag]-original_elec_area[2,:][index_lag], linewidth=3.5,label=area)  #, color=c1, ls=ls, lw = lw)
    plt.legend()
    plt.xlabel('lags (ms)')
    plt.ylabel('Correlation')
    
    print([max(srm_elec_area[2,:]-original_elec_area[2,:]), lags[np.argmax(srm_elec_area[2,:]-original_elec_area[2,:])],area])


path1='C:\Princeton\Research\Hyper_alignment\encoding\All_subject\plots\\area_wise\\improvement'
os.chdir(path1)
filename='improvement.png'
plt.savefig(filename,dpi=600)  


# 247
filename='srm_247_selected_elec.mat'

path='C:\Princeton\Research\Hyper_alignment\\247'
os.chdir(path)

lags=np.squeeze(loadmat(filename)['lags'])

comp_org=np.squeeze(loadmat(filename)['comp_original']) 
comp_srm=np.squeeze(loadmat(filename)['comp_srm'])
prod_org=np.squeeze(loadmat(filename)['prod_original']) 
prod_srm=np.squeeze(loadmat(filename)['prod_srm'])

plt.figure
plt.plot(lags, comp_org, linewidth=1.5, label = "comp_org")  #, color=c1, ls=ls, lw = lw)
plt.plot(lags, comp_srm, linewidth=1.5, label = "comp_srm")  #, color=c1, ls=ls, lw = lw)
plt.legend()
plt.xlabel('lags (ms)')
plt.ylabel('Correlation')

plt.savefig('comp.png',dpi=600) 


plt.figure
plt.plot(lags, prod_org, linewidth=1.5, label = "prod_org")  #, color=c1, ls=ls, lw = lw)
plt.plot(lags, prod_srm, linewidth=1.5, label = "prod_srm")  #, color=c1, ls=ls, lw = lw)
plt.legend()
plt.xlabel('lags (ms)')
plt.ylabel('Correlation')
plt.savefig('prod.png',dpi=600) 

# relative improvement vs distance
filename='relative improvement.xlsx'

path='C:\Princeton\Research\Hyper_alignment\Podacast coordinates'
os.chdir(path)

df=pd.read_excel(filename, sheet_name='result')  
cm = plt.cm.get_cmap('RdYlBu')
plt.scatter(df['distance ifg'], df['distance stg'],  c=df['improvement'],cmap=cm,vmin=0.05, vmax=0.3,marker='o',)
plt.colorbar()