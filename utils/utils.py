#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 22:20:38 2021

@author: spathak
"""

from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torchvision import transforms
import pandas as pd
import torch
import math
from PIL import Image
import numpy as np
import sys
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics, utils
from collections import Counter
from sklearn.model_selection import GroupShuffleSplit

groundtruth_dic={'benign':0,'malignant':1}
inverted_groundtruth_dic={0:'benign',1:'malignant'}
views_allowed=['LCC','LMLO','RCC','RMLO']

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
cluster_data_path_prefix='/' #your image path

class MyCrop:
    """Randomly crop the sides."""

    def __init__(self, left=100,right=100,top=100,bottom=100):
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom

    def __call__(self, x):
        width, height=x.size
        size_left = random.randint(0,self.left)
        size_right = random.randint(width-self.right,width)
        size_top = random.randint(0,self.top)
        size_bottom = random.randint(height-self.bottom,height)
        x = TF.crop(x,size_top,size_left,size_bottom,size_right)
        return x
    
class MyGammaCorrection:
    def __init__(self, factor=0.2):
        self.lb = 1-factor
        self.ub = 1+factor

    def __call__(self, x):
        gamma = random.uniform(self.lb,self.ub)
        return TF.adjust_gamma(x,gamma)

class MyHorizontalFlip:
    """Flip horizontally."""

    def __init__(self):
        pass

    def __call__(self, x, breast_side):
        if breast_side=='L':
            return TF.hflip(x)
        else:
            return x

class MyPadding:
    def __init__(self, breast_side, max_height, max_width, height, width):
        self.breast_side = breast_side
        self.max_height=max_height
        self.max_width=max_width
        self.height=height
        self.width=width
          
    def __call__(self,img):
        print(img.shape)
        print(self.max_height-self.height)
        if self.breast_side=='L':
            image_padded=F.pad(img,(0,self.max_width-self.width,0,self.max_height-self.height,0,0),'constant',0)
        elif self.breast_side=='R':
            image_padded=F.pad(img,(self.max_width-self.width,0,0,self.max_height-self.height,0,0),'constant',0)
        print(image_padded.shape)
        return image_padded

class MyPaddingLongerSide:
    def __init__(self):
        self.max_height=1600
        self.max_width=1600
        
        
    def __call__(self,img):#,breast_side):
        width=img.size[0]
        height=img.size[1]
        if height<self.max_height:
            diff=self.max_height-height
            img=TF.pad(img,(0,math.floor(diff/2),0,math.ceil(diff/2)),0,'constant')
        if width<self.max_width:
            diff=self.max_width-width
            #if breast_side=='L':
            #    img=TF.pad(img,(0,0,diff,0),0,'constant')
            #elif breast_side=='R':
            img=TF.pad(img,(diff,0,0,0),0,'constant')
        return img
    
class BreastCancerDataset_generator(Dataset): #changed this
    """Face Landmarks dataset."""

    def __init__(self, df, modality, flipimage, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = df
        self.modality = modality
        self.transform = transform
        self.flipimage = flipimage
        self.hflip_img = MyHorizontalFlip()

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        data=self.df.iloc[idx]
        img, breast_side=collect_images(data)
        if self.flipimage:
            img=self.hflip_img(img,breast_side)
        if self.transform:
            img=self.transform(img)
        #print("after transformation:",img.shape)
        img=img[0,:,:]
        img=img.unsqueeze(0).unsqueeze(1)
        # img=img[0,:,:].unsqueeze(0)
        return idx, img, torch.tensor(groundtruth_dic[data['Groundtruth']])

def MyCollate(batch):
    i=0
    index=[]
    target=[]
    for item in batch:
        if i==0:
            data=batch[i][1]
        else:
            data=torch.cat((data,batch[i][1]),dim=0)
        index.append(item[0])
        target.append(item[2])
        i+=1
    index = torch.LongTensor(index)
    target = torch.LongTensor(target)
    return [index, data, target]#, views_names

def collect_images(data): #changed this
    #collect images for the model
    if data['Views'] in views_allowed:
        img_path = cluster_data_path_prefix + str(data['FullPath'])
        img = Image.open(img_path)
        return img, data['Views'][0]
    else:
        print('error in view')
        sys.exit()

def data_augmentation_train(mean,std_dev):
    preprocess_train = transforms.Compose([
        MyCrop(),
        transforms.Pad(100),
        transforms.RandomRotation(3),
        transforms.ColorJitter(brightness=0.20, contrast=0.20),
        transforms.RandomAdjustSharpness(sharpness_factor=0.20),
        MyGammaCorrection(0.20),
        MyPaddingLongerSide(),
        transforms.Resize((1600,1600)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std_dev)
    ])
    return preprocess_train

def fetch_groundtruth(df,acc_num,modality):
    col_names=df.filter(regex='Acc_'+modality+'.*').columns.tolist()
    acc_num=int(acc_num)
    groundtruth=df.loc[(df[col_names].astype('Int64')==acc_num).any(axis=1)]['final_gt']
    if groundtruth.empty:
        groundtruth=-1
    else:
        groundtruth=groundtruth.item()
    return groundtruth

def freeze_layers(model, layer_keyword):
    for name,param in model.named_parameters():
        #print(name)
        if layer_keyword in name:
            param.requires_grad=False
            #print(name,param.requires_grad)
    return model

def plot(filename):
    df=pd.read_excel(filename).sort_values(by=['Count'],ascending=False)
    print(df['Views'].tolist())
    print(df['Count'].tolist())
    plt.figure(figsize=(5,5))
    plt.bar(df['Views'].tolist(),df['Count'].tolist())
    plt.xticks(rotation=45,ha='right')
    plt.savefig('view_distribution.png', bbox_inches='tight')    

def stratified_class_count(df):
    class_count=df.groupby(by=['Groundtruth']).size()
    return class_count

def class_distribution_weightedloss(df):
    df_groundtruth=df['Groundtruth'].map(groundtruth_dic)
    class_weight=utils.class_weight.compute_class_weight(class_weight = 'balanced', classes = np.array([0,1]), y = df_groundtruth)
    print(dict(Counter(df_groundtruth)))
    print(class_weight)
    return torch.tensor(class_weight,dtype=torch.float32).to(device)

def class_distribution_poswt(df): #added this
    class_count=df.groupby(by=['Groundtruth']).size()
    pos_wt=torch.tensor([float(class_count['benign'])/class_count['malignant']])
    print(pos_wt)
    return pos_wt

def stratifiedgroupsplit(df, rand_seed):
    groups = df.groupby('Groundtruth')
    all_train = []
    all_test = []
    all_val = []
    train_testsplit = GroupShuffleSplit(test_size=0.15, n_splits=2, random_state=rand_seed)
    train_valsplit = GroupShuffleSplit(test_size=0.10, n_splits=2, random_state=rand_seed)
    for group_id, group in groups:
        # if a group is already taken in test or train it must stay there
        group = group[~group['Patient_Id'].isin(all_train+all_val+all_test)]
        # if group is empty 
        if group.shape[0] == 0:
            continue
        train_inds1, test_inds = next(train_testsplit.split(group, groups=group['Patient_Id']))
        train_inds, val_inds = next(train_valsplit.split(group.iloc[train_inds1], groups=group.iloc[train_inds1]['Patient_Id']))
    
        all_train += group.iloc[train_inds1].iloc[train_inds]['Patient_Id'].tolist()
        all_val += group.iloc[train_inds1].iloc[val_inds]['Patient_Id'].tolist()
        all_test += group.iloc[test_inds]['Patient_Id'].tolist()
        
    train = df[df['Patient_Id'].isin(all_train)]
    val = df[df['Patient_Id'].isin(all_val)]
    test = df[df['Patient_Id'].isin(all_test)]
    return train, val, test

def performance_metrics(conf_mat,y_true,y_pred,y_prob):
    prec=metrics.precision_score(y_true,y_pred,pos_label=1)
    rec=metrics.recall_score(y_true,y_pred) #sensitivity, TPR
    spec=conf_mat[0,0]/np.sum(conf_mat[0,:]) #TNR
    f1=metrics.f1_score(y_true,y_pred)
    acc=metrics.accuracy_score(y_true,y_pred)
    bal_acc=(rec+spec)/2
    cohen_kappa=metrics.cohen_kappa_score(y_true,y_pred)
    auc=metrics.roc_auc_score(y_true,y_prob)
    each_model_metrics=[prec,rec,spec,f1,acc,bal_acc,cohen_kappa,auc]
    return each_model_metrics
    
def confusion_matrix_norm_func(conf_mat,fig_name,class_name):
    #class_name=['W','N1','N2','N3','REM']
    conf_mat_norm=np.empty((conf_mat.shape[0],conf_mat.shape[1]))
    #conf_mat=confusion_matrix(y_true, y_pred)
    for i in range(conf_mat.shape[0]):
        conf_mat_norm[i,:]=conf_mat[i,:]/sum(conf_mat[i,:])
    #print(conf_mat_norm)
    print_confusion_matrix(conf_mat_norm,class_name,fig_name)
    
def print_confusion_matrix(conf_mat_norm, class_names, fig_name, figsize = (2,2), fontsize=5):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    #sns.set()
    #grid_kws = {"height_ratios": (.9, .05), "hspace": .3}
    fig, ax = plt.subplots(figsize=figsize)
    #cbar_ax = fig.add_axes([.93, 0.1, 0.05, 0.77])
    #fig = plt.figure(figsize=figsize)
    heatmap=sns.heatmap(
        yticklabels=class_names,
        xticklabels=class_names,
        data=conf_mat_norm,
        ax=ax,
        cmap='YlGnBu',
        cbar=False,
        #cbar_ax=cbar_ax,
        annot=True,
        annot_kws={'size':fontsize},
        fmt=".2f",
        square=True
        #linewidths=0.75
        )
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    ax.set_ylabel('True label',labelpad=0,fontsize=fontsize)
    ax.set_xlabel('Predicted label',labelpad=0,fontsize=fontsize)
    #cbar_ax.tick_params(labelsize=fontsize) 
    #ax.get_yaxis().set_visible(False)
    #plt.tight_layout()
    #plt.show()
    ax.set_title(fig_name)
    fig.savefig(fig_name+'.pdf', format='pdf', bbox_inches='tight')    

#conf_mat=np.array([[775,52],[170,166]])
#confusion_matrix_norm_func(conf_mat,'sota_MaxWelling_variableview_MG',class_name=['benign','malignant'])