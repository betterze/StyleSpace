#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 18:51:36 2020|

@author: wuzongze
"""

import pickle 
import numpy as np
import pandas as pd
import argparse
import os 

def LoadAMask(opt):
    for i in range(0,1000,int(opt.num_per)):
        try:
            tmp=os.path.join(opt.align_folder,str(i))
            with open(tmp, 'rb') as handle:
                var_grad = pickle.load(handle)
            
            if not 'all_var_grad' in locals():
                num_layer=len(var_grad)
                all_var_grad=[[] for i in range(num_layer)]
            
            for k in range(num_layer):
                all_var_grad[k].append(var_grad[k])
        except FileNotFoundError:
            print(i)
            continue
            
    for i in range(num_layer):
        all_var_grad[i]=np.concatenate(all_var_grad[i])
    print('num of sample:',all_var_grad[0].shape[0])
    return all_var_grad
    

def TopRate(all_var_grad):
    num_layer=len(all_var_grad)
    num_semantic=all_var_grad[0].shape[2]
    discount_factor=2 #large number means pay higher weight precision (prefer small area) 
    all_count_top=[]
    for lindex in range(num_layer):
        layer_g=all_var_grad[lindex]
        num_channel=layer_g.shape[1]
        count_top=np.zeros([num_channel,num_semantic])
        for cindex in range(num_channel):
            semantic_in=layer_g[:,cindex,:,0]/(layer_g[:,cindex,:,2]**discount_factor)
            semantic_top=np.nanargmax(np.abs(semantic_in),axis=1)
            
            semantic_top=pd.Series(semantic_top)
            tmp=semantic_top.value_counts()
            count_top[cindex,tmp.index]=tmp.values
        all_count_top.append(count_top)
    
    tmp=all_var_grad[0][:,0,:,2]
    mask_counts2=~np.isnan(tmp)
    mask_counts3=mask_counts2.sum(axis=0)
    mask_counts3[mask_counts3==0]=1 # ignore 0
    
    all_count_top2=[]
    for lindex in range(len(all_count_top)):
        all_count_top2.append(all_count_top[lindex]/mask_counts3)
    return all_count_top2

def PadTRGB(opt,all_count_top2):
    with open(opt.s_path, "rb") as fp:   #Pickling
        s_names,all_s=pickle.load( fp)
    
    tmp_index=0
    all_count_top3=[[] for i in range(len(s_names))]
    num_sa=all_count_top2[0].shape[1]
    
    
    for i in range(len(s_names)):
        s_name=s_names[i]
        if 'ToRGB' in s_name:
            tmp=np.zeros([all_s[i].shape[1],num_sa])
        else:
            tmp=all_count_top2[tmp_index]
            tmp_index+=1
        all_count_top3[i]=tmp
    
    all_count_top2=all_count_top3
    return all_count_top2

#%%
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='predict pose of object')
    
    parser.add_argument('-align_folder',default='./npy/ffhq/align_mask_32',type=str,help='path to align_mask_32 folder') 
    parser.add_argument('-s_path',default='./npy/ffhq/S',type=str,help='path to ') 
    parser.add_argument('-save_folder',default='./npy/ffhq/',type=str,help='path to save folder') 
    
    parser.add_argument('-num_per',default='4',type=str,help='path to model file') 
    parser.add_argument('-include_trgb', action='store_true')
    
    opt = parser.parse_args()
    
    #%%
    all_var_grad=LoadAMask(opt)
    all_count_top2=TopRate(all_var_grad)
    if not opt.include_trgb:
        all_count_top2=PadTRGB(opt,all_count_top2)
    #%%
    tmp=os.path.join(opt.save_folder,'semantic_top_32')
    with open(tmp, 'wb') as handle:
        pickle.dump(all_count_top2, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        
    #%%
    
    
    
    
    
    
    
    
    
    
    
    
    
    