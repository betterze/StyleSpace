#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 13:51:06 2020

@author: wuzongze
"""

import pickle 
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import time
import argparse

def ExpendSMask(semantic_masks,num_semantic):
    
    semantic_masks2=[]
    for i in range(num_semantic):
        tmp=semantic_masks==i
        semantic_masks2.append(tmp)
    semantic_masks2=np.array(semantic_masks2)
    semantic_masks2=np.transpose(semantic_masks2, [1,0,2,3])
    return semantic_masks2
    

#def OverlapScore(mask2,tmp_mask):
#    if tmp_mask.sum()==0:
#        return np.nan,np.nan
#    tmp=(mask2*tmp_mask).sum()/tmp_mask.sum()
#    tmp1=(mask2*(1-tmp_mask)).sum()/(1-tmp_mask).sum()
#    
##    tmp2=(tmp-tmp1)/tmp1
#    return tmp,tmp1
    
def OverlapScore(mask2,tmp_mask):
    o=tmp_mask.sum() #size of semantic mask
    if o==0:
        return np.nan,np.nan,np.nan
    
    p=o/(mask2.shape[0]*mask2.shape[1])
    
    threshold=np.percentile(mask2.reshape(-1),(1-p)*100)
    gmask=mask2>threshold
    
    n=np.sum(np.logical_and(gmask,tmp_mask))
    u=np.sum(np.logical_or(gmask,tmp_mask))
    
    return n,u,o
    
    
def GetScore(mask2,semantic_mask2):
#    scores=np.zeros(len(semantic_mask2))
    scores=[]
    for i in range(len(semantic_mask2)):
        tmp_mask=semantic_mask2[i]
        n,u,o=OverlapScore(mask2,tmp_mask)
        scores.append([n,u,o])
    scores=np.array(scores)
    return scores


def CFFHQ(semantic_masks):
    
#    mapping=[[0],[1],[2,3],[4,5],[6],[7,8],[9],[10],[11,12,13],[14],[15],[16],[17],[18]]
    mapping=[[0],[1],[2,3],[4,5],[7,8],[10],[11,12,13],[14],[16],[17]]
    semantic_masks2=np.zeros(semantic_masks.shape)
    
    for i in range(len(mapping)):
        
        for k in mapping[i]:
            select=semantic_masks==k
            semantic_masks2[select]=i+1
    semantic_masks2=semantic_masks2.astype('uint8')
    return semantic_masks2







#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='predict pose of object')
    parser.add_argument('-img_sindex',default='0',type=str,help='path to model file') 
    parser.add_argument('-num_per',default='4',type=str,help='path to model file') 
    
    opt = parser.parse_args()
    
    #%%
    out_size=32
    
    dataset_name='dog'
    
#    if dataset_name=='ffhq':
#        num_semantic=19
#    elif dataset_name=='bed':
#        num_semantic=21
    
    
    img_path='/cs/labs/danix/wuzongze/Gan_Manipulation/stylegan2/results/npy/'+dataset_name+'/'
#    img_path2='/mnt/local/wuzongze/gradient_map/'+dataset_name+'/'
    with open(img_path+'gradient_mask_32/'+opt.img_sindex, 'rb') as handle:
        var_grad = pickle.load(handle)
    
    semantic_masks=np.load(img_path+'semantic_mask.npy') #car_c
    if dataset_name=='ffhq':
        semantic_masks=CFFHQ(semantic_masks)
    
    
    num_semantic=int(semantic_masks.max()+1)
    semantic_masks=semantic_masks[int(opt.img_sindex):(int(opt.img_sindex)+int(opt.num_per))]
    semantic_masks2=ExpendSMask(semantic_masks,num_semantic)
    
    mask_size=semantic_masks2.shape[-1]
    step=int(mask_size/out_size)
    
    
    semantic_masks2=semantic_masks2.reshape(int(opt.num_per),num_semantic,out_size,step,out_size,step)
    
    semantic_masks2=np.sum(semantic_masks2,axis=(3,5))
    semantic_masks2_single=np.argmax(semantic_masks2,axis=1)
    
    semantic_masks2=ExpendSMask(semantic_masks2_single,num_semantic)
    
#    mask_counts=semantic_masks2.sum(axis=(2,3))>0
#    mask_counts.sum(axis=0)
    
#    _,_,mask_size=semantic_masks.shape
    #%%
    '''
    from vis.visualization import overlay
    from PIL import Image
    imgs=np.load(img_path+'images.npy')
    img_index=1
    img=imgs[img_index]
    semantic_mask=semantic_masks2_single[img_index]
    step2=int(img.shape[0]/out_size)
    mask=np.zeros(img.shape,dtype='uint8')
    
    d={}
    d[0]=[255,204,204]
    d[1]=[153,255,255]
    d[2]=[255,255,51]
    d[3]=[51,255,51]
    d[4]=[51,102,0]
    d[5]=[0,255,255]
    d[6]=[51,102,0]
    d[7]=[0,0,255]
    d[8]=[255,0,0]
    d[9]=[255,255,51]
    d[10]=[51,102,0]
    d[11]=[51,255,255]
    d[12]=[255,102,255]
    d[13]=[51,102,0]
    
    for i in range(out_size):
        for j in range(out_size):
            tmp=semantic_mask[i,j]
            mask[(i*step2):((i+1)*step2),(j*step2):((j+1)*step2)]=d[tmp]
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.imshow(overlay(mask, img))
    plt.axis('off')
    plt.savefig("/cs/labs/danix/wuzongze/downloads/semantic.png", bbox_inches = 'tight',
        pad_inches = 0)
    
    lindex,cindex=8,28
    tmp=var_grad[lindex][img_index,cindex,:].reshape((3,out_size,out_size))
    tmp2=np.abs(tmp).mean(axis=0)
#    plt.imshow(tmp2)
#    tmp2=np.log(tmp2+1)
    tmp2=tmp2/tmp2.max()
#    tmp2= np.moveaxis(tmp2, 0, -1)
    tmp2=(tmp2*255).astype('uint8')
    tmp2=Image.fromarray(tmp2)
    tmp2=tmp2.resize((img.shape[:-1]))
    tmp2=np.array(tmp2)
    tmp3=np.zeros(img.shape,dtype='uint8')
    tmp3[:,:,1]=tmp2
#    tmp2=np.tile(tmp2[:,:,None],(1,1,3))
    
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.imshow(overlay(tmp3, img))
    plt.axis('off')
    plt.savefig("/cs/labs/danix/wuzongze/downloads/"+str(lindex)+'_'+str(cindex)+".png", bbox_inches = 'tight',
        pad_inches = 0)
    '''
    #%%
    '''
    save_path='/cs/labs/danix/wuzongze/result/1112/'
    img_index=1
    semantic_index=7 #7,-1
    lindex,cindex=11,286 #6,202  11,286
    
    tmp_mask=semantic_masks2[img_index,semantic_index]  #semantic mask
    mask=var_grad[lindex][img_index,cindex].reshape((3,out_size,out_size))
    mask2=np.abs(mask).mean(axis=0)
    
    
    
    o=tmp_mask.sum() #size of semantic mask
    
    p=o/(mask2.shape[0]*mask2.shape[1])
    
    threshold=np.percentile(mask2.reshape(-1),(1-p)*100)
    gmask=mask2>threshold
    
    
    plt.figure()
    plt.imshow(mask2)
    plt.axis('off')
    
    
    plt.figure()
    plt.imshow(gmask)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.subplots_adjust(top=1,bottom=00,left=0,right=1)
    plt.savefig(save_path+str(lindex)+'_'+str(cindex)+'_'+str(semantic_index)+'_m', bbox_inches='tight',pad_inches = 0)
    
    plt.figure()
    plt.imshow(tmp_mask)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.subplots_adjust(top=1,bottom=00,left=0,right=1)
    plt.savefig(save_path+str(lindex)+'_'+str(cindex)+'_'+str(semantic_index)+'_s', bbox_inches='tight',pad_inches = 0)
    
#    plt.tight_layout()
    plt.close('all')
    '''
    #%%
#    linex,cindex=11,286
#    linex,cindex=6,501
#    start = time.time()
    all_scores=[]
    for linex in range(len(var_grad)):
#        print(linex)
        layer_g=var_grad[linex]
        num_img,num_channel,_=layer_g.shape
        
        scores2=np.zeros((num_img,num_channel,num_semantic,3))
#        start = time.time()
        for img_index in range(num_img):
            print(linex,img_index)
#            start = time.time()
            semantic_mask2=semantic_masks2[img_index]
            for cindex in range(num_channel):
                mask=layer_g[img_index,cindex].reshape((3,out_size,out_size))
                mask2=np.abs(mask).mean(axis=0)  #need code 
                
#                mask2 = resize(mask, (mask_size,mask_size),anti_aliasing=True)
#                plt.imshow(mask2)
                scores=GetScore(mask2,semantic_mask2)
                scores2[img_index,cindex,:,:]=scores
        all_scores.append(scores2)
#        end = time.time()
#        print(end-start)
#        with open(img_path+'align_scores_300', 'wb') as handle:
#            pickle.dump(all_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #%%
    with open(img_path+'align_score_32/'+opt.img_sindex, 'wb') as handle:
        pickle.dump(all_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    
#    mask=var_grad[11][1,286].reshape((3,out_size,out_size)).mean(axis=0)
#    plt.imshow(mask)
    
    #%%
    
    
    
    
#    tmp=mask.reshape(-1)
#    tmp.sort()
#    
#    threshold=np.percentile(mask,90)
#    
#    mask3=mask>threshold
    
    
    
    
    
    
    