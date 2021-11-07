#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 21:33:42 2020

@author: wuzongze
"""

import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1" #(or "1" or "2")
import pickle
import dnnlib.tflib as tflib
import numpy as np
from PIL import Image
import pandas as pd

import matplotlib.pyplot as plt
import argparse

def convert_images_from_uint8(images, drange=[-1,1], nhwc_to_nchw=False):
    """Convert a minibatch of images from uint8 to float32 with configurable dynamic range.
    Can be used as an input transformation for Network.run().
    """
    if nhwc_to_nchw:
        imgs_roll=np.rollaxis(images, 3, 1)
    return imgs_roll/ 255 *(drange[1] - drange[0])+ drange[0]

#%%

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='predict pose of object')
    parser.add_argument('-img_path',type=str,help='path to image folder')
    parser.add_argument('-save_path',type=str,help='path to save attribute file') 
    parser.add_argument('-classifer_path',default='./metrics_checkpoint',type=str,help='path to a folder of classifers') 
    parser.add_argument('-batch_size',default=5,type=int,help='batch size') 
    
    opt = parser.parse_args()
    
    img_path=opt.img_path
    save_path=opt.save_path
    classifer_path=opt.classifer_path
    batch_size=opt.batch_size
    
    #%%
    
    imgs=np.load(img_path)
    names_tmp=os.listdir(classifer_path)
    names=[]
    for name in names_tmp:
        if 'celebahq-classifier' in name:
            names.append(name)
    names.sort()
    
    tflib.init_tf()
    results={}
    for name in names:
        print(name)
        tmp=os.path.join(classifer_path,name)
        with open(tmp, 'rb') as f:
                classifier = pickle.load(f)
        
        logits=np.zeros(len(imgs))
        for i in range(int(imgs.shape[0]/batch_size)):
            if i%(100)==0:
                print(i/100)
            tmp_imgs=imgs[(i*batch_size):((i+1)*batch_size)]
            tmp_imgs=convert_images_from_uint8(tmp_imgs, drange=[-1,1], nhwc_to_nchw=True)
            tmp = classifier.run(tmp_imgs, None)
            
            tmp1=tmp.reshape(-1) 
            logits[(i*batch_size):((i+1)*batch_size)]=tmp1
        
        tmp1=name[20:-4]
        results[tmp1]=logits
        
        results2=pd.DataFrame(results)
        results2.to_csv(save_path,index=False)
    
    
    
    
    
    
    
    
    
    
