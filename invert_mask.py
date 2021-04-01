#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 22:33:26 2020

@author: wuzongze
"""


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1" #(or "1" or "2")
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import dnnlib.tflib as tflib
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import convolve2d
import time
from manipulate import convert_images_to_uint8
import argparse
from skimage.transform import resize

def LoadModel(dataset_name):
    # Initialize TensorFlow.
    tflib.init_tf()
    model_path='./model/'
#    model_name='stylegan2-'+dataset_name+'-config-f.pkl'
    model_name=dataset_name+'.pkl'
    
    tmp=os.path.join(model_path,model_name)
    with open(tmp, 'rb') as f:
        _, _, Gs = pickle.load(f)
    return Gs

from tensorflow.python.ops.gradient_checker import _compute_dx_and_dy,_compute_theoretical_jacobian
from tensorflow.python.framework import dtypes
def _compute_gradient2(x,
                      x_shape,
                      dx,
                      y,
                      y_shape,
                      dy,
                      x_init_value=None,
                      delta=1e-3,
                      extra_feed_dict=None):
  """Computes the theoretical and numerical jacobian."""
  t = dtypes.as_dtype(x.dtype)
  allowed_types = [dtypes.float16, dtypes.bfloat16, dtypes.float32,
                   dtypes.float64, dtypes.complex64, dtypes.complex128]
  assert t.base_dtype in allowed_types, "Don't support type %s for x" % t.name
  t2 = dtypes.as_dtype(y.dtype)
  assert t2.base_dtype in allowed_types, "Don't support type %s for y" % t2.name

  if x_init_value is not None:
    i_shape = list(x_init_value.shape)
    assert(list(x_shape) == i_shape), "x_shape = %s, init_data shape = %s" % (
        x_shape, i_shape)
    x_data = x_init_value
  else:
    x_data = np.random.random_sample(x_shape).astype(t.as_numpy_dtype)
    if t.is_complex:
      x_data.imag = np.random.random_sample(x_shape)

  jacob_t = _compute_theoretical_jacobian(
      x, x_shape, x_data, dy, y_shape, dx, extra_feed_dict=extra_feed_dict)
  return jacob_t

def _compute_gradient_list2(x,
                           x_shape,
                           y,
                           y_shape,
                           x_init_value=None,
                           delta=1e-3,
                           init_targets=None,
                           extra_feed_dict=None):
  """Compute gradients for a list of x values."""
  assert isinstance(x, list)
  dx, dy = zip(*[_compute_dx_and_dy(xi, y, y_shape) for xi in x])

  if init_targets is not None:
    assert isinstance(init_targets, (list, tuple))
    for init in init_targets:
      init.run()
  if x_init_value is None:
    x_init_value = [None] * len(x)
  ret = [_compute_gradient2(xi, x_shapei, dxi, y, y_shape, dyi, x_init_valuei,
                           delta, extra_feed_dict=extra_feed_dict)
         for xi, x_shapei, dxi, dyi, x_init_valuei in zip(x, x_shape, dx, dy,
                                                          x_init_value)]
  return ret

def compute_gradient2(x,
                     x_shape,
                     y,
                     y_shape,
                     x_init_value=None,
                     delta=1e-3,
                     init_targets=None,
                     extra_feed_dict=None):
  """Computes and returns the theoretical and numerical Jacobian.

  If `x` or `y` is complex, the Jacobian will still be real but the
  corresponding Jacobian dimension(s) will be twice as large.  This is required
  even if both input and output is complex since TensorFlow graphs are not
  necessarily holomorphic, and may have gradients not expressible as complex
  numbers.  For example, if `x` is complex with shape `[m]` and `y` is complex
  with shape `[n]`, each Jacobian `J` will have shape `[m * 2, n * 2]` with

      J[:m, :n] = d(Re y)/d(Re x)
      J[:m, n:] = d(Im y)/d(Re x)
      J[m:, :n] = d(Re y)/d(Im x)
      J[m:, n:] = d(Im y)/d(Im x)

  Args:
    x: a tensor or list of tensors
    x_shape: the dimensions of x as a tuple or an array of ints. If x is a list,
    then this is the list of shapes.
    y: a tensor
    y_shape: the dimensions of y as a tuple or an array of ints.
    x_init_value: (optional) a numpy array of the same shape as "x"
      representing the initial value of x. If x is a list, this should be a list
      of numpy arrays.  If this is none, the function will pick a random tensor
      as the initial value.
    delta: (optional) the amount of perturbation.
    init_targets: list of targets to run to initialize model params.
    extra_feed_dict: dict that allows fixing specified tensor values
      during the Jacobian calculation.

  Returns:
    Two 2-d numpy arrays representing the theoretical and numerical
    Jacobian for dy/dx. Each has "x_size" rows and "y_size" columns
    where "x_size" is the number of elements in x and "y_size" is the
    number of elements in y. If x is a list, returns a list of two numpy arrays.
  """
  # TODO(mrry): remove argument `init_targets`
  if extra_feed_dict is None:
    extra_feed_dict = {}

  if isinstance(x, list):
    return _compute_gradient_list2(x, x_shape, y, y_shape, x_init_value, delta,
                                  init_targets, extra_feed_dict=extra_feed_dict)
  else:
    if init_targets is not None:
      assert isinstance(init_targets, (list, tuple))
      for init in init_targets:
        init.run()
    dx, dy = _compute_dx_and_dy(x, y, y_shape)
    ret = _compute_gradient2(x, x_shape, dx, y, y_shape, dy, x_init_value, delta,
                            extra_feed_dict=extra_feed_dict)
    return ret

def ShowMask(img,jacob_t,lindex,cindex,out_size,img_size):
    g=jacob_t
    tmp=g[cindex,:].reshape((3,out_size,out_size))
#    plt.imshow(tmp.mean(axis=0))
    mask=tmp.mean(axis=0)
    mask-=mask.min()
    mask/=mask.max()
#    plt.imshow(mask)
    
    mask2 = resize(mask, (img_size,img_size),
                       anti_aliasing=True)
    
#    import cv2 as cv
    
    img2=convert_images_to_uint8(img, nchw_to_nhwc=True)[0]
#    img2=np.transpose(img, [0, 2, 3, 1])[0]
#    alpha=0.5
#    dst = cv.addWeighted(img2, alpha, mask2, 1-alpha, 0.0)    #%%
    import matplotlib.cm as cm
    from vis.visualization import overlay
    jet_heatmap = np.uint8(cm.jet(mask2)[..., :3] * 255)
#    jet_heatmap = np.uint8(cm.jet(mask2[:,:,None]) * 255)
    plt.figure()
#    plt.imshow(img2)
    plt.imshow(overlay(jet_heatmap, img2))
    plt.title(str(lindex)+'_'+str(cindex)+'_'+str(out_size))

#%%


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='predict pose of object')
    parser.add_argument('-img_sindex',default='0',type=str,help='path to model file') 
    parser.add_argument('-num_per',default='4',type=str,help='path to model file') 
    parser.add_argument('-dataset_name',default='ffhq',type=str,help='path to model file') 
    parser.add_argument('--remove_trgb', action='store_true')
    
    opt = parser.parse_args()
    
    
    
    #%%
    dataset_name=opt.dataset_name
    remove_trgb=opt.remove_trgb
    Gs=LoadModel(dataset_name)
    
    img_size=Gs.output_shape[-1]
    out_size=32
    
    batch_size=1
    pool_size=int(img_size/out_size)
    Gs._get_vars()
    
    
#    model_name='stylegan2-ffhq-config-f.pkl'
    tmp='./npy/'+dataset_name+'/'
    with open(tmp+'S_100K', "rb") as fp:   #Pickling
        s_names,all_s=pickle.load( fp)
    
    
    dlatents=np.load(tmp+'W.npy')[:,None,:] #[:,0]
    dlatents=np.tile(dlatents,(1,Gs.components.synthesis.input_shape[1],1))
    

    layer_num=Gs.components.synthesis.input_shape[1]
#    layer_num=int((len(s_names)+1)/3*2)
    print('layer_num: ', str(layer_num))
    
    
    #%%
    s_names3=[]
#    all_s2=[]
    for i in range(len(s_names)):
        sname=s_names[i]
        sname1=sname.split('/')[1:]
#        if remove_trgb and sname1[1]=='ToRGB':
#            continue
#        all_s2.append(all_s[i])
        sname2=['G_synthesis_2']+sname1
        sname3='/'.join(sname2)
        s_names3.append(sname3)
    
    s_names2=[]
    for tmp in s_names3:
        if not 'ToRGB' in tmp:
            s_names2.append(tmp)
    #%%
    
    model_input=tf.placeholder(dtype='float32',shape=(1,layer_num,512),name='model_input')
    
    model_output=Gs.components.synthesis.get_output_for(model_input)
    
    model_output2=tf.keras.layers.AveragePooling2D(pool_size=pool_size, strides=pool_size,
                                                   padding='valid',data_format='channels_first')(model_output)
    print(model_output2.shape)
    assert(model_output2.shape[-1]==out_size)
    
    watch_s=[]
    for i in range(len(s_names2)):
         tmp= tf.get_default_graph().get_tensor_by_name(s_names2[i]) #[:,0,0,channel_index,0]
         watch_s.append(tmp)
    
    watch_s_size=[]
    for t in watch_s:
        tmp=t.get_shape().as_list()
        watch_s_size.append(tmp)

    #%%
    
    all_var_grad=[[] for i in range(len(s_names2))]
    for m in range(int(opt.num_per)):
        print(m)
        img_index=int(opt.img_sindex)+m
        
        d={}
        d['model_input:0']=dlatents[img_index:(img_index+batch_size),:,:] #[img_index:(img_index+batch_size),0,:,:]
        for i in range(len(s_names2)):
            d[s_names3[i]]=all_s[i][img_index:(img_index+batch_size)]
        d['G_synthesis_2/4x4/Const/Shape:0']=np.array([batch_size,18,  512], dtype=np.int32)
        
        img= Gs.components.synthesis.run(d['model_input:0']) 
        d['G_synthesis_2/images_out:0']=img
        
        watch_s_value=[]
        for i in range(len(s_names2)):
             tmp= d[s_names2[i]] #[:,0,0,channel_index,0]
             watch_s_value.append(tmp)
        
        var_grad=compute_gradient2(x=watch_s,x_shape=watch_s_size,y=model_output2,y_shape=(1,3,out_size,out_size),
                                          x_init_value=watch_s_value,extra_feed_dict=d)
        for k in range(len(s_names2)):
            all_var_grad[k].append(var_grad[k])
        
    
    for k in range(len(s_names2)):
        all_var_grad[k]=np.array(all_var_grad[k])
    
    
    #%%
    
    
    
    save_path='./npy/'+dataset_name+'/gradient_mask_32/'
    with open(save_path+opt.img_sindex, 'wb') as handle:
        pickle.dump(all_var_grad, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        