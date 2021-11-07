#!/usr/bin/python
# -*- encoding: utf-8 -*-

#import sys 
#from logger import setup_logger
from model import BiSeNet

import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
from skimage.transform import resize
import argparse

def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.6, vis_parsing_anno_color, 0.4, 0)

    # Save result or not
    if save_im:
        cv2.imwrite(save_path[:-4] +'.png', vis_parsing_anno)
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    return vis_im

def evaluate(respth='./res/test_res', dspth='./data', cp='model_final_diss.pth'):

    if not os.path.exists(respth):
        os.makedirs(respth)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = osp.join('res/cp', cp)
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        for image_path in os.listdir(dspth):
            img = Image.open(osp.join(dspth, image_path))
            image = img.resize((512, 512), Image.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            # print(parsing)
            print(np.unique(parsing))

            vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=osp.join(respth, image_path))

def ResizeA(parsing):
    parsing2=[]
    for i in range(len(parsing)):
        tmp=resize(parsing[i],(16,16),
                       anti_aliasing=True)
        parsing2.append(tmp)
    parsing2=np.array(parsing2)
    return parsing2


def LoadModel(model_path):
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
#    save_pth = osp.join('./res/cp', model_name)
    net.load_state_dict(torch.load(model_path))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return net,to_tensor

def Inference(imgs,net,to_tensor):
    all_parsing=[]
    with torch.no_grad():
        for i in range(len(imgs)):
            if i %1000==0:
                print(i)
            img=imgs[i]
            img=Image.fromarray(img)
            image=img
#            image = img.resize((512, 512), Image.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().detach().numpy()
            parsing=parsing.argmax(0)
            
#            parsing2=ResizeA(parsing)
#            parsing=parsing2.argmax(0)
            
             # print(parsing)
    #        print(np.unique(parsing))
#            image = image.resize((16, 16), Image.BILINEAR)
#            vis_im=vis_parsing_maps(image, parsing, stride=1, save_im=False)
#            Image.fromarray(vis_im)
#            plt.imshow(vis_im)
#            plt.imshow(parsing)
            parsing=parsing.astype('uint8')
            all_parsing.append(parsing)
    all_parsing=np.array(all_parsing).astype('uint8')
    return all_parsing

def CFFHQ(semantic_masks): 
    ''' combine small regions to big region '''
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
    
    parser = argparse.ArgumentParser(description='Semantic segmentation for natural human face')
    parser.add_argument('-model_path',type=str,default='./79999_iter.pth',
                    help='path to model')
    parser.add_argument('-img_path',type=str,default='../npy/ffhq/images.npy',
                    help='path to image npy')
    parser.add_argument('-save_path',type=str,default='../npy/ffhq/semantic_mask.npy',
                    help='path to save semantic segmentation')
    args = parser.parse_args()
    
    #%%
    
    net,to_tensor=LoadModel(args.model_path)
    imgs=np.load(args.img_path)
    #%%
    all_parsing=Inference(imgs,net,to_tensor)
    
    semantic_masks2=CFFHQ(all_parsing)
    
    np.save(args.save_path,semantic_masks2)
    #%%
    
#    i=10
#    plt.figure()
#    plt.subplot(1,2,1)
#    plt.imshow(imgs[i])
#    plt.title('original')
#    plt.subplot(1,2,2)
#    plt.imshow(semantic_masks2[i])
#    plt.title('BiSeNet')
    
    


