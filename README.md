# StyleSpace Analysis: Disentangled Controls for StyleGAN Image Generation 

Demo video: <a href="https://youtu.be/U7qRotRGr1w"><img src="https://img.shields.io/badge/-YouTube-red?&style=for-the-badge&logo=youtube&logoColor=white" height=20></a>
CVPR 2021 Oral: <a href="https://arxiv.org/abs/2011.12799"><img src="https://upload.wikimedia.org/wikipedia/commons/a/a8/ArXiv_web.svg" height=20></a>

Single Channel Manipulation: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/betterze/StyleSpace/blob/main/StyleSpace_single.ipynb)
Localized or attribute specific Manipulation: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/betterze/StyleSpace/blob/main/StyleSpace_advance.ipynb)


<p align="center">
  <a href="https://youtu.be/U7qRotRGr1w"><img src='https://github.com/betterze/StyleSpace/blob/main/imgs/stylespace_short.gif' width=600 ></a>
</p>

![](imgs/disentanglement.png)

> **StyleSpace Analysis: Disentangled Controls for StyleGAN Image Generation**<br>
Zongze Wu, Dani Lischinski, Eli Shechtman <br>
> [paper](https://arxiv.org/abs/2011.12799) (CVPR 2021 Oral) 
> [video](https://youtu.be/U7qRotRGr1w)
>
>**Abstract:** We explore and analyze the latent style space of StyleGAN2, a state-of-the-art architecture for image generation, using models pretrained on several different datasets. We first show that StyleSpace, the space of channel-wise style parameters, is significantly more disentangled than the other intermediate latent spaces explored by previous works. Next, we describe a method for discovering a large collection of style channels, each of which is shown to control a distinct visual attribute in a highly localized and disentangled manner. Third, we propose a simple method for identifying style channels that control a specific attribute, using a pretrained classifier or a small number of example images. Manipulation of visual attributes via these StyleSpace controls is shown to be better disentangled than via those proposed in previous works. To show this, we make use of a newly proposed Attribute Dependency metric. Finally, we demonstrate the applicability of StyleSpace controls to the manipulation of real images. Our findings pave the way to semantically meaningful and well-disentangled image manipulations via simple and intuitive interfaces.


# DCI metric

To calculate DCI metric, we need a latent code file and corresponding attribute file. You may reproduce the DCI results for FFHQ through following steps:

1. Download pretrained classifers for CelebA 40 attributes.
```
mkdir metrics_checkpoint
gdown https://drive.google.com/drive/folders/1MvYdWCBuMfnoYGptRH-AgKLbPTsIQLhl -O ./metrics_checkpoint --folder
```
2. Generate latent code W, S and corresponding images. Please use the full distribution of the data without using truncation trick.
```
dataset_name='ffhq' 
output_path='./npy/ffhq_NT'
python GetCode.py --dataset_name $dataset_name --output_path $output_path --code_type 'w' --no_truncation
python GetCode.py --dataset_name $dataset_name --output_path $output_path --code_type 's_flat' 
python GetCode.py --dataset_name $dataset_name --output_path $output_path --code_type 'images'  --resize 256
```
3. Annotate the images using pretrained classifers.
```
img_path=$output_path'/images_100K.npy'
save_path=$output_path'/attribute'
classifer_path='./metrics_checkpoint'
python GetAttribute.py -img_path  $img_path -save_path $save_path -classifer_path $classifer_path
```
4. Calculate DCI for W and S space. 
```
latent_path=$output_path'/W.npy'
attribute_path=$output_path'/attribute'
save_path=$output_path'/DCI_W'
python DCI.py -latent_path $latent_path   -attribute_path $attribute_path -save_path $save_path

latent_path=$output_path'/S_Flat.npy'
save_path=$output_path'/DCI_S'
python DCI.py -latent_path $latent_path   -attribute_path $attribute_path -save_path $save_path
```
5. Show DCI results. 
```
latent_path=$output_path'/W.npy'
attribute_path=$output_path'/attribute'
save_path=$output_path'/DCI_W'
python DCI.py -latent_path $latent_path   -attribute_path $attribute_path -save_path $save_path -mode test

latent_path=$output_path'/S_Flat.npy'
save_path=$output_path'/DCI_S'
python DCI.py -latent_path $latent_path   -attribute_path $attribute_path -save_path $save_path -mode test
```

# Localized channels
1. Generate W, S and images 
```
dataset_name='ffhq'
python GetCode.py --dataset_name $dataset_name --code_type 'w'
python GetCode.py --dataset_name $dataset_name --code_type 's'
python GetCode.py --dataset_name $dataset_name --code_type 'images_1K'
```

2. Generate gradient maps, this step requires a number of gpus and long time. Please modify the paramethers in bash_script/gradient_map.sh.
```
sbatch bash_script/gradient_map.sh
```

3. Generate Semantic segmentation for natural human face. You could replace this part by custom model. Since the gradient map is only 32x32 resolution, it could not distingush small regions, please combine small semantic regions to a big one (up lip+ down lip + mouth =mouth).
```
git clone https://github.com/zllrunning/face-parsing.PyTorch
mv face-parsing.PyTorch  face_parsing
cd face-parsing
gdown https://drive.google.com/uc?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812 -O ./  #download pretrained model 
cp ../GetMask.py ./GetMask2.py
python GetMask2.py -model_path './79999_iter.pth' -img_path '../npy/ffhq/images.npy' -save_path '../npy/ffhq/semantic_mask.npy'
cd .. #back to root folder StyleSpace
```

4. Calculate the overlap between semantic mask and gradient mask
```
sbatch bash_script/align_mask.sh
```

5. Calculate the percentage of most activated regions for each channel. The output semantic_top_32 file contain the annotation for each channel, please refer to [this nootbook](https://github.com/betterze/StyleSpace/blob/main/StyleSpace_advance.ipynb) for visualizing the effect of most activated channels in each semantic region. 
```
python semantic_channel.py -align_folder './npy/ffhq/align_mask_32'  -s_path './npy/ffhq/S' -save_folder  './npy/ffhq/' 
```

# Attribute specific channels

To get the attribute specific channels (for example, channels for smiling), we need to use classifers to annotate a set of generated images. Please refer to [this part](https://github.com/betterze/StyleSpace#dci-metric) to download the classifers (1), and annotate the images (3). The only difference is that when generating images, we use trication trick (remove the  --no_truncation flag) as following:

```
dataset_name='ffhq' 
output_path='./npy/ffhq'
python GetCode.py --dataset_name $dataset_name --code_type 'w' 
python GetCode.py --dataset_name $dataset_name --code_type 'images'  --resize 256
```

After obtaining the attribue file, please refer to [this nootbook](https://github.com/betterze/StyleSpace/blob/main/StyleSpace_advance.ipynb) for visualizing the effect of most activated channels of a certrain attribute.




# Results
#### generated face manipulation
![](imgs/ffhq.png)

#### generated car and bedroom manipulation
![](imgs/car_bed.png)
#### real face manipulation
![](imgs/real.png)



