#!/bin/sh

# 1.Download pretrained classifers for CelebA 40 attributes.
mkdir metrics_checkpoint
gdown https://drive.google.com/drive/folders/1MvYdWCBuMfnoYGptRH-AgKLbPTsIQLhl -O ./metrics_checkpoint --folder

# 2.Generate latent code W, S and corresponding images. Please use the full distribution of the data without using truncation trick.
dataset_name='ffhq' 
output_path='./npy/ffhq_NT'
python GetCode.py --dataset_name $dataset_name --output_path $output_path --code_type 'w' --no_truncation
python GetCode.py --dataset_name $dataset_name --output_path $output_path --code_type 's_flat' 
python GetCode.py --dataset_name $dataset_name --output_path $output_path --code_type 'images_100K'  --resize 256

# 3.Annotate the images using pretrained classifers.
img_path=$output_path'/images_100K.npy'
save_path=$output_path'/attribute'
classifer_path='./metrics_checkpoint'
python GetAttribute.py -img_path  $img_path -save_path $save_path -classifer_path $classifer_path

#4.Calculate DCI for W and S space.
latent_path=$output_path'/W.npy'
attribute_path=$output_path'/attribute'
save_path=$output_path'/DCI_W'
python DCI.py -latent_path $latent_path   -attribute_path $attribute_path -save_path $save_path

latent_path=$output_path'/S_Flat.npy'
save_path=$output_path'/DCI_S'
python DCI.py -latent_path $latent_path   -attribute_path $attribute_path -save_path $save_path

# 5.Show DCI results.
latent_path=$output_path'/W.npy'
attribute_path=$output_path'/attribute'
save_path=$output_path'/DCI_W'
python DCI.py -latent_path $latent_path   -attribute_path $attribute_path -save_path $save_path -mode test

latent_path=$output_path'/S_Flat.npy'
save_path=$output_path'/DCI_S'
python DCI.py -latent_path $latent_path   -attribute_path $attribute_path -save_path $save_path -mode test


