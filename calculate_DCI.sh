#!/bin/sh


# download the pretrained CelebA classifers
mkdir metrics_checkpoint
gdown https://drive.google.com/drive/folders/1MvYdWCBuMfnoYGptRH-AgKLbPTsIQLhl -O ./metrics_checkpoint --folder


dataset_name='ffhq' 
output_path='./npy/ffhq_NT'
# input prepare data. When calculate the DCI, please use the full distribution of the data without using truncation trick
python GetCode.py --dataset_name $dataset_name --output_path $output_path --code_type 'w' --no_truncation
python GetCode.py --dataset_name $dataset_name --output_path $output_path --code_type 's_flat' 
python GetCode.py --dataset_name $dataset_name --output_path $output_path --code_type 'images'  --resize 256

img_path=$output_path'/images_100K.npy'
save_path=$output_path'/attribute'
classifer_path='./metrics_checkpoint'
python GetAttribute.py -img_path  $img_path -save_path $save_path -classifer_path $classifer_path


latent_path=$output_path'/W.npy'
attribute_path=$output_path'/attribute'
save_path=$output_path'/DCI_W'
python DCI.py -latent_path $latent_path   -attribute_path $attribute_path -save_path $save_path
python DCI.py -latent_path $latent_path   -attribute_path $attribute_path -save_path $save_path -mode test

latent_path=$output_path'/S_Flat.npy'
save_path=$output_path'/DCI_S'
python DCI.py -latent_path $latent_path   -attribute_path $attribute_path -save_path $save_path
python DCI.py -latent_path $latent_path   -attribute_path $attribute_path -save_path $save_path -mode test

