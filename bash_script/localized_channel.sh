#!/bin/sh

# 1.Generate W, S and images 
dataset_name='ffhq'
python GetCode.py --dataset_name $dataset_name --code_type 'w'
python GetCode.py --dataset_name $dataset_name --code_type 's'
python GetCode.py --dataset_name $dataset_name --code_type 'images_1K'


# 2.Generate gradient maps, this step requires a number of gpus and long time 
sbatch bash_script/gradient_map.sh #please modify the paramethers 

# 3.Generate Semantic segmentation for natural human face. You could replace this part by custom model. Since the gradient map is only 32x32 resolution, it could not distingush small regions, please combine small semantic regions to a big one (up lip+ down lip + mouth =mouth).
git clone https://github.com/zllrunning/face-parsing.PyTorch
mv face-parsing.PyTorch  face_parsing
cd face-parsing
gdown https://drive.google.com/uc?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812 -O ./  #download pretrained model 
cp ../GetMask.py ./GetMask2.py
python GetMask2.py -model_path './79999_iter.pth' -img_path '../npy/ffhq/images.npy' -save_path '../npy/ffhq/semantic_mask.npy'
cd .. #back to root folder StyleSpace


# 4.Calculate the overlap between semantic mask and gradient mask 
sbatch bash_script/align_mask.sh #please modify the paramethers 

# 5.Calculate the percentage of most activated regions for each channel 
python semantic_channel.py -align_folder './npy/ffhq/align_mask_32'  -s_path './npy/ffhq/S' -save_folder  './npy/ffhq/' 



