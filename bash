#!/bin/sh

#git clone https://github.com/NVlabs/stylegan2-ada.git
#mv stylegan2-ada stylegan2_ada


mkdir model 
mkdir npy
mkdir html


cd model 
wget https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-ffhq-config-f.pkl
mv stylegan2-ffhq-config-f.pkl ffhq.pkl
cd ..

cd npy 
mkdir ffhq
cd ..

git clone https://github.com/openai/CLIP


















