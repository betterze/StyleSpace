#!/bin/sh
#SBATCH --mem=60g
#SBATCH -c 2
#SBATCH --time=1-0
#SBATCH --gres=gpu:1
#SBATCH --array=0-249%20
#SBATCH --output='./log/gradient_map%A%a'


one=($(seq 0 4 996))
one_index=$((${SLURM_ARRAY_TASK_ID}%${#one[@]}))

python invert_mask.py  -model_path './model/ffhq.pkl' -data_path './npy/ffhq'  -img_sindex ${one[$one_index]} -num_per 4
























