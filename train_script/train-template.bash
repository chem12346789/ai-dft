#!/bin/bash

#slurm options
#SBATCH -p gpu
#SBATCH -J train-ccdft-CHECKPOINT-HIDDEN_SIZE
#SBATCH -o log/%j.log
#SBATCH --exclude=gpu[001,006,007]
#SBATCH --mem-per-gpu=20G

## user's own commands below
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12

export NVIDIA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -nr | head -1 | awk '{ print $NF }')

export PYTHONPATH=~/python:$PYTHONPATH
export LD_LIBRARY_PATH=~/anaconda3/lib:$LD_LIBRARY_PATH

~/anaconda3/bin/python train.py -dl -0 0 1 -b cc-pvdz --extend_atom 0 1 2 --extend_xyz 0 1 2 --eval_step 100 --load CHECKPOINT --batch_size 1 --epoch 100000 --hidden_size HIDDEN_SIZE  --residual 1 --num_layer 3 ENE_GRID_FACTOR