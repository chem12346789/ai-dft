#!/bin/bash

export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12

export NVIDIA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -nr | head -1 | awk '{ print $NF }')

export PYTHONPATH=~/python:$PYTHONPATH
export LD_LIBRARY_PATH=~/anaconda3/lib:$LD_LIBRARY_PATH

nohup sh -c '~/anaconda3/bin/python train.py -dl -0.1 0.1 3 -b cc-pvdz --extend_atom 0 --extend_xyz 0 1 2 --eval_step 100 --load NEW --batch_size 32 --epoch 100000 --hidden_size 302  --residual 1 --num_layer 3 --ene_grid_factor 1 --precision float32 >log/train1.log' >log/train1.out &
