#!/bin/bash

export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12

export NVIDIA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -nr | head -1 | awk '{ print $NF }')

export PYTHONPATH=~/python:$PYTHONPATH
export LD_LIBRARY_PATH=~/anaconda3/lib:$LD_LIBRARY_PATH

nohup sh -c '~/anaconda3/bin/python train.py -dl -0.5 0.5 11 -b cc-pCVDZ --extend_atom 0 --extend_xyz 0 --eval_step 100 --batch_size 64 --epoch 10000 --hidden_size 64 --num_layer 4 --residual 0 --precision float32 --ene_weight 1.0 >log/train1.log' >log/train1.out &
