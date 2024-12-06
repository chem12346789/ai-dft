#!/bin/bash

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

export PYTHONPATH=~/python:$PYTHONPATH
export PYSCF_MAX_MEMORY=80000
export LD_LIBRARY_PATH=~/anaconda3/lib:$LD_LIBRARY_PATH
export DATA_PATH=/raid/data/chenzihao/data/grids_mrks_ccsd_1

export NVIDIA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=power.draw,index --format=csv,nounits,noheader | sort -n | head -1 | awk '{ print $NF }')
# export CUDA_VISIBLE_DEVICES=NUMBER_OF_GPU

~/anaconda3/envs/pyscf/bin/python train.py -dl -1.0 2.5 36 -b cc-pVDZ --extend_atom 0-1 --extend_xyz 0 --eval_step EVAL_STEP --batch_size BATCH_SIZE --epoch 5000 --input_size INPUT_SIZE --hidden_size HIDDEN_SIZE --output_size OUTPUT_SIZE --num_layer NUM_LAYER --residual RESIDUAL --precision PRECISION --ene_weight ENE_WEIGHT --pot_weight POT_WEIGHT --with_eval WITH_EVAL --load LOAD_MODEL

echo $! >>log/save_pid.txt 2>&1
