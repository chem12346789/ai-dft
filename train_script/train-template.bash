#!/bin/bash

#slurm options
#SBATCH -p gpu
#SBATCH -J train-ccdft-HIDDEN_SIZE-ENE_GRID_FACTOR
#SBATCH -o log/train-ccdft-HIDDEN_SIZE-ENE_GRID_FACTOR.log
#SBATCH --exclude=gpu[001,003-007]

## user's own commands below
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

export NVIDIA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -nr | head -1 | awk '{ print $NF }')
export PYSCF_TMPDIR=~/workdir-save/tmp

export PYTHONPATH=~/python:$PYTHONPATH
export LD_LIBRARY_PATH=~/anaconda3/lib:$LD_LIBRARY_PATH

~/anaconda3/bin/python train.py -dl 0 0.5 1 -b cc-pCVDZ --extend_atom 0 --extend_xyz 0 --eval_step EVAL_STEP --batch_size BATCH_SIZE --epoch 100000 --hidden_size HIDDEN_SIZE --num_layer NUM_LAYER --residual RESIDUAL --precision float32 --ene_weight ENE_GRID_FACTOR --with_eval True

# ~/anaconda3/bin/python train.py -dl -0.1 0.1 3 -b cc-pvdz --extend_atom 0 --extend_xyz 0 1 2 --eval_step 100 --load CHECKPOINT --batch_size 32 --epoch 100000 --hidden_size HIDDEN_SIZE  --residual 1 --num_layer 3 ENE_GRID_FACTOR
