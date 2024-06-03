#!/bin/bash

#slurm options
#SBATCH -p gpu
#SBATCH -J train-ccdft-CHECKPOINT-HIDDEN_SIZE
#SBATCH -o log/%j.log
#SBATCH --exclude=gpu[002,006]
#SBATCH --mem-per-gpu=20G

## user's own commands below
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12

export PYTHONPATH=~/python:$PYTHONPATH
export LD_LIBRARY_PATH=~/anaconda3/lib:$LD_LIBRARY_PATH

~/anaconda3/bin/python train.py -dl -0.5 0.5 11 -b cc-pvdz --extend_atom 0 1 2 --extend_xyz 0 1 2 --eval_step 100 --load CHECKPOINT --batch_size 4096 --epoch 100000 --hidden_size HIDDEN_SIZE --residual 0 --num_layer 3 ENE_GRID_FACTOR
