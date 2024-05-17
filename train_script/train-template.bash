#!/bin/bash

#slurm options
#SBATCH -p gpu
#SBATCH -J train-ccdft-CHECKPOINT-HIDDEN_SIZE
#SBATCH -o log/%j.log
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu[004-006]

## user's own commands below
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12

export PYTHONPATH=~/python:$PYTHONPATH
export LD_LIBRARY_PATH=~/anaconda3/lib:$LD_LIBRARY_PATH

~/anaconda3/bin/python train.py -dl -0.5 0.5 51 -b cc-pvdz --extend_atom 0 1 --extend_xyz 0 1 2 --eval_step 100 --batch_size 65536 --epoch 100000 --hidden_size HIDDEN_SIZE
