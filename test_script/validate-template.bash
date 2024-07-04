#!/bin/bash

#slurm options
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -J validate-CHECKPOINT-HIDDEN_SIZE
#SBATCH -o log/%j.log
#SBATCH --exclude=gpu[001,002,004-007]

## set environment variables
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12

export LD_LIBRARY_PATH=~/anaconda3/lib:$LD_LIBRARY_PATH

## user's own commands below
# ~/anaconda3/bin/python test.py -dl -0.05 -0.05 1 -b cc-pCVDZ --extend_atom 0 --extend_xyz 0 --load CHECKPOINT --name_mol methane ethane ethylene acetylene propane pentane cyclopentane isopentane benzene --hidden_size HIDDEN_SIZE --residual 0 --num_layer 4 --precision float32
~/anaconda3/bin/python test.py -dl -0.45 0.45 10 -b cc-pCVDZ --extend_atom 0 --extend_xyz 0 --load CHECKPOINT --name_mol methane ethane ethylene acetylene propane pentane cyclopentane isopentane benzene --hidden_size HIDDEN_SIZE --residual RESIDUAL --num_layer NUM_LAYER --precision float32
