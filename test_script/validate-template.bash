#!/bin/bash

#slurm options
#SBATCH -n 20
#SBATCH -p gpu
#SBATCH -J validate-CHECKPOINT-HIDDEN_SIZE
#SBATCH -o log/CHECKPOINT.log
#SBATCH --exclude=gpu[01-04]

## set environment variables
export OMP_NUM_THREADS=20
export MKL_NUM_THREADS=20
export OPENBLAS_NUM_THREADS=20

export PYSCF_TMPDIR=~/workdir/tmp
export PYSCF_MAX_MEMORY=80000
export PYTHONPATH=~/python:$PYTHONPATH
export LD_LIBRARY_PATH=~/anaconda3/envs/pyscf/lib:$LD_LIBRARY_PATH

## user's own commands below
# ~/anaconda3/envs/pyscf-numpy1/bin/python test.py -dl -0.45 0.45 10 -b cc-pCVTZ --extend_atom 0 --extend_xyz 0 --load CHECKPOINT --name_mol benzene cyclopentane isopentane pentane butane butyne isobutane butadiene methane methyl-openshell ethane ethylene acetylene propane propyne propylene cyclopropene cyclopropane --input_size INPUT_SIZE --hidden_size HIDDEN_SIZE --output_size OUTPUT_SIZE --residual RESIDUAL --num_layer NUM_LAYER --precision float32 --load_epoch -1 >log/CHECKPOINT.out
# 
~/anaconda3/envs/pyscf-numpy1/bin/python test.py -dl -0.45 0.45 10 -b cc-pCVTZ --extend_atom 0 --extend_xyz 0 --load CHECKPOINT --name_mol benzene cyclopentane isopentane pentane butane butyne isobutane butadiene methane methyl-openshell ethane ethylene acetylene propane propyne propylene cyclopropene cyclopropane --input_size INPUT_SIZE --hidden_size HIDDEN_SIZE --output_size OUTPUT_SIZE --residual RESIDUAL --num_layer NUM_LAYER --precision float32 --load_epoch -1 >log/CHECKPOINT.out
# 
# ~/anaconda3/envs/pyscf-numpy1/bin/python test.py -dl -0.45 0.45 10 -b cc-pCVTZ --extend_atom 0 --extend_xyz 0 --load CHECKPOINT --name_mol benzene cyclopentane isopentane pentane butane butyne isobutane butadiene methane methyl-openshell ethane ethylene acetylene propane propyne propylene cyclopropene cyclopropane --input_size INPUT_SIZE --hidden_size HIDDEN_SIZE --output_size OUTPUT_SIZE --residual RESIDUAL --num_layer NUM_LAYER --precision float32 --load_epoch 1000 >log/CHECKPOINT.out
# 