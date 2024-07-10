#!/bin/bash

#slurm options
#SBATCH -p gpu
#SBATCH -J validate-CHECKPOINT-HIDDEN_SIZE
#SBATCH -o log/CHECKPOINT.log
#SBATCH --exclude=gpu[002-007]

## set environment variables
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12

export PYSCF_MAX_MEMORY=40000
export PYSCF_TMPDIR=~/workdir/tmp
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH=~/python:$PYTHONPATH
export LD_LIBRARY_PATH=~/anaconda3/lib:$LD_LIBRARY_PATH

## user's own commands below
~/anaconda3/bin/python test.py -dl -0.45 0.45 10 -b cc-pCVDZ --extend_atom 0 --extend_xyz 0 --load CHECKPOINT --name_mol methane ethane ethylene acetylene propyne propylene allene cyclopropene cyclopropane propane butane butyne isobutane butadiene pentane cyclopentane isopentane benzene --input_size 1 --hidden_size HIDDEN_SIZE --output_size 1 --residual 0 --num_layer 4 --precision float32 > log/CHECKPOINT.out
# ~/anaconda3/bin/python test.py -dl -0.45 0.45 10 -b cc-pCVDZ --extend_atom 0 --extend_xyz 0 --load CHECKPOINT --name_mol methane ethane ethylene acetylene propane propyne propylene allene cyclopropene cyclopropane pentane cyclopentane isopentane benzene --hidden_size HIDDEN_SIZE --residual 0 --num_layer 4 --precision float64
# ~/anaconda3/bin/python test.py -dl -0.45 0.45 10 -b cc-pCVDZ --extend_atom 0 --extend_xyz 0 --load CHECKPOINT --name_mol propyne butane --hidden_size HIDDEN_SIZE --residual RESIDUAL --num_layer NUM_LAYER --precision float32
