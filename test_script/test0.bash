#!/bin/bash

#slurm options
#SBATCH -p gpu
#SBATCH -J test-ccdft-0
#SBATCH -o log/test0.log
#SBATCH --exclude=gpu[001,002,004-007]

## set environment variables
export OMP_NUM_THREADS=20
export MKL_NUM_THREADS=20
export OPENBLAS_NUM_THREADS=20

export PYSCF_MAX_MEMORY=40000
export PYSCF_TMPDIR=~/workdir/tmp
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH=~/python:$PYTHONPATH
export LD_LIBRARY_PATH=~/anaconda3/lib:$LD_LIBRARY_PATH

## user's own commands below
~/anaconda3/bin/python test.py -dl -0.5 0.5 11 -b cc-pCVDZ --extend_atom 0 --extend_xyz 0 --load 2024-07-07-20-09-53 --name_mol methane ethane ethylene acetylene propyne --hidden_size 64 --residual 0 --num_layer 4 --precision float32 >log/test0.out
