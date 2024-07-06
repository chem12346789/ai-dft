#!/bin/bash

#slurm options
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -J cc-pCVDZ1
#SBATCH -o log/cc-pCVDZ1.log
#SBATCH --exclude=gpu[001,003-007]

export OMP_NUM_THREADS=20
export MKL_NUM_THREADS=20
export OPENBLAS_NUM_THREADS=20

export PYSCF_MAX_MEMORY=40000
export PYSCF_TMPDIR=~/workdir/tmp
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH=~/python:$PYTHONPATH
export LD_LIBRARY_PATH=~/anaconda3/lib:$LD_LIBRARY_PATH

~/anaconda3/bin/python gen_dm_ene.py -dl -0.5 0.5 11 -b cc-pCVDZ --extend_atom 0 --extend_xyz 0 --name_mol methane ethylene propane propylene cyclopropene butane isobutane
