#!/bin/bash

#slurm options
#SBATCH -p gpu
#SBATCH -J cc-pCVDZ1
#SBATCH -o log/cc-pCVDZ1.out
#SBATCH --exclude=gpu[001,003-007]

export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12

export PYSCF_MAX_MEMORY=40000
export PYSCF_TMPDIR=~/workdir/tmp
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH=~/python:$PYTHONPATH
export LD_LIBRARY_PATH=~/anaconda3/lib:$LD_LIBRARY_PATH

nohup sh -c '~/anaconda3/bin/python gen_dm_ene.py -dl -0.05 -0.05 1 -b cc-pCVDZ --extend_atom 0 --extend_xyz 0 --name_mol pentane cyclopentane isopentane benzene --load_inv True >log/cc-pCVDZ1.out' >log/cc-pCVDZ1.sbatch &
