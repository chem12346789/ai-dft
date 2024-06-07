#!/bin/bash

#slurm options
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -J gen_data-MOL
#SBATCH -o log/%j.log
#SBATCH --exclude=gpu[001-006]
#SBATCH --mem-per-gpu=20G

## user's own commands below
export OMP_NUM_THREADS=14
export MKL_NUM_THREADS=14
export OPENBLAS_NUM_THREADS=14

export PYSCF_TMPDIR=~/workdir/tmp
export PYTHONPATH=~/python:$PYTHONPATH
export LD_LIBRARY_PATH=~/anaconda3/lib:$LD_LIBRARY_PATH

~/anaconda3/bin/python gen_dm_ene.py -dl START END STEP -b BASIS --extend_atom 0 --extend_xyz 0 1 2 --name_mol MOL
