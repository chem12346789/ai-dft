#!/bin/bash

#slurm options
#SBATCH -n 20
#SBATCH -p gpu
#SBATCH -J gen_data-MOL
#SBATCH -o log/%j.log
#SBATCH --exclude=gpu[01,03]

## user's own commands below
export OMP_NUM_THREADS=20
export MKL_NUM_THREADS=20
export OPENBLAS_NUM_THREADS=20

export PYSCF_TMPDIR=~/workdir/tmp
export PYTHONPATH=~/python:$PYTHONPATH
export LD_LIBRARY_PATH=~/anaconda3/lib:$LD_LIBRARY_PATH

export NVIDIA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES=NUMBER_OF_GPU

~/anaconda3/envs/pyscf/bin/python gen_dm_ene.py -dl START END STEP -b BASIS --extend_atom 3 --extend_xyz 0 --name_mol MOL
