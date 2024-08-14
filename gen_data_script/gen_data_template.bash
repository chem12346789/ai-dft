#!/bin/bash

#slurm options
#SBATCH -n 20
#SBATCH -p gpu
#SBATCH -J gen_data-MOL
#SBATCH -o log/%j.log
#SBATCH --exclude=gpu[01,07]

## user's own commands below
export OMP_NUM_THREADS=20
export MKL_NUM_THREADS=20
export OPENBLAS_NUM_THREADS=20

export PYSCF_TMPDIR=~/workdir-save/tmp
export PYTHONPATH=~/python:$PYTHONPATH
export LD_LIBRARY_PATH=~/anaconda3/lib:$LD_LIBRARY_PATH

~/anaconda3/envs/pyscf/bin/python gen_dm_ene.py -dl START END STEP -b BASIS --extend_atom 0 2 --extend_xyz 0 --name_mol MOL
