#!/bin/bash

#slurm options
#SBATCH -n 20
#SBATCH --mem 100000
#SBATCH --nodelist=gpu02
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -J gen_data_MOL_EXTEND_ATOM
#SBATCH -o log/%j.log
#SBATCH --exclude=gpu[06]

## user's own commands below
export OMP_NUM_THREADS=20
export MKL_NUM_THREADS=20
export OPENBLAS_NUM_THREADS=20

export PYSCF_TMPDIR=~/workdir/tmp
export PYSCF_MAX_MEMORY=40000
export PYTHONPATH=~/python:$PYTHONPATH
export LD_LIBRARY_PATH=~/anaconda3/lib:$LD_LIBRARY_PATH
export DATA_PATH=~/workdir/cadft/data/grids_mrks_ccsd_1

export NVIDIA_VISIBLE_DEVICES=1
# export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -nr | head -1 | awk '{ print $NF }')
export CUDA_VISIBLE_DEVICES=1

~/anaconda3/envs/pyscf/bin/python gen_dm_ene.py -dl START END STEP -b BASIS --extend_atom EXTEND_ATOM --extend_xyz 0 --name_mol MOL --load_inv True