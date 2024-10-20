#!/bin/bash

#slurm options
#SBATCH -n 1
#SBATCH --cpus-per-task=12
#SBATCH --mem 100000
#SBATCH -p gpu
#SBATCH --nodelist=gpu07
#SBATCH -J validate-CHECKPOINT-HIDDEN_SIZE
#SBATCH -o log/CHECKPOINT.log

## set environment variables
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12

export PYSCF_TMPDIR=~/workdir/tmp
export PYSCF_MAX_MEMORY=50000
export PYTHONPATH=~/python:$PYTHONPATH
export LD_LIBRARY_PATH=~/anaconda3/envs/pyscf/lib:$LD_LIBRARY_PATH
export DATA_PATH=~/workdir/cadft/data/grids_mrks_scf_RESIDUAL

export NVIDIA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=power.draw,index --format=csv,nounits,noheader | sort -n | head -1 | awk '{ print $NF }')

## user's own commands below
# ~/anaconda3/envs/pyscf/bin/python test.py -dl -0.45 0.45 10 -b cc-pCVTZ --extend_atom 0 --extend_xyz 0 --load CHECKPOINT --name_mol propane propyne propylene allene methane ethane ethylene acetylene cyclopropene cyclopropane benzene cyclopentane isopentane pentane butane butyne isobutane butadiene --input_size INPUT_SIZE --hidden_size HIDDEN_SIZE --output_size OUTPUT_SIZE --residual RESIDUAL --num_layer NUM_LAYER --require_grad True --precision float64 --load_epoch -1 >log/CHECKPOINT.out
~/anaconda3/envs/pyscf/bin/python test.py -dl -0.45 0.45 10 -b cc-pCVTZ --extend_atom 1-2 --extend_xyz 0 --load CHECKPOINT --name_mol methane --input_size INPUT_SIZE --hidden_size HIDDEN_SIZE --output_size OUTPUT_SIZE --residual RESIDUAL --num_layer NUM_LAYER --require_grad True --precision float64 --load_epoch -1 >log/CHECKPOINT.out
#