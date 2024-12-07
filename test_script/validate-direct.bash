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
export OMP_NUM_THREADS=24
export MKL_NUM_THREADS=24
export OPENBLAS_NUM_THREADS=24

export PYSCF_TMPDIR=/raid/data/chenzihao/tmp
export PYSCF_MAX_MEMORY=120000
export PYTHONPATH=~/python:$PYTHONPATH
export LD_LIBRARY_PATH=~/anaconda3/envs/pyscf/lib:$LD_LIBRARY_PATH
export DATA_PATH=/raid/data/chenzihao/data/grids_mrks_ccsd_1
export DATA_CC_PATH=/raid/data/chenzihao/data/test

export NVIDIA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=power.draw,index --format=csv,nounits,noheader | sort -n | head -1 | awk '{ print $NF }')

## user's own commands below
# ~/anaconda3/envs/pyscf/bin/python test.py -dl -0.45 2.45 30 -b cc-pCVTZ --extend_atom 0 --extend_xyz 0 --load CHECKPOINT --name_mol propane propyne propylene allene cyclopropene cyclopropane methane ethane ethylene acetylene benzene cyclopentane isopentane pentane butane butyne isobutane butadiene --input_size INPUT_SIZE --hidden_size HIDDEN_SIZE --output_size OUTPUT_SIZE --residual RESIDUAL --num_layer NUM_LAYER --require_grad True --precision float64 --load_epoch -1 >log/CHECKPOINT.out
#
~/anaconda3/envs/pyscf/bin/python test.py -dl -0.5 0.5 11 -b cc-pVDZ --extend_atom 0-1 --extend_xyz 0 --load CHECKPOINT --name_mol benzene cyclopentane isopentane pentane butane butyne isobutane butadiene methane ethane ethylene acetylene propane propyne propylene allene cyclopropene cyclopropane --input_size INPUT_SIZE --hidden_size HIDDEN_SIZE --output_size OUTPUT_SIZE --residual RESIDUAL --num_layer NUM_LAYER --precision float64 --load_epoch -1 >log/CHECKPOINT.out
#
# ~/anaconda3/envs/pyscf/bin/python test.py -dl -0.5 1.5 21 -b cc-pCVTZ --extend_atom 0-1 --extend_xyz 0 --load CHECKPOINT --name_mol butane propane --input_size INPUT_SIZE --hidden_size HIDDEN_SIZE --output_size OUTPUT_SIZE --residual RESIDUAL --num_layer NUM_LAYER --precision float64 --load_epoch -1 >log/CHECKPOINT.out

echo $! >>log/save_pid.txt 2>&1
