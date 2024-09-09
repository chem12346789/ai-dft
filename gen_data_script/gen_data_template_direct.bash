#!/bin/bash

export OMP_NUM_THREADS=20
export MKL_NUM_THREADS=20
export OPENBLAS_NUM_THREADS=20

export PYSCF_MAX_MEMORY=40000
export PYTHONPATH=~/python:$PYTHONPATH
export LD_LIBRARY_PATH=~/anaconda3/lib:$LD_LIBRARY_PATH
export DATA_PATH=~/grids_mrks_ccsd_1

export NVIDIA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES=NUMBER_OF_GPU
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=power.draw,index --format=csv,nounits,noheader | sort -nr | head -1 | awk '{ print $NF }')

~/.conda/envs/pyscf/bin/python gen_dm_ene.py -dl START END STEP -b BASIS --extend_atom EXTEND_ATOM --extend_xyz 0 --name_mol MOL --load_inv True
