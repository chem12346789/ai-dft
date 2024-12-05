#!/bin/bash

export OMP_NUM_THREADS=20
export MKL_NUM_THREADS=20
export OPENBLAS_NUM_THREADS=20

export PYTHONPATH=~/python:$PYTHONPATH
export LD_LIBRARY_PATH=~/anaconda3/lib:$LD_LIBRARY_PATH
export PYSCF_MAX_MEMORY=400000
export PYSCF_TMPDIR=/raid/data/chenzihao/tmp
export DATA_PATH=/raid/data/chenzihao/data/grids_mrks_ccsd_1

export NVIDIA_VISIBLE_DEVICES=1
# export CUDA_VISIBLE_DEVICES=NUMBER_OF_GPU
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=power.draw,index --format=csv,nounits,noheader | sort -n | head -1 | awk '{ print $NF }')

~/anaconda3/envs/pyscf/bin/python gen_dm_ene.py -dl START END STEP -b BASIS --extend_atom EXTEND_ATOM --extend_xyz 0 --name_mol methane ethane ethylene acetylene propane cyclopropane cyclopropene propylene propyne allene butane --load_inv True
