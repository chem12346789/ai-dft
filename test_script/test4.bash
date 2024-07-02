#!/bin/bash

#slurm options
#SBATCH -p gpu
#SBATCH -J train-ccdft-0
#SBATCH -o log/%j.log
#SBATCH --exclude=gpu[006]

export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12

export NVIDIA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -nr | head -1 | awk '{ print $NF }')

export PYSCF_TMPDIR=~/workdir/tmp
export PYTHONPATH=~/python:$PYTHONPATH
export LD_LIBRARY_PATH=~/anaconda3/lib:$LD_LIBRARY_PATH

## user's own commands below
~/anaconda3/bin/python test.py -dl -0.45 0.45 10 -b cc-pCVDZ --extend_atom 0 --extend_xyz 0 --load 2024-06-28-19-54-30 --name_mol methane ethane ethylene acetylene propane pentane cyclopentane isopentane benzene --hidden_size 16 --residual 5 --num_layer 4 --precision float32 >log/test4.sbatch
