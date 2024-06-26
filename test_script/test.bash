#!/bin/bash

export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12

export NVIDIA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -nr | head -1 | awk '{ print $NF }')

export PYSCF_TMPDIR=~/workdir/tmp
export PYTHONPATH=~/python:$PYTHONPATH
export LD_LIBRARY_PATH=~/anaconda3/lib:$LD_LIBRARY_PATH

## user's own commands below
# nohup sh -c '~/anaconda3/bin/python test.py -dl -0.1 0.1 3 -b cc-pCVDZ --extend_atom 0 --extend_xyz 0 --load 2024-06-24-15-29-57 --name_mol propane --hidden_size 64 --residual 0 --num_layer 2 >log/test.log' >log/test.out &
~/anaconda3/bin/python test.py -dl -0.5 0.5 5 -b cc-pCVDZ --extend_atom 0 --extend_xyz 0 --load 2024-06-26-12-34-27 --name_mol methane propane pentane benzene --hidden_size 64 --residual 0 --num_layer 5 --precision float32
~/anaconda3/bin/python test.py -dl -0.5 0.5 5 -b cc-pCVDZ --extend_atom 0 --extend_xyz 0 --load 2024-06-26-12-46-28 --name_mol methane propane pentane benzene --hidden_size 64 --residual 0 --num_layer 5 --precision float32
