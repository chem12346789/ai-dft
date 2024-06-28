#!/bin/bash

export OMP_NUM_THREADS=28
export MKL_NUM_THREADS=28
export OPENBLAS_NUM_THREADS=28

export PYSCF_TMPDIR=~/workdir/tmp
export PYTHONPATH=~/python:$PYTHONPATH
export LD_LIBRARY_PATH=~/anaconda3/lib:$LD_LIBRARY_PATH

## user's own commands below
nohup sh -c '~/anaconda3/bin/python test.py -dl -0.45 0.45 10 -b cc-pCVDZ --extend_atom 0 --extend_xyz 0 --load 2024-06-25-12-52-25 --name_mol methane ethane ethylene acetylene propane pentane cyclopentane isopentane benzene --hidden_size 64 --residual 0 --num_layer 4 --precision float32 >log/test3.log' >log/test3.out &
