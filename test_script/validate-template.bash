#!/bin/bash

#slurm options
#SBATCH -p gpu
#SBATCH -J validate-CHECKPOINT-HIDDEN_SIZE
#SBATCH -o log/CHECKPOINT.log
#SBATCH --exclude=gpu[01-02,07]

## set environment variables
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12

export PYSCF_MAX_MEMORY=160000
export PYSCF_TMPDIR=~/workdir-save/tmp
export PYTHONPATH=~/python:$PYTHONPATH
export LD_LIBRARY_PATH=~/anaconda3/lib:$LD_LIBRARY_PATH

## user's own commands below
~/anaconda3/envs/pyscf/bin/python test.py -dl -0.45 0.45 10 -b cc-pCVTZ --extend_atom 0 --extend_xyz 0 --load CHECKPOINT --name_mol methane ethane ethylene acetylene cyclopropene cyclopropane propane propylene propyne allene butane butyne isobutane butadiene pentane cyclopentane isopentane benzene --input_size 4 --hidden_size HIDDEN_SIZE --output_size 1 --residual 0 --num_layer 4 --precision float32 > log/CHECKPOINT.out

# ~/anaconda3/envs/pyscf/bin/python test.py -dl -0.45 0.45 10 -b cc-pCVTZ --extend_atom 0 --extend_xyz 0 --load CHECKPOINT --name_mol hexane 2-methylpentane 3-methylpentane 2,2-dimethylbutane 2,3-dimethylbutane cyclohexane 1-hexene methylcyclopentane 3,3-dimethyl-1-butene 4-methyl-1-pentene 2-methyl-1-pentene propylcyclopropane --input_size 4 --hidden_size HIDDEN_SIZE --output_size 1 --residual 0 --num_layer 4 --precision float32 > log/CHECKPOINT.out

# ~/anaconda3/bin/python test.py -dl -0.45 0.45 10 -b cc-pCVDZ --extend_atom 0 --extend_xyz 0 --load CHECKPOINT --name_mol methane ethane ethylene acetylene propane propyne propylene allene cyclopropene cyclopropane pentane cyclopentane isopentane benzene --hidden_size HIDDEN_SIZE --residual 0 --num_layer 4 --precision float64

# ~/anaconda3/bin/python test.py -dl -0.45 0.45 10 -b cc-pCVDZ --extend_atom 0 --extend_xyz 0 --load CHECKPOINT --name_mol propyne butane --hidden_size HIDDEN_SIZE --residual RESIDUAL --num_layer NUM_LAYER --precision float32
