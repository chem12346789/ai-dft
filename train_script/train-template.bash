#!/bin/bash

#slurm options
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -J train-ccdft-HIDDEN_SIZE-EVAL_STEP-BATCH_SIZE-NUM_LAYER-RESIDUAL-ENE_WEIGHT
#SBATCH -o log/%j.log
#SBATCH --exclude=gpu[001-002,004-007]

## user's own commands below
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

export PYSCF_TMPDIR=~/workdir-save/tmp
export PYTHONPATH=~/python:$PYTHONPATH
export LD_LIBRARY_PATH=~/anaconda3/lib:$LD_LIBRARY_PATH

~/anaconda3/bin/python train.py -dl -0.5 0.5 11 -b cc-pCVDZ --extend_atom 0 2 --extend_xyz 0 --eval_step EVAL_STEP --batch_size BATCH_SIZE --epoch 25000 --input_size INPUT_SIZE --hidden_size HIDDEN_SIZE --output_size OUTPUT_SIZE --num_layer NUM_LAYER --residual RESIDUAL --precision float32 --ene_weight ENE_WEIGHT --with_eval WITH_EVAL

# ~/anaconda3/bin/python train.py -dl -0.5 0.5 11 -b cc-pCVDZ --extend_atom 0 --extend_xyz 0 --eval_step EVAL_STEP --load 2024-07-08-19-49-14 --batch_size BATCH_SIZE --epoch 2500 --hidden_size HIDDEN_SIZE --num_layer NUM_LAYER --residual RESIDUAL --precision float32 --ene_weight ENE_WEIGHT --with_eval True

# ~/anaconda3/bin/python train.py -dl -0.1 0.1 3 -b cc-pvdz --extend_atom 0 --extend_xyz 0 1 2 --eval_step 100 --load CHECKPOINT --batch_size 32 --epoch 100000 --hidden_size HIDDEN_SIZE  --residual 1 --num_layer 3 ENE_GRID_FACTOR
