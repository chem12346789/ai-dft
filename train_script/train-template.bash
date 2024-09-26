#!/bin/bash

#slurm options
#SBATCH -n 2
#SBATCH -p gpu
#SBATCH --nodelist=gpu05
#SBATCH -J train-ccdft-EVAL_STEP-INPUT_SIZE-HIDDEN_SIZE-OUTPUT_SIZE-NUM_LAYER-RESIDUAL-BATCH_SIZE-ENE_WEIGHT-POT_WEIGHT-WITH_EVAL
#SBATCH -o log/%j.log

## user's own commands below
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

export NVIDIA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=power.draw,index --format=csv,nounits,noheader | sort -n | head -1 | awk '{ print $NF }')
# export CUDA_VISIBLE_DEVICES=NUMBER_OF_GPU

export PYTHONPATH=~/python:$PYTHONPATH
export PYSCF_MAX_MEMORY=80000
export LD_LIBRARY_PATH=~/anaconda3/lib:$LD_LIBRARY_PATH
export DATA_PATH=~/workdir/cadft/data/grids_mrks_scf_RESIDUAL

# ~/anaconda3/envs/pyscf/bin/python train.py -dl -1.0 2.5 36 -b cc-pCVTZ --extend_atom 0 --extend_xyz 0 --eval_step EVAL_STEP --batch_size BATCH_SIZE --epoch 5000 --input_size INPUT_SIZE --hidden_size HIDDEN_SIZE --output_size OUTPUT_SIZE --num_layer NUM_LAYER --residual RESIDUAL --precision float32 --ene_weight ENE_WEIGHT --pot_weight POT_WEIGHT --with_eval WITH_EVAL --load LOAD_MODEL
~/anaconda3/envs/pyscf/bin/python train.py -dl -1.0 2.5 36 -b cc-pCVTZ --extend_atom 0 2 0-1 --extend_xyz 0 --eval_step EVAL_STEP --batch_size BATCH_SIZE --epoch 1500 --input_size INPUT_SIZE --hidden_size HIDDEN_SIZE --output_size OUTPUT_SIZE --num_layer NUM_LAYER --residual RESIDUAL --precision float32 --ene_weight ENE_WEIGHT --pot_weight POT_WEIGHT --with_eval WITH_EVAL --load LOAD_MODEL
#
# ~/anaconda3/envs/pyscf/bin/python train.py -dl -0.1 0.1 3 -b cc-pCVTZ --extend_atom 0 --extend_xyz 0 --eval_step EVAL_STEP --batch_size BATCH_SIZE --epoch 1000 --input_size INPUT_SIZE --hidden_size HIDDEN_SIZE --output_size OUTPUT_SIZE --num_layer NUM_LAYER --residual RESIDUAL --precision float32 --ene_weight ENE_WEIGHT --pot_weight POT_WEIGHT --with_eval WITH_EVAL --load LOAD_MODEL
