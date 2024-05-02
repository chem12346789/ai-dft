#!/bin/bash

#slurm options
#SBATCH -p cpu
#SBATCH -c 7
#SBATCH -J MOL_BASIS_START_END_STEP
#SBATCH -o log/%j.log

## user's own commands below
export OMP_NUM_THREADS=7
export MKL_NUM_THREADS=7
export OPENBLAS_NUM_THREADS=7

export PYSCF_TMPDIR=~/workdir/tmp
export PYTHONPATH=~/python:$PYTHONPATH
export LD_LIBRARY_PATH=~/anaconda3/lib:$LD_LIBRARY_PATH

~/anaconda3/bin/python gen_dm_ene.py -dl START END STEP -b BASIS --extend_atom 0 1 --extend_xyz 0 1 2 --name_mol MOL

# ~/anaconda3/bin/python ../gen_dm_ene.py --save True -me METHOD -fs 4 -l LEVEL -m MOL -bs True -b BASIS_NAME -is 500000 -ei 1e-8 --scf_step 250000 --error_scf 1e-8 -dl START END STEP
