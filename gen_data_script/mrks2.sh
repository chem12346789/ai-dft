export OMP_NUM_THREADS=28
export MKL_NUM_THREADS=28
export OPENBLAS_NUM_THREADS=28

export NVIDIA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -nr | head -1 | awk '{ print $NF }')

export PYSCF_TMPDIR=~/workdir/tmp
export PYTHONPATH=~/python:$PYTHONPATH
export LD_LIBRARY_PATH=~/anaconda3/lib:$LD_LIBRARY_PATH

nohup sh -c '~/anaconda3/bin/python gen_dm_ene.py -dl -0.5 0.5 11 -b aug-cc-pCVTZ --extend_atom 0 --extend_xyz 0 --name_mol acetylene >log/acetylene_aug-cc-pCVTZ.out' >log/mrks2_aug-cc-pCVTZ.out &
