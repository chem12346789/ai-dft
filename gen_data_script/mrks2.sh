export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12

export NVIDIA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -nr | head -1 | awk '{ print $NF }')

export PYSCF_TMPDIR=~/workdir/tmp
export PYTHONPATH=~/python:$PYTHONPATH
export LD_LIBRARY_PATH=~/anaconda3/lib:$LD_LIBRARY_PATH

nohup sh -c '~/anaconda3/bin/python gen_dm_ene.py -dl -0.5 0 6 -b cc-pCVTZ --extend_atom 0 --extend_xyz 0 1 2 --name_mol ethane ethylene acetylene methane --load_inv True >log/cc-pCVTZ2.out' >log/cc-pCVTZ2.log &
