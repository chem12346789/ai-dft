export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12
export NVIDIA_VISIBLE_DEVICES=2

export PYSCF_TMPDIR=~/workdir/tmp
export PYTHONPATH=~/python:$PYTHONPATH
export LD_LIBRARY_PATH=~/anaconda3/lib:$LD_LIBRARY_PATH

nohup sh -c '~/anaconda3/bin/python gen_dm_ene.py -dl -0 0 1 -b cc-pcvqz --extend_atom 0 1 2 --extend_xyz 0 1 2 --name_mol Acetylene1 >log/Acetylene1.out' >log/mrks1.out &
