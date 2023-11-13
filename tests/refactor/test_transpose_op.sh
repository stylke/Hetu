export HETU_INTERNAL_LOG_LEVEL=WARN
export CUDA_VISIBLE_DEVICES=0,1,2,3 
mpirun --allow-run-as-root -np 4 python test_transpose_op.py 