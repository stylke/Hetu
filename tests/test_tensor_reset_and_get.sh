export HT_INTERNAL_LOG_LEVEL=TRACE
export CUDA_VISIBLE_DEVICES=0,1,2,3 
mpirun --allow-run-as-root -np 4 python test_tensor_reset_and_get.py 