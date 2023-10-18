# Use hetu 3d-parallel to train gpt

1. mkdir `data` folder under `examples/nlp/gpt`, copy data in this folder like `data/wikicorpus_en_training_0.hdf5`

2. run `bash scripts/train_hetu_gpt_parallel.sh` to use 2d parallel (dp=2, tp=2, 4 gpus)

3. run `bash scripts/train_hetu_gpt_3d_parallel.sh` to use 3d parallel (dp=2, tp=2, pp=2, 8 gpus)

# Use hetu 2d-parallel to inference gpt

1. mkdir `data` folder under `examples/nlp/gpt`, copy data in this folder like `data/wikicorpus_en_training_0.hdf5`

2. run `bash scripts/inference_hetu_gpt_parallel.sh` to use 2d parallel (dp=2, tp=2, 4 gpus)
