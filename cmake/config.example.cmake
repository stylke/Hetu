######################
### Set paths ######
######################

# CUDA version >= 11.8
if(DEFINED ENV{CUDA_HOME})
  set(CUDAToolkit_ROOT $ENV{CUDA_HOME})
else()
  set(CUDAToolkit_ROOT /usr/local/cuda)
endif()

# CUDNN >= 8.2
# - CUDNN_ROOT: root directory of cudnn
set(CUDNN_ROOT)

# NCCL version >= 2.19
set(NCCL_ROOT)

# MPI >= 4.1
set(MPI_ROOT)

# PyBind11 2.6.2
# - pybind11_DIR: cmake directory of pybind11, 
#                 can be obtained by `python3 -m pybind11 --cmakedir` 
#                 if pybind11 has been installed via pip
# if not found, we'll download and compile it in time
set(pybind11_DIR)

# DNNL (oneDNN) 3.0
# - DNNL_ROOT: root directory of zeromq
# - DNNL_BUILD: build directory of zeromq
# if not found, we'll download and compile it in time
set(DNNL_ROOT)
set(DNNL_BUILD)

# if you have a protoc in conda bin, ignore `/path/to/anaconda3/bin`
set(CMAKE_IGNORE_PATH)

set(FLASH_ROOT)
