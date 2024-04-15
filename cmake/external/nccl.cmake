include(ExternalProject)

set(NCCL_SRC ${CMAKE_SOURCE_DIR}/third_party/nccl)

set(NCCL_WORK_SRC ${CMAKE_CURRENT_BINARY_DIR}/third_party/nccl)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/third_party)
  execute_process(
    COMMAND cp -r ${NCCL_SRC} ${NCCL_WORK_SRC}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/third_party
  )
message("nccl::${NCCL_WORK_SRC}")
execute_process(
  COMMAND make -j 32 src.build CUDA_HOME=$ENV{CUDA_HOME} NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"
  WORKING_DIRECTORY ${NCCL_WORK_SRC}
)

set(NCCL_ROOT "${CMAKE_CURRENT_BINARY_DIR}/third_party/nccl/build")
set(NCCL_INCLUDE_DIR ${NCCL_ROOT}/include)
set(NCCL_LIB_DIR ${NCCL_ROOT}/lib)

