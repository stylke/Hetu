include(ExternalProject)

set(NCCL_SRC ${CMAKE_SOURCE_DIR}/third_party/nccl)

set(NCCL_WORK_SRC ${CMAKE_CURRENT_BINARY_DIR}/third_party/nccl)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/third_party)
  execute_process(
    COMMAND cp -r ${NCCL_SRC} ${NCCL_WORK_SRC}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/third_party
  )
message("nccl::${NCCL_WORK_SRC}")
message("CUDA_HOME:${CUDAToolkit_ROOT}")
execute_process(
  COMMAND make -j 32 src.build CUDA_HOME=${CUDAToolkit_ROOT}
  WORKING_DIRECTORY ${NCCL_WORK_SRC}
)

set(NCCL_ROOT "${CMAKE_CURRENT_BINARY_DIR}/third_party/nccl/build")
set(NCCL_LIB_DIR ${NCCL_ROOT}/lib)
set(NCCL_INCLUDE_DIRS ${NCCL_ROOT}/include)
find_library(NCCL_LIBRARIES_SRC
    NAMES ${NCCL_LIBNAME}
    HINTS
    ${NCCL_LIB_DIR}
    REQUIRED)
message(STATUS "NCCL_INCLUDE_DIRS ${NCCL_INCLUDE_DIRS}")    
message(STATUS "NCCL_LIB_DIR ${NCCL_LIB_DIR}")
message(STATUS "NCCL_LIBNAME ${NCCL_LIBNAME}")
message(STATUS "NCCL_LIBRARIES_SRC ${NCCL_LIBRARIES_SRC}")

