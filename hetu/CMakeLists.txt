# set(HETU_SRC_DIR ${CMAKE_SOURCE_DIR})
set(HETU_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
file(GLOB_RECURSE HETU_SRC ${HETU_SRC_DIR}/hetu/*.cc)
if(${HETU_COMPILE_CUDA})
  add_definitions(-DHETU_CUDA_ENABLED)
  file(GLOB_RECURSE HETU_CU_SRC ${HETU_SRC_DIR}/hetu/*.cu)
  list(APPEND HETU_SRC ${HETU_CU_SRC})
endif()
list(FILTER HETU_SRC EXCLUDE REGEX "${HETU_SRC_DIR}/hetu/test/*")
# list(FILTER HETU_SRC EXCLUDE REGEX "${HETU_SRC_DIR}/hetu/csrc/*")
add_library(hetu_C SHARED ${HETU_SRC})
target_include_directories(hetu_C PUBLIC ${HETU_SRC_DIR})

if(${HETU_COMPILE_OMP})
  target_link_libraries(hetu_C PUBLIC MPI::MPI_CXX)
endif()

if(${HETU_COMPILE_CUDA})
  target_link_libraries(hetu_C PUBLIC CUDA::cudart CUDA::cublas CUDA::cusparse CUDA::curand CUDA::cuda_driver)
  set_property(TARGET hetu_C PROPERTY CUDA_ARCHITECTURES ${HETU_CUDA_ARCH})
endif()

if(${HETU_COMPILE_CUDNN})
  find_package(CUDNN 7.5 REQUIRED)
  target_link_libraries(hetu_C PUBLIC ${CUDNN_LIBRARY_PATH})
  target_include_directories(hetu_C PUBLIC ${CUDNN_INCLUDE_PATH})
endif()

if(${HETU_COMPILE_NCCL})
  find_package(NCCL 2.5 REQUIRED)
  target_link_libraries(hetu_C PUBLIC ${NCCL_LIBRARIES})
  target_include_directories(hetu_C PUBLIC ${NCCL_INCLUDE_DIRS})
endif()

if(${HETU_COMPILE_DNNL})
  find_package(DNNL)
  if(NOT DNNL_FOUND)
    message("dnnl not found, preparing...")
    include(${CMAKE_SOURCE_DIR}/cmake/external/dnnl.cmake)
    target_link_libraries(hetu_C PUBLIC ${DNNL_DLL_PATH})
    target_include_directories(hetu_C PUBLIC ${DNNL_INCLUDE_DIR})
    add_dependencies(hetu_C project_dnnl)
  else()
    target_link_libraries(hetu_C PUBLIC ${DNNL_LIBRARY})
    target_include_directories(hetu_C PUBLIC ${DNNL_BUILD_INCLUDE_DIR})
  endif()
endif()

if(${USE_FLASH_ATTN})
  find_package(CUTLASS)
  if(NOT CUTLASS_FOUND)
    message("cutlass not found, preparing...")
    include(${CMAKE_SOURCE_DIR}/cmake/external/cutlass.cmake)
    target_link_libraries(hetu_C PUBLIC ${CUTLASS_DLL_PATH})
    target_include_directories(hetu_C PUBLIC ${CUTLASS_INCLUDE_DIR})
    message("CU${CUTLASS_INCLUDE_DIR}")
    add_dependencies(hetu_C project_cutlass)
  else()
    target_link_libraries(hetu_C PUBLIC ${CUTLASS_LIBRARY})
    target_include_directories(hetu_C PUBLIC ${CUTLASS_INCLUDE_DIR})
  endif()
endif()

target_compile_options(hetu_C PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       -O3;
                       -std=c++17;
                       --use_fast_math;
                       -U__CUDA_NO_HALF_OPERATORS__,
                       -U__CUDA_NO_HALF_CONVERSIONS__,
                       -U__CUDA_NO_HALF2_OPERATORS__,
                       -U__CUDA_NO_BFLOAT16_CONVERSIONS__,
                       --threads 8
                       >)
