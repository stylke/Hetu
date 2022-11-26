cmake_minimum_required(VERSION 3.17)

set(HETU_COMPILE_CUDA ON)

project(Hetu CXX)

include(cmake/config.cmake)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_CXX_FLAGS "-O3 -Wall")

find_package(OpenMP)
if(OPENMP_FOUND)
  message(STATUS "OpenMP Found")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  add_definitions(-DHETU_OPENMP_ENABLED)
endif()

# TODO: add cuda option
set(HETU_COMPILE_CUDA ON)
if(${HETU_COMPILE_CUDA})
  set(CMAKE_CUDA_COMPILER ${CUDAToolkit_ROOT}/bin/nvcc)
  find_package(CUDAToolkit REQUIRED)
  enable_language(CUDA)
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_LESS "10.1")
    message(FATAL_ERROR "Required CUDA version >= 10.1, while current CUDA version is ${CMAKE_CUDA_COMPILER_VERSION}")
  endif()
  include_directories(SYSTEM ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

# TODO: move the following block to the hetu directory
######################################################
set(HETU_SRC_DIR ${CMAKE_SOURCE_DIR}/../..)
file(GLOB_RECURSE HETU_SRC ${HETU_SRC_DIR}/hetu/*.cc)
list(FILTER HETU_SRC EXCLUDE REGEX "${HETU_SRC_DIR}/hetu/test/.*.cc")
if(${HETU_COMPILE_CUDA})
  add_definitions(-DHETU_CUDA_ENABLED)
  file(GLOB_RECURSE HETU_CU_SRC ${HETU_SRC_DIR}/hetu/*.cu)
  list(FILTER HETU_CU_SRC EXCLUDE REGEX "${HETU_SRC_DIR}/hetu/test/.*.cu")
  list(APPEND HETU_SRC ${HETU_CU_SRC})
endif()
add_library(hetu_C SHARED ${HETU_SRC})
target_include_directories(hetu_C PUBLIC ${HETU_SRC_DIR})

find_package(NCCL 2.5 REQUIRED)
message("NCCL_LIBRARIES: ${NCCL_LIBRARIES}")
message("NCCL_INCLUDE_DIRS: ${NCCL_INCLUDE_DIRS}")
target_link_libraries(hetu_C PUBLIC ${NCCL_LIBRARIES})
target_include_directories(hetu_C PUBLIC ${NCCL_INCLUDE_DIRS})

find_package(MPI 3.1 REQUIRED)
target_link_libraries(hetu_C PUBLIC MPI::MPI_CXX)

if(${HETU_COMPILE_CUDA})
  find_package(CUDNN 7.5 REQUIRED)

  target_link_libraries(hetu_C PUBLIC CUDA::cudart CUDA::cublas CUDA::cusparse CUDA::curand CUDA::cuda_driver)
  target_link_libraries(hetu_C PUBLIC ${CUDNN_LIBRARY_PATH})
  target_include_directories(hetu_C PUBLIC ${CUDNN_INCLUDE_PATH})

  if (${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL "11")
    set(HETU_CUDA_ARCH 52 60 61 70 75 80 86)
  else()
    set(HETU_CUDA_ARCH 50 52 60 61 70 75)
  endif()
  set_property(TARGET hetu_C PROPERTY CUDA_ARCHITECTURES ${HETU_CUDA_ARCH})
endif()
######################################################

file(GLOB HETU_TEST_SRC ${HETU_SRC_DIR}/hetu/test/test_*.cc)
foreach(test ${HETU_TEST_SRC})
  get_filename_component(TName ${test} NAME_WE)
  add_executable(${TName} ${test})
  target_include_directories(${TName} PUBLIC ${HETU_SRC_DIR})
  target_link_libraries(${TName} PUBLIC hetu_C)
endforeach()
