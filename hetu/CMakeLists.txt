# Set source directory
set(HETU_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
file(GLOB_RECURSE HETU_SRC ${HETU_SRC_DIR}/hetu/*.cc)

# Add CUDA source files if enabled
if(${HETU_COMPILE_CUDA})
  add_definitions(-DHETU_CUDA_ENABLED)
  file(GLOB_RECURSE HETU_CU_SRC ${HETU_SRC_DIR}/hetu/*.cu)
  list(APPEND HETU_SRC ${HETU_CU_SRC})
endif()

# Exclude test files
list(FILTER HETU_SRC EXCLUDE REGEX "${HETU_SRC_DIR}/hetu/test/*")

# Add main library
add_library(hetu_C SHARED ${HETU_SRC})

# Set CMake prefix path
# A100
set(CMAKE_PREFIX_PATH /home/pkuhetu/Hetu-dev/fused_kernel/.local)
# 4090
# set(CMAKE_PREFIX_PATH /home/gehao/lhy/local)
include(../cmake/grpc.cmake)

# Get proto file paths
get_filename_component(hetu_proto "${HETU_SRC_DIR}/hetu/impl/communication/protos/heturpc.proto" ABSOLUTE)
get_filename_component(hetu_proto_path "${hetu_proto}" PATH)

# Generated sources
file(MAKE_DIRECTORY "${HETU_SRC_DIR}/hetu/impl/communication/rpc")
set(hetu_proto_srcs "${HETU_SRC_DIR}/hetu/impl/communication/rpc/heturpc.pb.cc")
set(hetu_proto_hdrs "${HETU_SRC_DIR}/hetu/impl/communication/rpc/heturpc.pb.h")
set(hetu_grpc_srcs "${HETU_SRC_DIR}/hetu/impl/communication/rpc/heturpc.grpc.pb.cc")
set(hetu_grpc_hdrs "${HETU_SRC_DIR}/hetu/impl/communication/rpc/heturpc.grpc.pb.h")

add_custom_command(
  OUTPUT "${hetu_proto_srcs}" "${hetu_proto_hdrs}" "${hetu_grpc_srcs}" "${hetu_grpc_hdrs}"
  COMMAND ${_PROTOBUF_PROTOC}
  ARGS --grpc_out "${HETU_SRC_DIR}/hetu/impl/communication/rpc"
    --cpp_out "${HETU_SRC_DIR}/hetu/impl/communication/rpc"
    -I "${hetu_proto_path}"
    --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
    "${hetu_proto}"
  DEPENDS "${hetu_proto}"
)

# Execute Python gRPC tools
execute_process(
  COMMAND python -m grpc_tools.protoc -I./protos --python_out=../../../python_refactor/hetu/rpc/  
  --pyi_out=../../../python_refactor/hetu/rpc/ --grpc_python_out=../../../python_refactor/hetu/rpc/  ./protos/heturpc.proto
  WORKING_DIRECTORY ${HETU_SRC_DIR}/hetu/impl/communication
)

# Include generated *.pb.h files
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

# Create gRPC proto library with -fPIC
add_library(hetu_grpc_proto STATIC
  ${hetu_grpc_srcs}
  ${hetu_grpc_hdrs}
  ${hetu_proto_srcs}
  ${hetu_proto_hdrs}
)
set_target_properties(hetu_grpc_proto PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Link libraries
target_link_libraries(hetu_grpc_proto
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF})

message("_REFLECTION:${_REFLECTION}")
message("_GRPC_GRPCPP:${_GRPC_GRPCPP}")
message("_PROTOBUF_LIBPROTOBUF:${_PROTOBUF_LIBPROTOBUF}")

# Link hetu_C with gRPC proto and other dependencies
target_link_libraries(hetu_C 
  PUBLIC hetu_grpc_proto
  PUBLIC absl::flags
  PUBLIC absl::flags_parse
  PUBLIC ${_REFLECTION}
  PUBLIC ${_GRPC_GRPCPP}
  PUBLIC ${_PROTOBUF_LIBPROTOBUF})

target_include_directories(hetu_C PUBLIC ${HETU_SRC_DIR})

# OpenMP support
if(${HETU_COMPILE_OMP})
  target_link_libraries(hetu_C PUBLIC MPI::MPI_CXX)
endif()

# CUDA support
if(${HETU_COMPILE_CUDA})
  target_link_libraries(hetu_C PUBLIC CUDA::cudart CUDA::cublas CUDA::cusparse CUDA::curand CUDA::cuda_driver)
  set_property(TARGET hetu_C PROPERTY CUDA_ARCHITECTURES ${HETU_CUDA_ARCH})
endif()

# cuDNN support
if(${HETU_COMPILE_CUDNN})
  find_package(CUDNN 7.5 REQUIRED)
  target_link_libraries(hetu_C PUBLIC ${CUDNN_LIBRARY_PATH})
  target_include_directories(hetu_C PUBLIC ${CUDNN_INCLUDE_PATH})
endif()

# NCCL support
if(${HETU_COMPILE_NCCL})
  find_package(NCCL 2.19)
  if (NOT NCCL_FOUND)
    message("nccl 2.19+ not found, preparing...")
    include(${CMAKE_SOURCE_DIR}/cmake/external/nccl.cmake)
    target_link_libraries(hetu_C PUBLIC ${NCCL_LIBRARIES_SRC})
    target_include_directories(hetu_C PUBLIC ${NCCL_INCLUDE_DIRS})
  else()
    target_link_libraries(hetu_C PUBLIC ${NCCL_LIBRARIES})
    target_include_directories(hetu_C PUBLIC ${NCCL_INCLUDE_DIRS})
  endif()
endif()

# NVML support
if(${HETU_COMPILE_NVML})
  find_package(NVML REQUIRED)
  target_link_libraries(hetu_C PUBLIC ${NVML_LIBRARIES})
  target_include_directories(hetu_C PUBLIC ${NVML_INCLUDE_DIRS})
endif()

# DNNL support
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

# Flash Attention support
if(${USE_FLASH_ATTN})
  find_package(CUTLASS)
  if(NOT CUTLASS_FOUND)
    message("cutlass not found, preparing...")
    include(${CMAKE_SOURCE_DIR}/cmake/external/cutlass.cmake)
    target_include_directories(hetu_C PUBLIC ${CUTLASS_INCLUDE_DIR}) 
    target_link_libraries(hetu_C PUBLIC ${CUTLASS_DLL_PATH})
  endif()

  find_package(FLASH)
  if(NOT FLASH_FOUND)
    message("flash attention not found, preparing...")
    include(${CMAKE_SOURCE_DIR}/cmake/external/flashattn.cmake)
    target_link_libraries(hetu_C PUBLIC ${FLASH_DLL_PATH})
    message("flash attention not found:${FLASH_INCLUDE_DIR}")
    target_include_directories(hetu_C PUBLIC ${FLASH_INCLUDE_DIR})
    add_dependencies(hetu_C project_flashattn)
  else()
    target_link_libraries(hetu_C PUBLIC ${FLASH_LIBRARY})
    target_include_directories(hetu_C PUBLIC ${FLASH_INCLUDE_DIR})
  endif()
endif()

# BitsandBytes support
message("preparing bitsandbytes")
include(${CMAKE_SOURCE_DIR}/cmake/external/bitsandbytes.cmake)
target_link_libraries(hetu_C PUBLIC ${BNB_LIB_PATH})
target_include_directories(hetu_C PUBLIC ${BNB_INCLUDE_DIR})
add_dependencies(hetu_C project_bitsandbytes)

# Compile options for CUDA
target_compile_options(hetu_C PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
  -O3;
  -std=c++17;
  --use_fast_math;
  -D__CUDA_NO_HALF_OPERATORS__;
  -D__CUDA_NO_HALF_CONVERSIONS__;
  -D__CUDA_NO_HALF2_OPERATORS__;
  -D__CUDA_NO_BFLOAT16_OPERATORS__;
  -D__CUDA_NO_BFLOAT16_CONVERSIONS__;
  --threads 8
>)
