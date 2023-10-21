include(ExternalProject)

set(CUTLASS_TAR ${CMAKE_SOURCE_DIR}/third_party/cutlass/cutlass.tar.gz)
set(CUTLASS_SHARED_LIB libcutlass.so)

if(${USE_FLASH_ATTN})
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/third_party)
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E tar xzf ${CUTLASS_TAR} 
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/third_party
  )

  set(CUTLASS_SOURCE ${CMAKE_CURRENT_BINARY_DIR}/third_party/cutlass)
  set(CUTLASS_INCLUDE_DIR ${CUTLASS_SOURCE}/include)
  set(CUTLASS_LIB_DIR ${CUTLASS_SOURCE}/lib)
  set(CUTLASS_DLL_PATH ${CUTLASS_LIB_DIR}/${CUTLASS_SHARED_LIB})
  set(CUTLASS_CMAKE_EXTRA_ARGS)

  ExternalProject_Add(project_cutlass
    PREFIX cutlass
    # PATCH_COMMAND ${MKLDNN_PATCH_DISCARD_COMMAND} COMMAND ${CUTLASS_PATCH_COMMAND}
    SOURCE_DIR ${CUTLASS_SOURCE}
    CMAKE_ARGS -DCUTLASS_NVCC_ARCHS=80
  )
  link_directories(${CUTLASS_LIB_DIR})
endif()
