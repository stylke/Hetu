include(ExternalProject)

set(CUTLASS_TAR ${CMAKE_SOURCE_DIR}/third_party/cutlass/cutlass.tar.gz)
set(CUTLASS_SHARED_LIB libcutlass.so)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/third_party)
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E tar xzf ${CUTLASS_TAR} 
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/third_party
  )

