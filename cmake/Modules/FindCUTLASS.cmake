# - Try to find CUTLASS (oneDNN)
# Once done this will define
# CUTLASS_FOUND - System has CUTLASS
# CUTLASS_INCLUDE_DIR - The CUTLASS include directories
# CUTLASS_BUILD_INCLUDE_DIR - CUTLASS include directories in build
# CUTLASS_LIBRARY - The libraries needed to use CUTLASS
# CUTLASS_DEFINITIONS - Compiler switches required for using CUTLASS
if (NOT DEFINED CUTLASS_ROOT)
    set(CUTLASS_ROOT $ENV{CONDA_PREFIX})
endif()
message("CUTLASS_ROOT:${CUTLASS_ROOT}")
find_path(CUTLASS_INCLUDE_DIR
          NAMES cutlass/cutlass.h
          HINTS
          ${CUTLASS_ROOT}
          ${CUTLASS_ROOT}/include
          REQUIRED)
find_library ( CUTLASS_LIBRARY 
               NAMES cutlass HINTS 
               ${CUTLASS_ROOT}
               ${CUTLASS_ROOT}/lib
               ${CUTLASS_ROOT}/lib64 )
message("CUTLASS_INCLUDE_DIR:${CUTLASS_INCLUDE_DIR}")
message("CUTLASS_BUILD_INCLUDE_DIR:${CUTLASS_BUILD_INCLUDE_DIR}")
message("CUTLASS_LIBRARY:${CUTLASS_LIBRARY}")

include ( FindPackageHandleStandardArgs )
find_package_handle_standard_args ( CUTLASS DEFAULT_MSG CUTLASS_LIBRARY CUTLASS_INCLUDE_DIR)
