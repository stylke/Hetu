# - Try to find DNNL(MKL-DNN)
# Once done this will define
# DNNL_FOUND - System has DNNL
# DNNL_INCLUDE_DIR - The DNNL include directories
# DNNL_BUILD_INCLUDE_DIR - DNNL include directories in build
# DNNL_LIBRARY - The libraries needed to use DNNL
# DNNL_DEFINITIONS - Compiler switches required for using DNNL
set(MKL_ROOT /root/hetu-pytest/others/oneDNN/)
set(MKL_BUILD ${MKL_ROOT}/build)
find_path ( DNNL_INCLUDE_DIR dnnl.h HINTS ${MKL_ROOT}/include )
find_path ( DNNL_BUILD_INCLUDE_DIR dnnl_config.h HINTS ${MKL_BUILD}/include )
find_library ( DNNL_LIBRARY NAMES dnnl mkldnn HINTS ${MKL_BUILD}/src )
message("DNNL_LIBRARY = ${DNNL_LIBRARY}")

include ( FindPackageHandleStandardArgs )
find_package_handle_standard_args ( MKL DEFAULT_MSG DNNL_LIBRARY DNNL_INCLUDE_DIR DNNL_BUILD_INCLUDE_DIR )
