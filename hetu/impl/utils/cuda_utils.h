#pragma once

#include "hetu/common/macros.h"
#include "hetu/core/device.h"
#include "hetu/core/ndarray.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <type_traits>
#include <functional>

namespace hetu {
namespace cuda {

DECLARE_HT_EXCEPTION(cuda_error);

} // namespace cuda
} // namespace hetu

#define HT_MAX_GPUS_COMPILE_TIME (16)
#define HT_MAX_GPUS_RUN_TIME (8)
#define HT_DEFAULT_NUM_THREADS_PER_BLOCK (1024)

#define CUDA_CALL(f)                                                           \
  for (cudaError_t status = (f); status != cudaSuccess; status = cudaSuccess)  \
  __HT_FATAL_SILENT(hetu::cuda::cuda_error)                                    \
    << "Cuda call " << #f << " failed: " << cudaGetErrorString(status)

/******************************************************
 * Some useful wrappers for CUDA functions
 ******************************************************/
// device
#define CudaSetDevice(device) CUDA_CALL(cudaSetDevice(device))
#define CudaGetDevice(ptr) CUDA_CALL(cudaGetDevice(ptr))
#define CudaGetDeviceCount(ptr) CUDA_CALL(cudaGetDeviceCount(ptr))
#define CudaDeviceGetAttribute(ptr, attr, stream)                              \
  CUDA_CALL(cudaDeviceGetAttribute(ptr, attr, stream))
#define CudaGetDeviceProperties(ptr, device)                                   \
  CUDA_CALL(cudaGetDeviceProperties(ptr, device))
// memory
#define CudaMalloc(ptr, size) CUDA_CALL(cudaMalloc(ptr, size))
#define CudaMallocAsync(ptr, size, stream)                                     \
  CUDA_CALL(cudaMallocAsync(ptr, size, stream))
#define CudaFree(ptr) CUDA_CALL(cudaFree(ptr))
#define CudaFreeAsync(ptr, stream) CUDA_CALL(cudaFreeAsync(ptr, stream))
#define CudaMemset(ptr, value, size) CUDA_CALL(cudaMemset(ptr, value, size))
#define CudaMemsetAsync(ptr, value, size, stream)                              \
  CUDA_CALL(cudaMemsetAsync(ptr, value, size, stream))
#define CudaMemcpy(dst_ptr, src_ptr, size, direction)                          \
  CUDA_CALL(cudaMemcpy(dst_ptr, src_ptr, size, direction))
#define CudaMemcpyAsync(dst_ptr, src_ptr, size, direction, stream)             \
  CUDA_CALL(cudaMemcpyAsync(dst_ptr, src_ptr, size, direction, stream))
#define CudaMemcpyPeerAsync(dst_ptr, dst_dev, src_ptr, src_dev, size, stream)  \
  CUDA_CALL(                                                                   \
    cudaMemcpyPeerAsync(dst_ptr, dst_dev, src_ptr, src_dev, size, stream))
#define CudaMemsetAsync(dst_ptr, val, size, stream)                            \
  CUDA_CALL(cudaMemsetAsync(dst_ptr, val, size, stream))
// stream
#define CudaStreamCreate(ptr) CUDA_CALL(cudaStreamCreate(ptr))
#define CudaStreamCreateWithFlags(ptr, flags)                                  \
  CUDA_CALL(cudaStreamCreateWithFlags(ptr, flags))
#define CudaStreamCreateWithPriority(ptr, flags, priority)                     \
  CUDA_CALL(cudaStreamCreateWithPriority(ptr, flags, priority))
#define CudaStreamDestroy(stream) CUDA_CALL(cudaStreamDestroy(stream))
#define CudaStreamSynchronize(stream) CUDA_CALL(cudaStreamSynchronize(stream))
#define CudaStreamWaitEvent(stream, event, flags)                              \
  CUDA_CALL(cudaStreamWaitEvent(stream, event, flags))
// event
#define CudaEventCreate(ptr, flags) CUDA_CALL(cudaEventCreate(ptr, flags))
#define CudaEventDestroy(event) CUDA_CALL(cudaEventDestroy(event))
#define CudaEventElapsedTime(ptr, start, end)                                  \
  CUDA_CALL(cudaEventElapsedTime(ptr, start, end))
#define CudaEventQuery(event) CUDA_CALL(cudaEventQuery(event))
#define CudaEventRecord(event, stream) CUDA_CALL(cudaEventRecord(event, stream))
#define CudaEventSynchronize(event) CUDA_CALL(cudaEventSynchronize(event))

namespace hetu {
namespace cuda {

class CUDADeviceGuard final {
 public:
  CUDADeviceGuard(int32_t device_id) : _cur_device_id(device_id) {
    if (_cur_device_id != -1) {
      CudaGetDevice(&_prev_device_id);
      if (_prev_device_id != _cur_device_id)
        CudaSetDevice(_cur_device_id);
    }
  }

  ~CUDADeviceGuard() {
    if (_prev_device_id != -1 && _prev_device_id != _cur_device_id)
      CudaSetDevice(_prev_device_id);
  }

  // disable copy constructor and move constructor
  CUDADeviceGuard(const CUDADeviceGuard& other) = delete;
  CUDADeviceGuard& operator=(const CUDADeviceGuard& other) = delete;
  CUDADeviceGuard(CUDADeviceGuard&& other) = delete;
  CUDADeviceGuard& operator=(CUDADeviceGuard&& other) = delete;

 private:
  int32_t _prev_device_id{-1};
  int32_t _cur_device_id;
};

// A helper buffer to pass arguments like shapes, strides, axes to cuda kernels
// Remember that kernel arguments are stored in constant memory, 
// which may have a slower access speed than global memory. 
// If that matters, use `to_int64_ndarray` instead.
template <std::size_t N>
struct Int64Buffer {
  int64_t values[N];
  __forceinline__ __host__ __device__ const int64_t&
  operator[](size_t i) const {
    return values[i];
  }
  __forceinline__ __host__ __device__ int64_t& operator[](size_t i) {
    return values[i];
  }
};

template <std::size_t N>
inline Int64Buffer<N> to_int64_buffer(const std::vector<int64_t>& vec) {
  HT_VALUE_ERROR_IF(vec.size() > N)
    << "Trying to set " << vec.size() << " values in buffer with size " << N;
  Int64Buffer<N> ret;
  std::copy(vec.begin(), vec.end(), &(ret.values[0]));
  return ret;
}

template <std::size_t N, class InputIt>
inline Int64Buffer<N> to_int64_buffer(InputIt first, InputIt last) {
  auto n = std::distance(first, last);
  HT_VALUE_ERROR_IF(n > N) << "Trying to set " << n
                           << " values in buffer with size " << N;
  Int64Buffer<N> ret;
  std::copy(first, last, &(ret.values[0]));
  return ret;
}

// Helper functions to copy shapes, strides, or axes into NDArrays (blocking)
NDArray to_int64_ndarray(const std::vector<int64_t>& vec,
                         DeviceIndex device_id);
NDArray to_int64_ndarray(const int64_t* from, size_t n, DeviceIndex device_id);

} // namespace cuda
} // namespace hetu
