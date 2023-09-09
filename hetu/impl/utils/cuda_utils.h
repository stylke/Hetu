#pragma once
#define __CUDA_NO_HALF_OPERATORS__

#include "hetu/common/macros.h"
#include "hetu/core/device.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

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
// memory
#define CudaMalloc(ptr, size) CUDA_CALL(cudaMalloc(ptr, size))
#define CudaFree(ptr) CUDA_CALL(cudaFree(ptr))
#define CudaMemcpy(dst_ptr, src_ptr, size, direction)                          \
  CUDA_CALL(cudaMemcpy(dst_ptr, src_ptr, size, direction))
#define CudaMemcpyAsync(dst_ptr, src_ptr, size, direction, stream)             \
  CUDA_CALL(cudaMemcpyAsync(dst_ptr, src_ptr, size, direction, stream))
#define CudaMemcpyPeerAsync(dst_ptr, dst_dev, src_ptr, src_dev, size, stream)  \
  CUDA_CALL(                                                                   \
    cudaMemcpyPeerAsync(dst_ptr, dst_dev, src_ptr, src_dev, size, stream))
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
#define CudaEventCreate(ptr) CUDA_CALL(cudaEventCreate(ptr))
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

} // namespace cuda
} // namespace hetu
