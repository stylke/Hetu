#include "hetu/impl/memory/CUDAMemoryPool.cuh"
#include "hetu/impl/utils/cuda_utils.h"
#include <mutex>

namespace hetu {
namespace impl {

DataPtr CUDAMemoryPool::AllocDataSpace(size_t num_bytes) {
  std::lock_guard<std::mutex> lock(_mtx);
  if (num_bytes == 0)
    return {nullptr, 0, Device(kCUDA, _device_id)};

  auto alignment = get_data_alignment();
  if (num_bytes % alignment != 0)
    num_bytes = ((num_bytes / alignment) + 1) * alignment;
  DeviceIndex prev_id = SetDevice();
  void* ptr;
  CudaMalloc(&ptr, num_bytes);
  ResetDevice(prev_id);
  this->_allocated += num_bytes;
  return {ptr, num_bytes, Device(kCUDA, _device_id)};
}

void CUDAMemoryPool::FreeDataSpace(DataPtr ptr) {
  std::lock_guard<std::mutex> lock(_mtx);
  if (ptr.size == 0)
    return;

  DeviceIndex prev_id = SetDevice();
  CudaFree(ptr.ptr);
  ResetDevice(prev_id);
  this->_allocated -= ptr.size;
}

DeviceIndex CUDAMemoryPool::SetDevice() {
  int cur_id = -1;
  CudaGetDevice(&cur_id);
  if (cur_id != _device_id)
    CudaSetDevice(_device_id);
  return cur_id;
}

void CUDAMemoryPool::ResetDevice(DeviceIndex prev_id) {
  if (prev_id != _device_id)
    CudaSetDevice(prev_id);
}

namespace {

static std::vector<std::shared_ptr<CUDAMemoryPool>> cuda_memory_pools;
static std::once_flag cuda_memory_pool_register_flag;

struct CUDAMemoryPoolRegister {
  CUDAMemoryPoolRegister() {
    std::call_once(cuda_memory_pool_register_flag, []() {
      int32_t num_devices;
      CudaGetDeviceCount(&num_devices);
      for (int32_t i = 0; i < num_devices; i++) {
        auto pool =
          std::make_shared<CUDAMemoryPool>(static_cast<DeviceIndex>(i));
        cuda_memory_pools.push_back(pool);
        RegisterMemoryPool(pool);
      }
    });
  }
};

// static CUDAMemoryPoolRegister cudu_memory_pool_register;

} // namespace

} // namespace impl
} // namespace hetu
