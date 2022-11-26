#pragma once

#include "hetu/core/memory_pool.h"

namespace hetu {
namespace impl {

class CUDAMemoryPool final : public MemoryPool {
 public:
  CUDAMemoryPool(DeviceIndex device_id) : MemoryPool(), _device_id(device_id) {}

  DataPtr AllocDataSpace(size_t num_bytes);

  void FreeDataSpace(DataPtr ptr);

  inline Device device() {
    return {kCUDA, _device_id};
  }

  inline size_t get_data_alignment() const noexcept {
    return 256;
  }

 private:
  DeviceIndex SetDevice();
  void ResetDevice(DeviceIndex prev_id);

  const DeviceIndex _device_id;
  size_t _allocated = 0;
};

} // namespace impl
} // namespace hetu
