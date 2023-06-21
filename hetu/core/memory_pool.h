#pragma once

#include "hetu/common/macros.h"
#include "hetu/core/device.h"
#include <mutex>
#include <memory>

namespace hetu {

struct DataPtr {
  void* ptr;
  size_t size;
  Device device;
};

class MemoryPool {
 public:
  MemoryPool() = default;

  virtual DataPtr AllocDataSpace(size_t num_bytes) = 0;

  virtual void FreeDataSpace(DataPtr ptr) = 0;

  virtual Device device() = 0;

protected:
  std::mutex _mtx;  
};

void RegisterMemoryPool(std::shared_ptr<MemoryPool> memory_pool);

std::shared_ptr<MemoryPool> GetMemoryPool(const Device& device);

DataPtr AllocFromMemoryPool(const Device& device, size_t num_bytes);

void FreeToMemoryPool(DataPtr ptr);

} // namespace hetu
