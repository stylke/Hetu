#pragma once

#include "hetu/common/macros.h"
#include "hetu/core/device.h"
#include "hetu/core/stream.h"
#include <memory>
#include <mutex>
#include <future>

namespace hetu {

struct DataPtr {
  void* ptr;
  size_t size;
  Device device;
  uint64_t id; // id provided by the memory pool
};

using DataPtrList = std::vector<DataPtr>;

std::ostream& operator<<(std::ostream&, const DataPtr&);

using DataPtrDeleter = std::function<void(DataPtr)>;

class MemoryPool {
 public:
  MemoryPool(Device device, std::string name)
  : _device{std::move(device)}, _name{std::move(name)} {}

  virtual DataPtr AllocDataSpace(size_t num_bytes,
                                 const Stream& stream = Stream()) = 0;

  virtual DataPtr BorrowDataSpace(void* ptr, size_t num_bytes,
                                  DataPtrDeleter deleter) = 0;

  virtual void FreeDataSpace(DataPtr data_ptr) = 0;

  virtual void MarkDataSpaceUsedByStream(DataPtr data_ptr,
                                         const Stream& stream) = 0;

  virtual void MarkDataSpacesUsedByStream(DataPtrList& data_ptrs,
                                          const Stream& stream) = 0;

  virtual std::future<void> WaitDataSpace(DataPtr data_ptr,
                                          bool async = true) = 0;

  const Device& device() const {
    return _device;
  }

  const std::string& name() const {
    return _name;
  }

protected:
  uint64_t next_id() {
    // Only called on alloc or borrow, so the lock has been acquired.
    return _next_id++;
  }
  
  const Device _device;
  const std::string _name;
  uint64_t _next_id;
  std::mutex _mtx;
};

void RegisterMemoryPool(std::shared_ptr<MemoryPool> memory_pool);

std::shared_ptr<MemoryPool> GetMemoryPool(const Device& device);

DataPtr AllocFromMemoryPool(const Device& device, size_t num_bytes,
                            const Stream& stream = Stream());

DataPtr BorrowToMemoryPool(const Device& device, void* ptr, size_t num_bytes,
                           DataPtrDeleter deleter);

void FreeToMemoryPool(DataPtr ptr);

} // namespace hetu
