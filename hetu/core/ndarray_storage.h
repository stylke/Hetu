#pragma once

#include "hetu/core/memory_pool.h"
#include "hetu/core/device.h"
#include <functional>

namespace hetu {

class NDArrayStorage {
 public:
  
  NDArrayStorage(DataPtr ptr): _ptr(ptr) {}

  NDArrayStorage(void* p, size_t size, const Device& device,
                 DataPtrDeleter deleter)
  : _ptr{DataPtr{p, size, device}} {
    GetMemoryPool(device)->BorrowDataSpace(_ptr, std::move(deleter));
  }

  ~NDArrayStorage() {
    FreeToMemoryPool(_ptr);
  }

  inline size_t size() const {
    return _ptr.size;
  }

  inline void* mutable_data() {
    return _ptr.ptr;
  }

  inline const void* data() const {
    return _ptr.ptr;
  }

  inline const Device& device() const {
    return _ptr.device;
  }

  inline Device& device() {
    return _ptr.device;
  }

  inline DataPtr data_ptr() const {
    return _ptr;
  }

 protected:
  DataPtr _ptr;
  bool _writable{true};
};

} // namespace hetu
