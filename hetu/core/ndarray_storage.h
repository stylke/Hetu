#pragma once

#include "hetu/core/memory_pool.h"
#include "hetu/core/device.h"
#include <functional>

namespace hetu {

class NDArrayStorage {
 public:
  using DeleterFn = std::function<void(DataPtr)>;
  NDArrayStorage(size_t size, Device device)
  : _size(size), _device(device), _ptr(AllocFromMemoryPool(device, size)) {}

  NDArrayStorage(void* p, size_t size, const Device& device, DeleterFn deleter)
  : _size(size), _device(device), _deleter(deleter) {
    HT_ASSERT(deleter)
      << "Deleter fn must not be empty when borrowing storages";
    HT_ASSERT(p && size > 0) << "Borrowing an empty storage is not allowed";
    _ptr = {p, size, device};
  }

  ~NDArrayStorage() {
    if (_deleter)
      _deleter(_ptr);
    else
      FreeToMemoryPool(_ptr);
  }

  inline size_t size() const {
    return _size;
  }

  inline void* mutable_data() {
    return _ptr.ptr;
  }

  inline const void* data() const {
    return _ptr.ptr;
  }

  inline Device device() const {
    return _device;
  }

 protected:
  size_t _size;
  Device _device;
  DataPtr _ptr;
  DeleterFn _deleter;
  bool _writable;
};

} // namespace hetu
