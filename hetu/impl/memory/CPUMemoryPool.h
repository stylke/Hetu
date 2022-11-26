#pragma once

#include "hetu/core/memory_pool.h"

namespace hetu {
namespace impl {

void* cpu_alloc(size_t num_bytes, size_t alignment);
void cpu_free(void* ptr);

class CPUMemoryPool final : public MemoryPool {
 public:
  CPUMemoryPool() = default;

  DataPtr AllocDataSpace(size_t num_bytes);

  void FreeDataSpace(DataPtr ptr);

  inline size_t get_data_alignment() const noexcept {
    return 16;
  }

  inline Device device() {
    return {kCPU};
  }

 private:
  size_t _allocated = 0;
};

} // namespace impl
} // namespace hetu
