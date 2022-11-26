#include "hetu/impl/memory/CPUMemoryPool.h"
#include <mutex>

namespace hetu {
namespace impl {

void* cpu_alloc(size_t num_bytes, size_t alignment) {
  if (num_bytes == 0)
    return nullptr;
  HT_ASSERT_GE(num_bytes, 0) << "Invalid number of bytes: " << num_bytes;

  void* ptr;
  int err = posix_memalign(&ptr, alignment, num_bytes);
  HT_ASSERT_EQ(err, 0) << "Failed to allocate " << num_bytes
                       << " bytes of memory. Error: " << strerror(err);
  return ptr;
}

void cpu_free(void* ptr) {
  free(ptr);
}

DataPtr CPUMemoryPool::AllocDataSpace(size_t num_bytes) {
  if (num_bytes == 0)
    return {nullptr, 0, Device(kCPU)};

  auto alignment = get_data_alignment();
  void* ptr = cpu_alloc(num_bytes, alignment);
  this->_allocated += num_bytes;
  return {ptr, num_bytes, Device(kCPU)};
}

void CPUMemoryPool::FreeDataSpace(DataPtr ptr) {
  if (ptr.size == 0)
    return;

  cpu_free(ptr.ptr);
  this->_allocated -= ptr.size;
}

namespace {

static std::shared_ptr<CPUMemoryPool> cpu_memory_pool;
static std::once_flag cpu_memory_pool_register_flag;

struct CPUMemoryPoolRegister {
  CPUMemoryPoolRegister() {
    std::call_once(cpu_memory_pool_register_flag, []() {
      cpu_memory_pool = std::make_shared<CPUMemoryPool>();
      RegisterMemoryPool(cpu_memory_pool);
    });
  }
};

static CPUMemoryPoolRegister cpu_memory_pool_register;

} // namespace

} // namespace impl
} // namespace hetu
