#pragma once

#include "hetu/core/memory_pool.h"
#include "hetu/utils/task_queue.h"

namespace hetu {
namespace impl {

class CUDAMemoryPool final : public MemoryPool {
 public:
  CUDAMemoryPool(DeviceIndex device_id);

  ~CUDAMemoryPool();

  DataPtr AllocDataSpace(size_t num_bytes, const Stream& stream = Stream());

  void BorrowDataSpace(DataPtr data_ptr, DataPtrDeleter deleter);

  void FreeDataSpace(DataPtr data_ptr);

  void MarkDataSpaceUsedByStream(DataPtr data_ptr, const Stream& stream);

  void MarkDataSpacesUsedByStream(DataPtrList& data_ptrs, const Stream& stream);

  std::future<void> WaitDataSpace(DataPtr data_ptr, bool async = true);

  inline size_t get_data_alignment() const noexcept {
    return 256;
  }

 private:
  struct CudaDataPtrInfo {
    size_t num_bytes;
    Stream alloc_stream;
    std::unordered_set<Stream> used_streams;

    CudaDataPtrInfo(size_t num_bytes_, Stream alloc_stream_)
    : num_bytes(num_bytes_), alloc_stream{std::move(alloc_stream_)} {}
  };

  size_t _allocated{0};
  size_t _peak_allocated{0};
  std::unordered_map<const void*, CudaDataPtrInfo> _data_ptr_info;
  std::vector<int> _free_stream_flags{HT_NUM_STREAMS_PER_DEVICE, 0};
  std::unique_ptr<TaskQueue> _free_stream_watcher;
};

} // namespace impl
} // namespace hetu
