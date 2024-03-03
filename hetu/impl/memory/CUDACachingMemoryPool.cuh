#pragma once

#include "hetu/impl/memory/CUDAMemoryPool.cuh"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/utils/task_queue.h"
#include "hetu/utils/emhash7_hashmap.h"
#include "hetu/utils/robin_hood_hashing.h"
#include <deque>

namespace hetu {
namespace impl {

class CUDACachingMemoryPool final : public CUDAMemoryPool {
 public:
  CUDACachingMemoryPool(DeviceIndex device_id);

  ~CUDACachingMemoryPool();

  DataPtr AllocDataSpace(size_t num_bytes,
                         const Stream& stream = Stream()) override;

  DataPtr BorrowDataSpace(void* ptr, size_t num_bytes,
                          DataPtrDeleter deleter,
                          const Stream& stream = Stream()) override;

  void FreeDataSpace(DataPtr data_ptr) override;

  void EmptyCache() override;

  void MarkDataSpaceUsedByStream(DataPtr data_ptr,
                                 const Stream& stream) override;

  void MarkDataSpacesUsedByStream(DataPtrList& data_ptrs,
                                  const Stream& stream) override;

  std::future<void> WaitDataSpace(DataPtr data_ptr, bool async = true) override;

  void PrintSummary() override;

 private:
  
  // Status after allocation (AllocDataSpace):
  // (1) status = OCCUPIED_BY_ALLOC_STREAM.
  // Status transition of MarkUsedBy (between AllocDataSpace and FreeDataSpace):
  // (1) if only used by the alloc stream, status = OCCUPIED_BY_ALLOC_STREAM;
  // (2) else, status = OCCUPIED_BY_MULTI_STREAMS.
  // Status transition of FreeDataSpace (freed by user):
  // (1) if status == OCCUPIED_BY_ALLOC_STREAM, then status = AVAILABLE_FOR_ALLOC_STREAM;
  // (2) else (status == OCCUPIED_BY_MULTI_STREAMS), then status = UNAVAILABLE_UNTIL_FREE;
  enum class OccupationStatus : int8_t {
    OCCUPIED_BY_ALLOC_STREAM = 0,
    OCCUPIED_BY_MULTI_STREAMS,
    AVAILABLE_FOR_ALLOC_STREAM,
    UNAVAILABLE_UNTIL_FREE
  };

  struct CudaDataPtrInfo {
    void* ptr;
    size_t num_bytes;
    PackedStreamId alloc_stream;
    std::unordered_set<PackedStreamId> used_streams;
    DataPtrDeleter deleter;
    OccupationStatus status;
    mempool_clock_t alloc_at;
    mempool_clock_t free_at;
    uint32_t free_event_cnt;

    CudaDataPtrInfo(void* ptr_, size_t num_bytes_, const Stream& alloc_stream_,
                    mempool_clock_t alloc_at_, DataPtrDeleter deleter_ = {})
    : ptr(ptr_),
      num_bytes(num_bytes_),
      alloc_stream(alloc_stream_.pack()),
      alloc_at(alloc_at_), 
      deleter(deleter_),
      free_at(0), 
      free_event_cnt(0) {
      if (!alloc_stream_.is_blocking())
        used_streams.insert(alloc_stream);
      status = OccupationStatus::OCCUPIED_BY_ALLOC_STREAM;
    }

    inline void insert_used_stream(PackedStreamId used_stream) {
      if (used_stream != alloc_stream) {
        used_streams.insert(used_stream);
        status = OccupationStatus::OCCUPIED_BY_MULTI_STREAMS;
      }
    }

    inline bool allocated() const noexcept {
      return alloc_at >= free_at;
    }
  };

  struct DataPtrLookupTable {
    emhash7::HashMap<size_t, std::vector<std::tuple<DataPtrId, void*, size_t>>,
                     robin_hood::hash<size_t>>
      table;
    DataPtrLookupTable(size_t capacity = 1024) {
      table.reserve(capacity);
    }
  };

  bool FindAvailableFromLookupTable(size_t num_bytes,
                                    DataPtrLookupTable& lookup_table,
                                    bool remove_if_found, DataPtr& ret);

  void InsertAvailableToLookupTable(const DataPtr& data_ptr,
                                    DataPtrLookupTable& lookup_table);

  void EmptyCacheInLookupTable(DataPtrLookupTable& lookup_table,
                               bool maybe_allocated);
  
  void WatchEvents();

  // Info of all data pointers that are currently in used (allocated)
  emhash7::HashMap<DataPtrId, CudaDataPtrInfo> _data_ptr_info;
  // Cached data pointers that are available for specific streams
  emhash7::HashMap<PackedStreamId, std::unique_ptr<DataPtrLookupTable>>
    _available_for_single_stream;
  // Cached data pointers that are available for all stream
  std::unique_ptr<DataPtrLookupTable> _available_for_all_streams;
  // Events to indicate whether marked usages have finished
  emhash7::HashMap<PackedStreamId,
                   std::deque<std::tuple<std::unique_ptr<CUDAEvent>, DataPtrId,
                                         mempool_clock_t>>>
    _free_events;

  size_t _allocated{0};
  size_t _reserved{0};
  size_t _peak_reserved{0};
  uint64_t _alloc_cnt{0};
  uint64_t _cuda_malloc_cnt{0};
  uint64_t _free_cnt{0};
  uint64_t _mark_cnt{0};
};

} // namespace impl
} // namespace hetu
