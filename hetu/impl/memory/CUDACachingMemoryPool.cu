#include "hetu/impl/memory/CUDACachingMemoryPool.cuh"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/cuda_utils.h"
#include <mutex>

namespace hetu {
namespace impl {

namespace {

inline static std::string _make_name(DeviceIndex device_id) {
  return "CUDACachingMemPool(" + std::to_string(static_cast<int>(device_id)) +
    ")";
}

} // namespace

CUDACachingMemoryPool::CUDACachingMemoryPool(DeviceIndex device_id)
: CUDAMemoryPool(device_id, _make_name(device_id)) {
  _data_ptr_info.reserve(8192);
  _available_for_single_stream.reserve(HT_NUM_STREAMS_PER_DEVICE);
  _available_for_all_streams.reset(new DataPtrLookupTable());
}

CUDACachingMemoryPool::~CUDACachingMemoryPool() {
  // TODO: free the memory instead of let the OS collect them
}

DataPtr CUDACachingMemoryPool::AllocDataSpace(size_t num_bytes,
                                              const Stream& stream) {
  HT_VALUE_ERROR_IF(!stream.device().is_cuda())
    << "Cuda arrays must be allocated on cuda streams. Got " << stream;
  if (num_bytes == 0)
    return DataPtr{nullptr, 0, device(), static_cast<uint64_t>(-1)};
  PackedStreamId packed_stream_id = stream.pack();
  std::lock_guard<std::mutex> lock(_mtx);

  WatchEvents();

  uint64_t alloc_at = next_clock();
  DataPtr data_ptr;
  data_ptr.device = device();
  auto alignment = get_data_alignment();
  size_t aligned_num_bytes = DIVUP(num_bytes, alignment) * alignment;
  bool found_avaiable = false;

  // Find among data spaces that are available only for this stream.
  auto it = _available_for_single_stream.find(packed_stream_id);
  if (it == _available_for_single_stream.end()) {
    // It might be useful later, insert anyway.
    auto insertion = _available_for_single_stream.emplace(
      packed_stream_id, std::make_unique<DataPtrLookupTable>());
    HT_RUNTIME_ERROR_IF(!insertion.second)
      << "Failed to insert lookup table to " << stream;
  } else {
    found_avaiable = FindAvailableFromLookupTable(
      aligned_num_bytes, *(it->second), true, data_ptr);
    if (found_avaiable) {
      // Update the `alloc_at` clock, which is later than the previous `free_at`
      auto it = _data_ptr_info.find(data_ptr.id);
      HT_RUNTIME_ERROR_IF(it == _data_ptr_info.end())
        << "Cannot find data " << data_ptr << " from info";
      it->second.alloc_at = alloc_at;
      it->second.status = OccupationStatus::OCCUPIED_BY_ALLOC_STREAM;
    }
  }

  // Find among data spaces that are available for all streams.
  if (!found_avaiable) {
    found_avaiable = FindAvailableFromLookupTable(
      aligned_num_bytes, *_available_for_all_streams, true, data_ptr);
    if (found_avaiable) {
      auto insertion = _data_ptr_info.emplace(
        data_ptr.id, 
        CudaDataPtrInfo(data_ptr.ptr, data_ptr.size, stream, alloc_at));
      HT_RUNTIME_ERROR_IF(!insertion.second)
        << "Failed to insert data " << data_ptr << " to info";
    }
  }

  // Cannot find avaiable memory to re-use. Alloc from system.
  if (!found_avaiable) {
    // TODO: Check whether the memory limitation has been reached. 
    // If yes, we shall free/re-use some cached memories on other streams.
    hetu::cuda::CUDADeviceGuard guard(device().index());
    void* ptr;
    CudaMalloc(&ptr, aligned_num_bytes);
    data_ptr = {ptr, aligned_num_bytes, device(), next_id()};
    _reserved += aligned_num_bytes;
    _peak_reserved = MAX(_peak_reserved, _reserved);
    _cuda_malloc_cnt++;

    auto insertion = _data_ptr_info.emplace(
      data_ptr.id, 
      CudaDataPtrInfo(data_ptr.ptr, aligned_num_bytes, stream, alloc_at));
    HT_RUNTIME_ERROR_IF(!insertion.second)
      << "Failed to insert data " << data_ptr << " to info";
  }

  _allocated += data_ptr.size;
  _alloc_cnt++;

  return data_ptr;
}

bool CUDACachingMemoryPool::FindAvailableFromLookupTable(
  size_t num_bytes, DataPtrLookupTable& lookup_table, bool remove_if_found,
  DataPtr& ret) {
  // the caller should hold the mutex
  auto it = lookup_table.table.find(num_bytes);
  if (it != lookup_table.table.end() && !it->second.empty()) {
    auto& tuple = it->second.back();
    ret.id = std::get<0>(tuple);
    ret.ptr = std::get<1>(tuple);
    ret.size = std::get<2>(tuple);
    if (remove_if_found)
      it->second.pop_back();
    return true;
  }
  return false;
}

void CUDACachingMemoryPool::InsertAvailableToLookupTable(
  const DataPtr& data_ptr, DataPtrLookupTable& lookup_table) {
  // the caller should hold the mutex
  auto it = lookup_table.table.find(data_ptr.size);
  if (it == lookup_table.table.end()) {
    auto insertion = lookup_table.table.emplace(
      data_ptr.size, std::vector<std::tuple<DataPtrId, void*, size_t>>());
    HT_RUNTIME_ERROR_IF(!insertion.second)
      << "Failed to insert key " << data_ptr.size << " to lookup table";
    it = insertion.first;
    it->second.reserve(1024);
  }
  it->second.emplace_back(
    std::make_tuple(data_ptr.id, data_ptr.ptr, data_ptr.size));
}

void CUDACachingMemoryPool::EmptyCacheInLookupTable(
  DataPtrLookupTable& lookup_table, bool maybe_allocated) {
  // the caller should hold the mutex
  for (auto it = lookup_table.table.begin(); it != lookup_table.table.end();
       it++) {
    auto& table_of_size = it->second;
    std::vector<std::tuple<DataPtrId, void*, size_t>> allocated;
    if (maybe_allocated)
      allocated.reserve(table_of_size.size());
    for (auto& tuple : table_of_size) {
      DataPtrId data_ptr_id = std::get<0>(tuple);
      void* ptr = std::get<1>(tuple);
      size_t size = std::get<2>(tuple);
      if (maybe_allocated) {
        auto it2 = _data_ptr_info.find(data_ptr_id);
        if (it2 != _data_ptr_info.end()) {
          if (it2->second.allocated()) {
            // Do not free if it is currently allocated.
            allocated.push_back(tuple);
            continue;
          } else {
            // Going to be freed. Remove from the info.
            _data_ptr_info.erase(it2);
          }
        }
      }
      CudaFree(ptr);
      _reserved -= size;
    }
    table_of_size.clear();
    if (!allocated.empty()) {
      table_of_size.insert(table_of_size.end(), allocated.begin(),
                           allocated.end());
    }
  }
}

void CUDACachingMemoryPool::EmptyCache() {
  std::lock_guard<std::mutex> lock(_mtx);
  for (auto& kv : _available_for_single_stream) {
    EmptyCacheInLookupTable(*(kv.second), true);
  }
  EmptyCacheInLookupTable(*_available_for_all_streams, false);
}

DataPtr CUDACachingMemoryPool::BorrowDataSpace(void* ptr, size_t num_bytes,
                                               DataPtrDeleter deleter) {
  HT_VALUE_ERROR_IF(ptr == nullptr || num_bytes == 0)
    << "Borrowing an empty storage is not allowed";
  HT_VALUE_ERROR_IF(!deleter)
    << "Deleter must not be empty when borrowing storages";

  std::lock_guard<std::mutex> lock(_mtx);
  // Note: The borrowed memory must be ready, so we use blocking stream here
  DataPtr data_ptr{ptr, num_bytes, device(), next_id()};
  Stream borrow_stream = Stream(device(), kBlockingStream);
  uint64_t borrow_at = next_clock();
  auto insertion = _data_ptr_info.emplace(data_ptr.id,
                                          CudaDataPtrInfo(data_ptr.ptr, data_ptr.size,
                                          borrow_stream, borrow_at, std::move(deleter)));
  HT_RUNTIME_ERROR_IF(!insertion.second)
    << "Failed to insert data " << data_ptr << " to info";

  return data_ptr;
}

void CUDACachingMemoryPool::FreeDataSpace(DataPtr data_ptr) {
  if (data_ptr.ptr == nullptr || data_ptr.size == 0)
    return;
  std::lock_guard<std::mutex> lock(_mtx);
  mempool_clock_t free_at = next_clock();

  auto it = _data_ptr_info.find(data_ptr.id);
  HT_RUNTIME_ERROR_IF(it == _data_ptr_info.end())
    << "Cannot find data " << data_ptr << " from info";
  auto& info = it->second;
  info.free_at = free_at;

  if (info.status == OccupationStatus::OCCUPIED_BY_ALLOC_STREAM) {
    // for borrow data
    // we free it directly
    if (info.deleter) {
      _data_ptr_info.erase(it);
      info.deleter(data_ptr);
      return;
    }
    info.status = OccupationStatus::AVAILABLE_FOR_ALLOC_STREAM;
    InsertAvailableToLookupTable(
      data_ptr, *(_available_for_single_stream[info.alloc_stream]));
    _allocated -= info.num_bytes;
  } else if (info.status == OccupationStatus::OCCUPIED_BY_MULTI_STREAMS) {
    info.status = OccupationStatus::UNAVAILABLE_UNTIL_FREE;
    auto& used_streams = info.used_streams;
    for (auto s_id : used_streams) {
      auto event = std::make_unique<CUDAEvent>(data_ptr.device, false);
      event->Record(Stream::unpack(s_id));
      _free_events[s_id].emplace_back(
        std::make_tuple(std::move(event), data_ptr.id, free_at));
    }
    info.free_event_cnt += used_streams.size();
  } else {
    HT_RUNTIME_ERROR << "Unexpected occupation status ("
                     << static_cast<int>(info.status)
                     << ") during call to 'FreeDataSpace' for " << data_ptr;
    __builtin_unreachable();
  }

  _free_cnt++;
}

void CUDACachingMemoryPool::WatchEvents() {
  // the caller should hold the mutex
  for (auto& kv : _free_events) {
    // auto packed_stream_id = kv.first;
    auto& stream_free_events = kv.second;
    while (!stream_free_events.empty()) {
      auto& tuple = stream_free_events.front();
      std::unique_ptr<CUDAEvent>& event = std::get<0>(tuple);
      cudaError_t status = event->Query();
      if (status == cudaErrorNotReady) {
        // ignore and clear the not-ready error
        (void) cudaGetLastError();
        // since events are enqueued in order, we can ignore the rest
        break;
      } else if (status != cudaSuccess) {
        __HT_FATAL_SILENT(hetu::cuda::cuda_error)
          << "cudaEventQuery failed: " << cudaGetErrorString(status);
        __builtin_unreachable();
      }

      // decrement the number of free events for that 
      DataPtrId data_ptr_id = std::get<1>(tuple);
      auto it = _data_ptr_info.find(data_ptr_id);
      HT_RUNTIME_ERROR_IF(it == _data_ptr_info.end())
        << "Cannot find data " << data_ptr_id << " from info";
      auto& info = it->second;
      if ((--info.free_event_cnt) == 0) {
        // for borrow data
        // we free it directly
        if (info.deleter) {
          _data_ptr_info.erase(it);
          info.deleter(DataPtr{info.ptr, info.num_bytes, device(), data_ptr_id});
        } else {
          InsertAvailableToLookupTable(
            DataPtr{info.ptr, info.num_bytes, device(), data_ptr_id},
            *_available_for_all_streams);
          _allocated -= info.num_bytes;
          _data_ptr_info.erase(it);
        }
      }
      stream_free_events.pop_front();
    }
  }
}

void CUDACachingMemoryPool::MarkDataSpaceUsedByStream(DataPtr data_ptr,
                                                      const Stream& stream) {
  if (data_ptr.ptr == nullptr || data_ptr.size == 0 || stream.is_blocking())
    return;
  HT_VALUE_ERROR_IF(!stream.device().is_cuda())
    << "Cuda arrays must be used on cuda streams. Got " << stream;
  PackedStreamId packed_stream_id = stream.pack();
  
  std::lock_guard<std::mutex> lock(_mtx);
  auto it = _data_ptr_info.find(data_ptr.id);
  HT_RUNTIME_ERROR_IF(it == _data_ptr_info.end())
    << "Cannot find data " << data_ptr << " from info";
  auto& info = it->second;
  info.insert_used_stream(packed_stream_id);
  _mark_cnt++;
}

void CUDACachingMemoryPool::MarkDataSpacesUsedByStream(DataPtrList& data_ptrs,
                                                       const Stream& stream) {
  if (stream.is_blocking())
    return;
  HT_VALUE_ERROR_IF(!stream.device().is_cuda())
    << "Cuda arrays must be used on cuda streams. Got " << stream;
  PackedStreamId packed_stream_id = stream.pack();
  std::lock_guard<std::mutex> lock(_mtx);
  for (auto& data_ptr : data_ptrs) {
    if (data_ptr.ptr == nullptr || data_ptr.size == 0)
      continue;
    auto it = _data_ptr_info.find(data_ptr.id);
    HT_RUNTIME_ERROR_IF(it == _data_ptr_info.end())
      << "Cannot find data " << data_ptr << " from info";
    auto& info = it->second;
    info.insert_used_stream(packed_stream_id);
    _mark_cnt++;
  }
}

std::future<void> CUDACachingMemoryPool::WaitDataSpace(DataPtr data_ptr,
                                                       bool async) {
  if (data_ptr.ptr == nullptr || data_ptr.size == 0)
    return async ? std::async([]() {}) : std::future<void>();

  std::unique_lock<std::mutex> lock(_mtx);
  auto it = _data_ptr_info.find(data_ptr.id);
  HT_RUNTIME_ERROR_IF(it == _data_ptr_info.end())
    << "Cannot find data " << data_ptr << " from info";
  PackedStreamId alloc_stream = it->second.alloc_stream;
  auto& used_streams = it->second.used_streams;
  if (used_streams.empty()) {
    // This only happens when alloc_stream and all used_streams are blocking
    return async ? std::async([]() {}) : std::future<void>();
  }

  // TODO: Avoid synchronizing allocation and used streams again 
  // when freeing the memory. However, remember that it necessitates 
  // tracking whether each async waits has completed or not.
  Stream wait_stream;
  if (used_streams.size() == 1 && *used_streams.begin() == alloc_stream) {
    wait_stream = Stream::unpack(alloc_stream);
  } else {
    Stream join_stream(data_ptr.device, kJoinStream);
    for (auto& used_stream : used_streams) {
      CUDAEvent event(data_ptr.device, false);
      event.Record(Stream::unpack(used_stream));
      event.Block(join_stream);
    }
    wait_stream = join_stream;
  }
  lock.unlock();

  if (async) {
    return std::async([wait_stream]() { CUDAStream(wait_stream).Sync(); });
  } else {
    CUDAStream(wait_stream).Sync();
    return std::future<void>();
  }
}

void CUDACachingMemoryPool::PrintSummary() {
  HT_LOG_INFO << name() << ": alloc=" << _allocated << " bytes, "
    << "reserved=" << _reserved << " bytes, "
    << "peak_reserved=" << _peak_reserved << " bytes, "
    << "alloc_cnt=" << _alloc_cnt << ", "
    << "cuda_malloc_cnt=" << _cuda_malloc_cnt << ", "
    << "free_cnt=" << _free_cnt << ", "
    << "mark_cnt=" << _mark_cnt;
}

namespace {

static std::once_flag cuda_caching_memory_pool_register_flag;

struct CUDACachingMemoryPoolRegister {
  CUDACachingMemoryPoolRegister() {
    std::call_once(cuda_caching_memory_pool_register_flag, []() {
      // Memory pools are lazily constructed, so we do not need to
      // get device count here.
      for (int32_t i = 0; i < HT_MAX_DEVICE_INDEX; i++) {
        RegisterMemoryPoolCtor(
          Device(kCUDA, i), [i]() -> std::shared_ptr<MemoryPool> {
            return std::make_shared<CUDACachingMemoryPool>(i);
          });
      }
    });
  }
};

static CUDACachingMemoryPoolRegister cuda_caching_memory_pool_register;

} // namespace

} // namespace impl
} // namespace hetu
