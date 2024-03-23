#include "hetu/impl/memory/CUDACachingMemoryPool.cuh"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/cuda_utils.h"
#include <mutex>
#include <cstdlib>
#include <string>
#include <stdexcept>
namespace hetu {
namespace impl {

namespace {

inline static std::string _make_name(DeviceIndex device_id) {
  return "CUDACachingMemPool(" + std::to_string(static_cast<int>(device_id)) +
    ")";
}

} // namespace

CUDACachingMemoryPool::CUDACachingMemoryPool(DeviceIndex device_id, size_t _max_split_size)
: CUDAMemoryPool(device_id, _make_name(device_id)),
  max_split_size(_max_split_size){
  _data_ptr_info.reserve(8192);
  _available_for_single_stream.reserve(HT_NUM_STREAMS_PER_DEVICE); 
  _available_for_all_streams.reset(new DataPtrLookupTable());
}

CUDACachingMemoryPool::~CUDACachingMemoryPool() {
  // TODO: free the memory instead of let the OS collect them
}

/* 
  2nd迭代尝试如下设计
  1. lookup table使用更快速的数据结构
  2. 尝试所有ptr放入全局池子?
    (目前的方式是：一个ptr只有在被多个stream访问时,或者merge时，才会被放入全局池，
    而从全局池被捞出来之后，又被送回单个stream池子)
  3. 调用cudaMalloc玩一玩放锁
  4. 更细化的Malloc size alignment策略
  5. EventPool.
*/ 
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
  auto alignment = get_data_alignment(); // 256 bytes
  size_t aligned_num_bytes = DIVUP(num_bytes, alignment) * alignment;
  bool found_avaiable = false;

  DataPtrLookupTable* target_table_if_split;

  // Find among data spaces that are available only for this stream.
  auto it = _available_for_single_stream.find(packed_stream_id);
  if (it == _available_for_single_stream.end()) { 
    // It might be useful later, insert anyway. 
    auto insertion = _available_for_single_stream.emplace(
      packed_stream_id, std::make_unique<DataPtrLookupTable>());

    HT_RUNTIME_ERROR_IF(!insertion.second)
      << "Failed to insert lookup table to " << stream;
  } else {
    found_avaiable = FindAvailableFromLookupTable(aligned_num_bytes, 
                                                    *(it->second), data_ptr);
    if (found_avaiable) {
      // Update the `alloc_at` clock, which is later than the previous `free_at`
      auto it2 = _data_ptr_info.find(data_ptr.id);

      HT_RUNTIME_ERROR_IF(it2 == _data_ptr_info.end())
          << "Cannot find data " << data_ptr << " from info";

      it2->second->alloc_at = alloc_at;
      it2->second->status = OccupationStatus::OCCUPIED_BY_ALLOC_STREAM;

      target_table_if_split = it->second.get();
      // HT_LOG_TRACE << "found from local: "<< data_ptr;
    }
  }
  
  // Find among data spaces that are available for all streams.
  if (!found_avaiable) {
    found_avaiable = FindAvailableFromLookupTable(
      aligned_num_bytes, *_available_for_all_streams, data_ptr); 
    if (found_avaiable) {
      // OccupationStatus is set to OCCUPIED_BY_ALLOC_STREAM by default.
      auto it2 = _data_ptr_info.find(data_ptr.id);
      HT_RUNTIME_ERROR_IF(it2 == _data_ptr_info.end())
          << "Cannot find data " << data_ptr << " from info";

      it2->second->alloc_at = alloc_at;
      it2->second->status = OccupationStatus::OCCUPIED_BY_ALLOC_STREAM;
      it2->second->alloc_stream = packed_stream_id;
      
      target_table_if_split = _available_for_all_streams.get();
      // HT_LOG_TRACE << "found from global: " << data_ptr;
    }
  }
 

  // Cannot find avaiable memory to re-use. Alloc from system.
  if (!found_avaiable) {
    void* ptr;
    // Maybe use aligned_num_bytes is better? I expect the performance difference is negligible
    size_t malloc_size = GetAlignedMallocSize(aligned_num_bytes); 

    // TODO: Check whether the memory limitation has been reached. 
    // If yes, we shall free/re-use some cached memories on other streams.
    if(AllocNewPtr(ptr, malloc_size) 
            // Release some cached ptr in the same stream, and retry.
            || (ReleaseAvailableCache(*(*it).second.get(), malloc_size) 
              && AllocNewPtr(ptr, malloc_size))
            // Release all cached unallocated ptrs and retry.
            || (EmptyCache() && AllocNewPtr(ptr, malloc_size))){
      
      data_ptr = DataPtr(ptr, malloc_size, device(), next_id());

      _reserved += malloc_size;
      _peak_reserved = MAX(_peak_reserved, _reserved);
      _cuda_malloc_cnt++;

      auto new_info = std::make_shared<CudaDataPtrInfo>(data_ptr.ptr, malloc_size, 
                                                    stream, alloc_at, data_ptr.id);

      auto insertion = _data_ptr_info.emplace(data_ptr.id, new_info);
      HT_RUNTIME_ERROR_IF(!insertion.second)
        << "Failed to insert data " << data_ptr << " to info";
    
      target_table_if_split = _available_for_single_stream[packed_stream_id].get();
      (*(insertion.first)).second->cached_pool = target_table_if_split;
    } else {
      HT_RUNTIME_ERROR
      << "CUDA out of memory. On GPU" << _device.index() << ", Hetu "
      << "reserved:" << _reserved/(1024*1024*1024) << " GB, "
      << "allocated:" << _allocated/(1024*1024*1024) << " GB. "
      << "Please set environment variable HETU_MAX_SPLIT_SIZE_MB smaller "
      << "if you find reserved-allocated is too much." ;
    }
    // HT_LOG_TRACE << "found from new:" << data_ptr;
  }

  // Now we have prepared the target ptr. Do splitting if we need and 
  // update statistics.
  if(shouldSplit(data_ptr.size, aligned_num_bytes)){
    // Make the high address part a new cached ptr
    void* remaining_ptr = data_ptr.ptr + aligned_num_bytes;
    size_t remaining_size = data_ptr.size - aligned_num_bytes;
    data_ptr.size = aligned_num_bytes;

    size_t new_id = next_id();
    auto cache_insertion = target_table_if_split->table.emplace(
                    remaining_ptr, remaining_size, device(), new_id);
    HT_RUNTIME_ERROR_IF(!cache_insertion.second)
        << "Failed to insert data " << data_ptr << " to info";

    auto new_info = std::make_shared<CudaDataPtrInfo>(remaining_ptr, 
                                    remaining_size, stream, 0, new_id);
    auto info_insertion = _data_ptr_info.emplace(new_id, new_info);   
    HT_RUNTIME_ERROR_IF(!info_insertion.second)
        << "Failed to insert data " << data_ptr << " to info"; 
    
    auto it = _data_ptr_info.find(data_ptr.id); //NOTE: maybe 可以优化,减少一次查询...
    auto& cur_info = (*it).second;

    cur_info->num_bytes = aligned_num_bytes; 
    new_info->cached_pool = target_table_if_split;
    cur_info->cached_pool = target_table_if_split;

    new_info->prev = cur_info;
    new_info->next = cur_info->next;
    if(cur_info->next != nullptr)
      cur_info->next->prev = new_info;
    cur_info->next = new_info;
  }

  _allocated += data_ptr.size;
  _alloc_cnt++;

  HT_LOG_TRACE << "ptr: "<< data_ptr << ",alloc: " << data_ptr.size << ", stream:" << stream;
  return data_ptr;
}

size_t CUDACachingMemoryPool::GetAlignedMallocSize(size_t request_size){
  if (request_size < kMallocMinBuffer){
    return kMallocMinBuffer;
  } else if(request_size < kMallocLargeBuffer){
    return kMallocLargeBuffer;
  } else {
    return DIVUP(request_size, kMallocRoundUp)* kMallocRoundUp;
  }
  return request_size;
}

// TODO: Release lock and re-acquire to hide the latency of cudaMalloc
bool CUDACachingMemoryPool::AllocNewPtr(void* &ptr, size_t size){
    hetu::cuda::CUDADeviceGuard guard(device().index());
    cudaError_t ret = CudaMallocTry(&ptr, size);
    if(ret == cudaSuccess){
      return true;
    } else if(ret == cudaErrorMemoryAllocation){
      return false;
    } else {
      HT_RUNTIME_ERROR << "Cuda Malloc failed with rare reason.";
    }
}

bool CUDACachingMemoryPool::shouldSplit(size_t size, size_t request_size){
  return size < max_split_size && (size - request_size) > kMinSplitRemaining;
}

bool CUDACachingMemoryPool::FindAvailableFromLookupTable(
  size_t num_bytes, DataPtrLookupTable& lookup_table, DataPtr& ret) {
  // the caller should hold the mutex
  auto it = lookup_table.table.lower_bound(DataPtr(num_bytes, nullptr)); 

  if (it != lookup_table.table.end()) { 
    if ((*it).size != num_bytes){
      size_t remaining = (*it).size - num_bytes;
      if ((*it).size > max_split_size && num_bytes <= max_split_size){
        // Do not split a oversized ptr
        return false;
      } else if(num_bytes > max_split_size && remaining >= kMaxInternalFragment) {
        // num_bytes > max_split_size, so we will directly allocate this large
        // ptr to request without splitting. But we need to limit the remaining 
        // size to avoid large internal fragment.
        return false;
      }
    }
    ret = (*it);
    lookup_table.table.erase(it);
    return true;
  } else {
    return false;
  }
}

void CUDACachingMemoryPool::InsertAvailableToLookupTable(
  const DataPtr& data_ptr, DataPtrLookupTable& lookup_table) {
  // the caller should hold the mutex
  auto result = lookup_table.table.emplace(data_ptr);
  HT_RUNTIME_ERROR_IF(!result.second)
      << "Failed to insert key " << data_ptr.size << " to lookup table.";
}

// Try to empty a ptr look up table: delete all the records and free corresponding 
// pointer. If maybe_allocated is set to true, it only delete records whose pointer
// is not in use. Caller should hold the mutex.
void CUDACachingMemoryPool::EmptyCacheInLookupTable(
  DataPtrLookupTable& lookup_table, bool maybe_allocated) {
  for(auto it = lookup_table.table.begin(); 
              it != lookup_table.table.end(); it++) {
    auto record_it = _data_ptr_info.find((*it).id);
    HT_RUNTIME_ERROR_IF(record_it == _data_ptr_info.end()) 
          << "Cannot find one ptr's info.";

    if(maybe_allocated && record_it->second->allocated())
      continue;
    else {
      CudaFree((*it).ptr);
      auto& info = (*record_it).second;

      if(info->prev != nullptr)
        info->prev->next = info->next;
      if(info->next != nullptr)
        info->next->prev = info->prev;
      
      _data_ptr_info.erase(record_it);

      it = lookup_table.table.erase(it);
      _reserved -= (*it).size;
    }
  }
}

bool CUDACachingMemoryPool::EmptyCache() {
  std::lock_guard<std::mutex> lock(_mtx);
  for (auto& kv : _available_for_single_stream) {
    EmptyCacheInLookupTable(*(kv.second), true);
  }
  EmptyCacheInLookupTable(*_available_for_all_streams, false);
  return true;
}

// Free some oversize pointer in the given lookup table to satisfy the request_size.
// It will try to satisfy the request_size with minimum number of ptrs. Caller should 
// hold the mutex.
bool CUDACachingMemoryPool::ReleaseAvailableCache(DataPtrLookupTable& lookup_table,
                                                  size_t request_size){
  // We only release oversize pointer. If max_split_size_mb is not specified,
  // no pointers will be regraded as oversize.                                                  
  if(max_split_size == std::numeric_limits<size_t>::max())
    return false; 

  DataPtr tmp_key = {request_size > max_split_size 
                                      ? request_size 
                                      : max_split_size, nullptr};
  // Find if there are any ptr larger than request_size
  auto it = lookup_table.table.lower_bound(tmp_key);
  if(it == lookup_table.table.end()) {
    // Request size is larger than all cached ptrs.
    size_t released_size = 0;
    --it;
    while(released_size < request_size && (*it).size > max_split_size){
      auto ptr_info = _data_ptr_info.find(((*it).id));
      HT_RUNTIME_ERROR_IF(ptr_info == _data_ptr_info.end())
        << "cannot find CudaDataPtrInfo for stream's cached ptr.";

      if(!ptr_info->second->allocated()){
        CudaFree((*it).ptr);
        released_size += (*it).size;
        _reserved += (*it).size;

        auto tmp = it++;
        lookup_table.table.erase(tmp);
        _data_ptr_info.erase(ptr_info);  
      } 

      if(it != lookup_table.table.begin())
        it--;
      else
        break;
    }

    if(released_size < request_size)
      return false;
  } else {
    auto ptr_info = _data_ptr_info.find(((*it).id));
    HT_RUNTIME_ERROR_IF(ptr_info == _data_ptr_info.end())
      << "cannot find CudaDataPtrInfo for stream's cached ptr.";
    _data_ptr_info.erase(ptr_info);
    CudaFree((*it).ptr);
    lookup_table.table.erase(it);  
  }   
  return true;
}

DataPtr CUDACachingMemoryPool::BorrowDataSpace(void* ptr, size_t num_bytes,
                                               DataPtrDeleter deleter,
                                               const Stream& stream) {
  HT_VALUE_ERROR_IF(ptr == nullptr || num_bytes == 0)
    << "Borrowing an empty storage is not allowed";
  HT_VALUE_ERROR_IF(!deleter)
    << "Deleter must not be empty when borrowing storages";

  std::lock_guard<std::mutex> lock(_mtx);
  WatchEvents();
  // Note: The borrowed memory must be ready, so we use blocking stream here
  DataPtr data_ptr{ptr, num_bytes, device(), next_id()};
  Stream borrow_stream = stream.is_defined() ? stream : Stream(device(), kBlockingStream);
  uint64_t borrow_at = next_clock();
  auto insertion = _data_ptr_info.emplace(data_ptr.id,
                                          std::make_shared<CudaDataPtrInfo>(data_ptr.ptr, data_ptr.size,
                                          borrow_stream, borrow_at, data_ptr.id, std::move(deleter)));
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
  auto info = it->second;

  // for borrow data, we currently adopt method 1
  // the exec graph running & switching memory profiling will be more accurate
  // note we will eventually use method 2

  // method 1: for borrow data we free it directly
  // we should block the used streams here
  /*
  if (info->deleter) {
    // Stream::unpack(info->alloc_stream).Sync();
    auto& used_streams = info->used_streams;
    for (auto s_id : used_streams) {
      Stream::unpack(s_id).Sync();
    }
    info->deleter(data_ptr);
    _data_ptr_info.erase(it);
    return;
  }
  */

  // method 2: move borrow data actual free to WatchEvents()
  // we only record the free event here
  if (info->deleter) {
    auto& used_streams = info->used_streams;
    if (used_streams.empty()) {
      info->deleter(data_ptr);
      _data_ptr_info.erase(it);
      return;
    }
    info->status = OccupationStatus::UNAVAILABLE_UNTIL_FREE;
    for (auto s_id : used_streams) {
      auto event = std::make_unique<CUDAEvent>(data_ptr.device, false);
      event->Record(Stream::unpack(s_id));
      _free_events[s_id].emplace_back(
        std::make_tuple(std::move(event), data_ptr.id));
    }
    info->free_event_cnt += used_streams.size();
    _free_cnt++;
    return;
  }

  if (info->status == OccupationStatus::OCCUPIED_BY_ALLOC_STREAM) {
    // HT_LOG_TRACE << "free here: " << data_ptr;
    info->status = OccupationStatus::AVAILABLE_FOR_ALLOC_STREAM;
    mempool_clock_t free_at = next_clock();
    info->free_at = free_at;

    DataPtrLookupTable* target_table = 
          _available_for_single_stream[info->alloc_stream].get();
    if(info->is_split())
      target_table = tryMerge(info, target_table); 
    info->cached_pool = target_table; // ? NOTE: 可以不要，试试
    // Meta information of data_ptr might have change in tryMerge
    data_ptr.size = info->num_bytes;
    data_ptr.ptr = info->ptr;
    HT_LOG_TRACE << data_ptr.ptr << " is freed from single stream. dataptr: " << data_ptr.id;

    InsertAvailableToLookupTable(data_ptr, *(target_table));
    _allocated -= info->num_bytes;
  } else if (info->status == OccupationStatus::OCCUPIED_BY_MULTI_STREAMS) {
    // HT_LOG_TRACE << "free here? " << data_ptr << std::endl;
    info->status = OccupationStatus::UNAVAILABLE_UNTIL_FREE;
    auto& used_streams = info->used_streams;

    for (auto s_id : used_streams) {
      auto event = std::make_unique<CUDAEvent>(data_ptr.device, false);
      event->Record(Stream::unpack(s_id)); // TODO: log here
      _free_events[s_id].emplace_back(
        std::make_tuple(std::move(event), data_ptr.id));
    }
    info->free_event_cnt += used_streams.size();
  } else {
    HT_RUNTIME_ERROR << "Unexpected occupation status ("
                     << static_cast<int>(info->status)
                     << ") during call to 'FreeDataSpace' for " << data_ptr;
    __builtin_unreachable();
  }

  _free_cnt++;
}

// Try to merge blocks. It assume that data_ptr has not been inserted into any
// lookup table. It will check if there are adjacent splitted ptr and try to merge
// them. 
// 
// "data_ptr" refers to newly released ptr(modifiable), and "table" refers to the lookup 
// table where the pointer (ptr) was originally intended to be inserted in.
// 
// Return a pointer of the target lookup table which we want to insert the merged ptr in.
DataPtrLookupTable* CUDACachingMemoryPool::tryMerge(std::shared_ptr<CudaDataPtrInfo>& data_info, 
                                                                      DataPtrLookupTable* table){
  // Decide which lookup table will the merged ptr be inserted in.
  // Only when src and dst are both not global lookup table will it select stream lookup table.
  auto table_selection = [&](DataPtrLookupTable* src, DataPtrLookupTable* dst) 
                                                          -> DataPtrLookupTable* {
    if(src == dst && src != _available_for_all_streams.get()){
      return src;
    } else {
      return _available_for_all_streams.get();
    }
  };
  void* prev = data_info->prev.get();
  void* next = data_info->next.get();
  void* now =  data_info.get();
  int left_id = -1;
  if(data_info->prev.get() != nullptr && !data_info->prev->allocated()){
    auto prev_info = data_info->prev; 
    size_t num = prev_info->cached_pool->table.erase( 
                    DataPtr{prev_info->num_bytes, prev_info->ptr});
    // HT_LOG_TRACE << "in try merge, erase prev:" << DataPtr{prev_info->num_bytes, prev_info->ptr} << ",id:" << prev_info->id;
    if(num == 0){
      for(auto& it : prev_info->cached_pool->table){
        HT_LOG_TRACE << it;
      }
      if(prev_info->cached_pool != _available_for_all_streams.get()){
        for(auto& it : _available_for_all_streams->table){
          HT_LOG_TRACE << "in global: " << it;
        }
      }
      HT_LOG_TRACE << "can find in data info? " << (_data_ptr_info.find(prev_info->id) != _data_ptr_info.end());

      HT_RUNTIME_ERROR << "cannot find caching record in tryMerge, pool is? "
      << prev_info->cached_pool<< " | " << data_info->ptr << ", "<<data_info->num_bytes << ", " << data_info->id
      << DataPtr{prev_info->ptr, prev_info->num_bytes, Device(), prev_info->id}
      << " prev has been merged?" << (merged_id.find(prev_info->id) != merged_id.end())
      << " cur has been merged!" << (merged_id.find(data_info->id) != merged_id.end());
    }
    HT_RUNTIME_ERROR_IF(now !=data_info.get()) << "TryMerge check 1";

    table = table_selection(prev_info->cached_pool, table);
    HT_RUNTIME_ERROR_IF(now !=data_info.get()) << "TryMerge check 2";

    data_info->ptr = prev_info->ptr;
    HT_RUNTIME_ERROR_IF(now !=data_info.get()) << "TryMerge check 3";

    data_info->num_bytes += prev_info->num_bytes;
    HT_RUNTIME_ERROR_IF(now !=data_info.get()) << "TryMerge check 4";
    data_info->prev = prev_info->prev;
    HT_RUNTIME_ERROR_IF(now !=data_info.get()) << "TryMerge check 5";

    if(prev_info->prev != nullptr)
      prev_info->prev->next = data_info;
    HT_RUNTIME_ERROR_IF(now !=data_info.get()) << "TryMerge check 6";


    _data_ptr_info.erase(prev_info->id); // ???
    HT_RUNTIME_ERROR_IF(now !=data_info.get()) << "TryMerge check 7";

    merged_id.insert(prev_info->id);
    HT_RUNTIME_ERROR_IF(now !=data_info.get()) << "TryMerge check 8";

    left_id = prev_info->id;
  }
  if(data_info->next.get() != nullptr && !data_info->next->allocated()){
    auto next_info = data_info->next; 
    // HT_LOG_TRACE << "in try merge, erase next:" << DataPtr{next_info->num_bytes, next_info->ptr} << ",id:" << next_info->id;
    size_t num = next_info->cached_pool->table.erase( 
                    DataPtr{next_info->num_bytes, next_info->ptr});
    if(num == 0){
      for(auto& it : next_info->cached_pool->table){
        HT_LOG_TRACE << it;
      }
      if(next_info->cached_pool != _available_for_all_streams.get()){
        for(auto& it : _available_for_all_streams->table){
          HT_LOG_TRACE << "in global: " << it;
        }
      }
      HT_LOG_TRACE << "prev is " << left_id;
      HT_LOG_TRACE << "prev_ptr " << prev << " cur "<< now << " next " << next;
      HT_LOG_TRACE << "right now cur" << data_info.get() << " next " << next_info.get();
      HT_LOG_TRACE << "can find in data info? " << (_data_ptr_info.find(next_info->id) != _data_ptr_info.end());
      HT_RUNTIME_ERROR << "cannot find caching record in tryMerge, pool is " 
        << next_info->cached_pool << ", global pools:" << _available_for_all_streams.get() << " | "
        << data_info->ptr << ", "<<data_info->num_bytes << ", " << data_info->id
        << DataPtr{next_info->ptr, next_info->num_bytes, Device(), next_info->id}
        << " next has been merged! " << (merged_id.find(next_info->id) != merged_id.end())
        << " cur has been merged! " << (merged_id.find(data_info->id) != merged_id.end());

    }
    table = table_selection(table, next_info->cached_pool);

    data_info->num_bytes += next_info->num_bytes;
    data_info->next = next_info->next;
    if(next_info->next != nullptr)
      next_info->next->prev = data_info;
    _data_ptr_info.erase(next_info->id);
    merged_id.insert(next_info->id);
  }
  return table;
}

// the caller should hold the mutex
void CUDACachingMemoryPool::WatchEvents() {
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
      if ((--info->free_event_cnt) == 0) {
        if (info->deleter) {
          info->deleter(DataPtr{info->ptr, info->num_bytes, device(), data_ptr_id});
          _data_ptr_info.erase(it);
        } else {
          DataPtrLookupTable* target_table = _available_for_all_streams.get();
          info->refresh();
          if(info->is_split())
            target_table = tryMerge(info, target_table); 
          DataPtr to_insert_ptr = {info->ptr, info->num_bytes, device(), data_ptr_id};
          info->cached_pool = target_table;
          HT_LOG_TRACE << to_insert_ptr.ptr << " is freed from multi stream. dataptr: " << to_insert_ptr.id;
          InsertAvailableToLookupTable(to_insert_ptr, *target_table); 
          _allocated -= info->num_bytes;
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
  info->insert_used_stream(packed_stream_id);
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
    info->insert_used_stream(packed_stream_id);
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
  PackedStreamId alloc_stream = it->second->alloc_stream;
  auto& used_streams = it->second->used_streams;
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
  HT_LOG_INFO << name() << ": alloc = " << _allocated << " bytes, "
    << "reserved = " << _reserved << " bytes, "
    << "peak_reserved = " << _peak_reserved << " bytes, "
    << "alloc_cnt = " << _alloc_cnt << ", "
    << "cuda_malloc_cnt = " << _cuda_malloc_cnt << ", "
    << "free_cnt = " << _free_cnt << ", "
    << "mark_cnt = " << _mark_cnt;
}

namespace {
static std::once_flag cuda_caching_memory_pool_register_flag;

size_t ParseMaxSplitSize(){
  const char* max_split_str = std::getenv("HETU_MAX_SPLIT_SIZE_MB");
  size_t max_split_size_mb;
  if(max_split_str != NULL){
      try {
        max_split_size_mb = std::stoi(max_split_str);
        // TODO: 敲定一个最小值....
      } catch (const std::exception& e) {
        HT_LOG_WARN
          << "invalid HETU_MAX_SPLIT_SIZE_MB: " << max_split_str
          << "is set. Please provide an integer whose value > 20. "
          << "Default value will be used in this process.";
      }
  } else
    max_split_size_mb = std::numeric_limits<size_t>::max();
  
  return max_split_size_mb*1024*1024;
}

struct CUDACachingMemoryPoolRegister {
  CUDACachingMemoryPoolRegister() { 
    std::call_once(cuda_caching_memory_pool_register_flag, []() {
      size_t _max_split_size = ParseMaxSplitSize();
      // Memory pools are lazily constructed, so we do not need to
      // get device count here.
      for (int32_t i = 0; i < HT_MAX_DEVICE_INDEX; i++) {
        RegisterMemoryPoolCtor(
          Device(kCUDA, i), [i, _max_split_size]() -> std::shared_ptr<MemoryPool> {
            return std::make_shared<CUDACachingMemoryPool>(i, _max_split_size);
          });
      }
    });
  }
};
static CUDACachingMemoryPoolRegister cuda_caching_memory_pool_register;

} // namespace

} // namespace impl
} // namespace hetu
