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

// 分配num bytes的显存
// 先会从available table中找
// 没有匹配项再cudaMalloc
DataPtr CUDACachingMemoryPool::AllocDataSpace(size_t num_bytes,
                                              const Stream& stream) {                         
  // mempool debug use    
  // HT_LOG_INFO << "Try to alloc: " << num_bytes << " to " << stream;
  HT_VALUE_ERROR_IF(!stream.device().is_cuda())
    << "Cuda arrays must be allocated on cuda streams. Got " << stream;
  if (num_bytes == 0)
    return DataPtr{nullptr, 0, device(), static_cast<DataPtrId>(-1)};

  PackedStreamId packed_stream_id = stream.pack(); 
  std::lock_guard<std::mutex> lock(_mtx);

  WatchEvents();

  // Update the `alloc_at` clock, which is later than the previous `free_at`
  uint64_t alloc_at = next_clock();
  DataPtr data_ptr;
  data_ptr.device = device();
  auto alignment = get_data_alignment(); // 256 bytes
  size_t aligned_num_bytes = DIVUP(num_bytes, alignment) * alignment;
  bool found_avaiable = false;

  DataPtrLookupTable* target_table_if_split = nullptr;
  auto info_it = _data_ptr_info.end();

  // Find among data spaces that are available only for this stream.
  auto table_it = _available_for_single_stream.find(packed_stream_id);
  if (table_it == _available_for_single_stream.end()) { 
    // It might be useful later, insert anyway. 
    auto insertion = _available_for_single_stream.emplace(packed_stream_id, std::make_unique<DataPtrLookupTable>());
    HT_RUNTIME_ERROR_IF(!insertion.second)
      << "Failed to insert lookup table to " << stream;
    table_it = insertion.first;
  } 
  // 已经创建过available table了
  else {
    found_avaiable = FindAvailable(aligned_num_bytes, 
                                   *(table_it->second), 
                                   data_ptr);
    if (found_avaiable) {
      // mempool debug use    
      // HT_LOG_INFO << "Find available " << data_ptr;
      info_it = _data_ptr_info.find(data_ptr.id);
      HT_RUNTIME_ERROR_IF(info_it == _data_ptr_info.end())
        << "Cannot find data " << data_ptr << " from info";
      HT_ASSERT(info_it->second->status == OccupationStatus::AVAILABLE_FOR_ALLOC_STREAM)
        << "Info status should be " << OccupationStatus::AVAILABLE_FOR_ALLOC_STREAM
        << ", but it is actually " << info_it->second->status;
      info_it->second->status = OccupationStatus::OCCUPIED_BY_ALLOC_STREAM;
      info_it->second->alloc_at = alloc_at;
      target_table_if_split = table_it->second.get();
      HT_ASSERT(info_it->second->cached_pool == table_it->second.get())
        << "Assumption error";
      info_it->second->cached_pool = nullptr;
    }
  }
  
  // Find among data spaces that are available for all streams.
  if (!found_avaiable) {
    found_avaiable = FindAvailable(aligned_num_bytes, 
                                   *_available_for_all_streams, 
                                    data_ptr); 
    if (found_avaiable) {
      info_it = _data_ptr_info.find(data_ptr.id);
      HT_RUNTIME_ERROR_IF(info_it == _data_ptr_info.end())
        << "Cannot find data " << data_ptr << " from info";
      HT_ASSERT(info_it->second->status == OccupationStatus::AVAILABLE_FOR_ALL_STREAM)
        << "Info status should be " << OccupationStatus::AVAILABLE_FOR_ALL_STREAM
        << ", but it is actually " << info_it->second->status;
      info_it->second->status = OccupationStatus::OCCUPIED_BY_ALLOC_STREAM;
      info_it->second->alloc_at = alloc_at;
      info_it->second->alloc_stream = packed_stream_id;
      target_table_if_split = _available_for_all_streams.get();
      HT_ASSERT(info_it->second->cached_pool == _available_for_all_streams.get())
        << "Assumption error";
      info_it->second->cached_pool = nullptr;
    }
  }
 
  // Cannot find any avaiable memory to re-use, then cudaMalloc from system.
  // *只有这种情况会cudaMalloc并将新分配的data ptr放入到info中
  // 同时记录(cuda) data ptr所在的table
  if (!found_avaiable) {
    void* ptr;
    // Maybe use aligned_num_bytes is better?
    // size_t malloc_size = GetAlignedMallocSize(aligned_num_bytes); 
    size_t malloc_size = aligned_num_bytes;
    // Check whether the memory limitation has been reached. 
    // If yes, we shall free/re-use some cached memories on other streams.
    if (AllocNewPtr(ptr, malloc_size))
        /* 
        // Release some oversized cached ptrs in the same stream, and retry.
        || (ReleaseOversized(*(table_it->second), malloc_size) && AllocNewPtr(ptr, malloc_size))
        // Release all cached ptrs in the same stream, and retry.
        || (ReleaseAll(*(table_it->second)) && AllocNewPtr(ptr, malloc_size))
        // Release all cached ptrs in the all stream available table, and retry.
        || (ReleaseAll(*_available_for_all_streams) && AllocNewPtr(ptr, malloc_size))
        // Synchronize all streams and free all cached ptrs in the system, and retry.
        || (EmptyCache() && AllocNewPtr(ptr, malloc_size))) */ {
      // ------ create new data ptr place 1 ------ 
      data_ptr = DataPtr{ptr, malloc_size, device(), next_id()};
      // mempool debug use    
      // HT_LOG_INFO << "[Create] cudaMalloc new " << data_ptr;
      _reserved += malloc_size;
      _peak_reserved = MAX(_peak_reserved, _reserved);
      _cuda_malloc_cnt++;
      auto new_info = std::make_shared<CudaDataPtrInfo>(data_ptr.ptr, malloc_size, 
                                                        stream, alloc_at, data_ptr.id);                                               
      // 此时cudaMalloc出来的新的(cuda) data ptr还不具有cache的table
      new_info->cached_pool = nullptr; // 默认值其实就是nullptr（这里只是为了强调一下）
      auto insertion = _data_ptr_info.emplace(data_ptr.id, new_info);
      HT_RUNTIME_ERROR_IF(!insertion.second)
        << "Failed to insert data " << data_ptr << " to info";
      info_it = insertion.first;
    } 
    // 清空cache后依然无法分配
    else {
      HT_RUNTIME_ERROR
        << "CUDA out of memory. On GPU " << _device.index() << ", "
        << "reserved: " << _reserved / (1024 * 1024 * 1024) << " GB, "
        << "allocated: " << _allocated / (1024 * 1024 * 1024) << " GB. "
        << "Please set environment variable HETU_MAX_SPLIT_SIZE_MB smaller, "
        << "if you find reserved-allocated is too much";
    }
  }

  // Now we have prepared the target ptr. Do splitting if we need and update statistics.
  // 此时的data ptr是一个不在任何available table中的东西
  // 且相应的info已经设置好了alloc_at、alloc_stream
  if (ShouldSplit(data_ptr.size, aligned_num_bytes)) {
    // split的只有可能是刚从table中取出来reuse的条目
    // 因为cudaMalloc分配的会不多不少
    // Make the high address part a new cached ptr
    auto cur_info = info_it->second;
    void* remaining_ptr = data_ptr.ptr + aligned_num_bytes;
    size_t remaining_size = data_ptr.size - aligned_num_bytes;
    size_t new_id = next_id();
    // ------ create new data ptr place 2 ------ 
    auto remaining_data_ptr = DataPtr{remaining_ptr, remaining_size, device(), new_id};
    // 将remaining data ptr插入table
    HT_ASSERT(target_table_if_split)
      << "Target table is a nullptr";
    // mempool debug use
    // HT_LOG_INFO << "[Create] create new split: " << remaining_data_ptr;
    // HT_LOG_INFO << "[Insert] split then insert to table: " << remaining_data_ptr;
    InsertAvailable(remaining_data_ptr, *target_table_if_split);
    // 将remaining (cuda) data ptr插入info
    auto new_info = std::make_shared<CudaDataPtrInfo>(remaining_ptr, remaining_size, stream, 0, new_id);
    new_info->status = target_table_if_split == _available_for_all_streams.get() ? 
                       OccupationStatus::AVAILABLE_FOR_ALL_STREAM : OccupationStatus::AVAILABLE_FOR_ALLOC_STREAM;
    new_info->cached_pool = target_table_if_split;
    auto info_insertion = _data_ptr_info.emplace(new_id, new_info);   
    HT_RUNTIME_ERROR_IF(!info_insertion.second)
      << "Failed to insert splitted (cuda) data ptr " << remaining_data_ptr << " to info"; 
    // 修正之前的data ptr和info中的(cuda) data ptr 
    cur_info->num_bytes = aligned_num_bytes; 
    data_ptr.size = aligned_num_bytes;
    // 用链表连接split的info
    new_info->prev = cur_info;
    new_info->next = cur_info->next;
    if (cur_info->next != nullptr)
      cur_info->next->prev = new_info;
    cur_info->next = new_info;
  }

  _allocated += data_ptr.size;
  _alloc_cnt++;
  HT_LOG_TRACE << "ptr: " << data_ptr << ", alloc: " << data_ptr.size << ", stream: " << stream;
  // mempool debug use
  // HT_LOG_INFO << "[Interface] alloc to user: " << data_ptr;
  /*
  info_it = _data_ptr_info.find(data_ptr.id);
  HT_ASSERT(data_ptr.size == info_it->second->num_bytes)
    << "Find info: size = " << info_it->second->num_bytes
    << ", but data ptr = " << data_ptr; 
  */
  return data_ptr;
}

// deprecated
size_t CUDACachingMemoryPool::GetAlignedMallocSize(size_t request_size) {
  if (request_size < kMallocMinBuffer) {
    return kMallocMinBuffer;
  } else if (request_size < kMallocLargeBuffer) {
    return kMallocLargeBuffer;
  } else {
    return DIVUP(request_size, kMallocRoundUp) * kMallocRoundUp;
  }
  return request_size;
}

// TODO: Release lock and re-acquire to hide the latency of cudaMalloc
bool CUDACachingMemoryPool::AllocNewPtr(void* &ptr, size_t size) {
  hetu::cuda::CUDADeviceGuard guard(device().index());
  cudaError_t ret = CudaMallocTry(&ptr, size);
  if (ret == cudaSuccess) {
    return true;
  } else if (ret == cudaErrorMemoryAllocation) {
    return false;
  } else {
    HT_RUNTIME_ERROR << "Cuda Malloc failed with rare reason";
  }
}

bool CUDACachingMemoryPool::ShouldSplit(size_t size, size_t request_size) {
  HT_ASSERT(size >= request_size)
    << "Assumption error";
  return size <= max_split_size && (size - request_size) >= kMinSplitRemaining;
}

// 从available table中找到某一个满足num bytes的条目
// *并删除
// 如果找不到则返回false
bool CUDACachingMemoryPool::FindAvailable(size_t num_bytes, 
                                          DataPtrLookupTable& lookup_table, 
                                          DataPtr& ret,
                                          bool remove_if_find) {
  // the caller should hold the mutex
  auto it = lookup_table.table.lower_bound(DataPtr(num_bytes, nullptr)); 
  if (it != lookup_table.table.end()) { 
    auto info_it = _data_ptr_info.find(it->id);
    HT_RUNTIME_ERROR_IF(info_it == _data_ptr_info.end()) 
      << "Cannot find one ptr's info";
    // 目前available table中的应该全是unallocated的条目
    HT_ASSERT(!info_it->second->allocated())
      << "Assumption error";
    /*
    // 已经alloc的条目无法再被占用
    if (info_it->second->allocated()) {
      it++;
      continue;
    }
    */
    HT_ASSERT(it->size >= num_bytes)
      << "Assumption error";
    if (it->size != num_bytes) {
      size_t remaining = it->size - num_bytes;
      // 只有两种情况允许使用cache的条目
      // 1、该条目要比max_split_size小
      // 后续会split该条目
      // 一部分用于新分配的而空余的那部分重新插入
      if (it->size > max_split_size && num_bytes <= max_split_size) {
        // Do not split a oversized ptr
        return false;
      } 
      // 2、想分配的size要比max_split_size还大且剩余的内部碎片较少
      // 后续会直接占用整个条目且不进行split
      else if (num_bytes > max_split_size && remaining > kMaxInternalFragment) {
        // num_bytes > max_split_size, so we will directly allocate this large
        // ptr to request without splitting. But we need to limit the remaining 
        // size to avoid large internal fragment.
        return false;
      }
    }
    ret = (*it);
    // 删除条目
    if (remove_if_find) {
      // mempool debug use
      // HT_LOG_INFO << "[Reuse] remove from table: " << ret;
      lookup_table.table.erase(it);
    }
    return true;
  }
  // 找不到任何一个可以容纳的下的条目 
  return false;
}

// 直接insert即可
// 默认size从小到大排序
void CUDACachingMemoryPool::InsertAvailable(const DataPtr& data_ptr, 
                                            DataPtrLookupTable& lookup_table) {
  // the caller should hold the mutex
  auto result = lookup_table.table.emplace(data_ptr);
  HT_RUNTIME_ERROR_IF(!result.second)
    << "Failed to insert key " << data_ptr.size << " to lookup table";
}

// Free some oversize pointer in the given lookup table to satisfy the request_size.
// It will try to satisfy the request_size with minimum number of ptrs. 
// Caller should hold the mutex.
bool CUDACachingMemoryPool::ReleaseOversized(DataPtrLookupTable& lookup_table,
                                             size_t request_size) {
  // We only release oversize pointer. If max_split_size_mb is not specified,
  // no pointers will be regraded as oversize.                                                 
  DataPtr tmp_key = {request_size > max_split_size ? request_size : max_split_size, nullptr};
  // Find if there are any ptr larger than request_size
  auto it = lookup_table.table.lower_bound(tmp_key);
  // 从lower bound的条目依次往上找
  // 由于在FindAvailable会对条目进行删除因此都是unallocated的
  // 且由于保证了max_split_size因此其实际上都是可以free的
  while (it != lookup_table.table.end()) {
    auto info_it = _data_ptr_info.find(it->id);
    HT_RUNTIME_ERROR_IF(info_it == _data_ptr_info.end())
      << "Cannot find CudaDataPtrInfo for stream's cached ptr";
    HT_ASSERT(!info_it->second->allocated())
      << "Assumption error";
    HT_ASSERT(info_it->second->can_free())
      << "Assumption error";
    if (!info_it->second->allocated()) {
      CudaFree(it->ptr);
      _reserved += it->size;
      lookup_table.table.erase(it);
      _data_ptr_info.erase(info_it); 
      return true;
    }
    it++;
  }
  // 没有任何能直接满足的条目
  HT_ASSERT (it == lookup_table.table.end())
    << "Assumption error";
  // Request size is larger than all free cached ptrs.
  size_t released_size = 0;
  it--;
  // 只会释放大于max_split_size的
  // 那些小于max_split_size的可能因为进行了split而无法cudaFree
  while (released_size < request_size && it->size > max_split_size) {
    auto info_it = _data_ptr_info.find(it->id);
    HT_RUNTIME_ERROR_IF(info_it == _data_ptr_info.end())
      << "Cannot find CudaDataPtrInfo for stream's cached ptr";
    HT_ASSERT(!info_it->second->allocated())
      << "Assumption error";
    if (!info_it->second->allocated()) {
      CudaFree(it->ptr);
      released_size += it->size;
      _reserved += it->size;
      it = lookup_table.table.erase(it);
      _data_ptr_info.erase(info_it);  
    }
    if (it != lookup_table.table.begin())
      it--;
    else
      break;
  }
  if (released_size < request_size) {
    return false;
  }
  return true;
}

// Try to empty a ptr look up table: delete all the records and free corresponding pointer. 
// If maybe_allocated is set to true, it only delete records whose pointer is not in use. 
// Caller should hold the mutex.
// For now, all entry in the look up table should be unallocated, and maybe_allocated is unused.
bool CUDACachingMemoryPool::ReleaseAll(DataPtrLookupTable& lookup_table, 
                                       bool maybe_allocated) {   
  for(auto it = lookup_table.table.begin(); it != lookup_table.table.end(); it++) {
    auto info_it = _data_ptr_info.find(it->id);
    HT_RUNTIME_ERROR_IF(info_it == _data_ptr_info.end()) 
      << "Cannot find one ptr's info";
    // 目前available table中的应该全是unallocated的条目
    HT_ASSERT(!info_it->second->allocated())
      << "Assumption error";
    /*
    if (maybe_allocated && info_it->second->allocated())
      continue;
    */
    if (info_it->second->can_free()) {
      CudaFree(it->ptr);
      auto info = info_it->second;
      if(info->prev != nullptr)
        info->prev->next = info->next;
      if(info->next != nullptr)
        info->next->prev = info->prev;
      _data_ptr_info.erase(info_it);
      it = lookup_table.table.erase(it);
      _reserved -= it->size;
    }
  }
  return true;
}

bool CUDACachingMemoryPool::EmptyCache() {
  for (auto& kv : _available_for_single_stream) {
    ReleaseAll(*(kv.second));
  }
  ReleaseAll(*_available_for_all_streams);
  return true;
}

// 直接bind外部的内存
// 不会走cache的那一套逻辑
// 即不会被放到任何available table中
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

  // mempool debug use
  // HT_LOG_INFO << "[Interface] free from user: " << data_ptr;
  if (data_ptr.ptr == nullptr || data_ptr.size == 0)
    return;
  std::lock_guard<std::mutex> lock(_mtx);
  mempool_clock_t free_at = next_clock();

  auto info_it = _data_ptr_info.find(data_ptr.id);
  HT_RUNTIME_ERROR_IF(info_it == _data_ptr_info.end())
    << "Cannot find data " << data_ptr << " from info";
  auto info = info_it->second;

  HT_ASSERT(info->ptr == data_ptr.ptr)
    << "Find info: ptr = " << info->ptr
    << ", but data ptr = " << data_ptr; 
  HT_ASSERT(info->num_bytes == data_ptr.size)
      << "Find info: size = " << info->num_bytes 
      << ", but data ptr = " << data_ptr;
  /*
  // 允许info要比实际的data ptr要大
  if (info->num_bytes > data_ptr.size) {
    // data ptr过大时
    // 空余的不能超过kMaxInternalFragment
    if (data_ptr.size > max_split_size) {
      HT_ASSERT(info->num_bytes - data_ptr.size <= kMaxInternalFragment)
        << "Find info: size = " << info->num_bytes 
        << ", but data ptr: size = " << data_ptr.size;
    }
    // data ptr不够大
    // 则说明没有发生split
    else {
      HT_ASSERT(info->num_bytes - data_ptr.size < kMinSplitRemaining)
        << "Find info: size = " << info->num_bytes 
        << ", but data ptr: size = " << data_ptr.size;
    }
  }
  // 其余大部分情况只可能相等
  else {
    HT_ASSERT(info->num_bytes == data_ptr.size)
      << "Find info: size = " << info->num_bytes 
      << ", but data ptr: size = " << data_ptr.size;
  }
  */

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
    _data_ptr_info.erase(info_it);
    return;
  }
  */

  // method 2: move borrow data actual free to WatchEvents()
  // we only record the free event here
  if (info->deleter) {
    auto& used_streams = info->used_streams;
    if (used_streams.empty()) {
      info->deleter(data_ptr);
      _data_ptr_info.erase(info_it);
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

  // stream独占
  if (info->status == OccupationStatus::OCCUPIED_BY_ALLOC_STREAM) {
    info->status = OccupationStatus::AVAILABLE_FOR_ALLOC_STREAM;
    info->free_at = free_at;
    DataPtrLookupTable* target_table = _available_for_single_stream[info->alloc_stream].get();
    if (info->is_split())
      target_table = TryMerge(info, target_table); 
    // Meta information of data_ptr might have change in TryMerge
    data_ptr.size = info->num_bytes;
    data_ptr.ptr = info->ptr;
    HT_ASSERT(data_ptr.id == info->id)
      << "The data ptr id and the info id are mismatched"; 
    info->cached_pool = target_table;
    // mempool debug use
    // HT_LOG_INFO << "[Insert] free occupy then insert to table: " << data_ptr;
    InsertAvailable(data_ptr, *target_table); 
    _allocated -= info->num_bytes;
  } 
  // 当前被很多stream占用
  // 需要插入event等到所有stream都完成
  else if (info->status == OccupationStatus::OCCUPIED_BY_MULTI_STREAMS) {
    info->status = OccupationStatus::UNAVAILABLE_UNTIL_FREE;
    auto& used_streams = info->used_streams;
    for (auto s_id : used_streams) {
      auto event = std::make_unique<CUDAEvent>(data_ptr.device, false);
      event->Record(Stream::unpack(s_id)); 
      _free_events[s_id].emplace_back(std::make_tuple(std::move(event), data_ptr.id));
    }
    info->free_event_cnt += used_streams.size();
  } 
  else {
    HT_RUNTIME_ERROR << "Unexpected occupation status ("
                     << static_cast<int>(info->status)
                     << ") during call to 'FreeDataSpace' for " << data_ptr;
    __builtin_unreachable();
  }

  _free_cnt++;
}

// Try to merge blocks. It assume that data_ptr has not been inserted into any
// lookup table. It will check if there are adjacent splitted ptr and try to merge them. 
// "data_ptr" refers to newly released ptr, and "table" refers to the lookup 
// table where the ptr was originally intended to be inserted in.
// Return a pointer of the target lookup table which we want to insert the merged ptr in.
DataPtrLookupTable* CUDACachingMemoryPool::TryMerge(std::shared_ptr<CudaDataPtrInfo>& data_info, 
                                                    DataPtrLookupTable* table) {
  // Decide which lookup table will the merged ptr be inserted in.
  // Only when src and dst are both not global lookup table will it select stream lookup table.
  HT_ASSERT(!data_info->allocated())
    << "TryMerge can only used when data ptr is unallocated";
  HT_ASSERT(data_info->status >= OccupationStatus::AVAILABLE_FOR_ALLOC_STREAM)
    << "TryMerge can only used when info status is available to alloc";
  HT_ASSERT(data_info->cached_pool == nullptr)
    << "TryMerge should guarantee the data ptr is not cached but released just now";
  auto table_selection = [&](DataPtrLookupTable* src, DataPtrLookupTable* dst) -> DataPtrLookupTable* {
    // 在一个table中
    // 可以直接merge
    if (src == dst) {
      return src;
    } 
    // 不在一个table中
    // 必须要求其中一个是all stream available table才可以merge
    // 否则对于两个stream的table其之间的同步关系难以得到保证
    // 因此保守起见我们不进行跨stream的merge
    else {
      if (src == _available_for_all_streams.get())
        return dst;
      else if (dst == _available_for_all_streams.get())
        return src;
      else 
        return nullptr;
    }
  };
  // 如果是split出来的
  // 我们把剩余部分从
  // 1、available table 以及 2、info
  // 中删去并重新合并（向now靠齐）
  if (data_info->prev != nullptr && !data_info->prev->allocated()) {
    auto prev_info = data_info->prev; 
    HT_ASSERT(prev_info->status >= OccupationStatus::AVAILABLE_FOR_ALLOC_STREAM)
      << "Prev info status should be available to alloc"
      << ", but found " << prev_info->status;
    HT_ASSERT(prev_info->cached_pool)
      << "Prev info cached pool shouldn't be nullptr";
    auto try_merge_table = table_selection(table, prev_info->cached_pool);
    if (try_merge_table) {
      table = try_merge_table;
      auto prev_data_ptr = DataPtr{prev_info->ptr, prev_info->num_bytes, device(), prev_info->id};
      auto table_it = prev_info->cached_pool->table.find(prev_data_ptr);
      HT_ASSERT(table_it != prev_info->cached_pool->table.end())
        << "Cannot find " << prev_data_ptr << " in the target table";
      // mempool debug use
      // HT_LOG_INFO << "[Merge] remove forever: " << *table_it;
      prev_info->cached_pool->table.erase(table_it);
      data_info->ptr = prev_info->ptr;
      data_info->num_bytes += prev_info->num_bytes;
      data_info->prev = prev_info->prev;
      if (prev_info->prev != nullptr)
        prev_info->prev->next = data_info;
      auto info_it = _data_ptr_info.find(prev_info->id);
      HT_ASSERT(info_it != _data_ptr_info.end())
        << "Cannot find " << prev_data_ptr << " in the info";
      _data_ptr_info.erase(info_it);
    }
  }
  if (data_info->next != nullptr && !data_info->next->allocated()){
    auto next_info = data_info->next; 
    HT_ASSERT(next_info->status >= OccupationStatus::AVAILABLE_FOR_ALLOC_STREAM)
      << "Next info status should be available to alloc"
      << ", but found " << next_info->status;
    HT_ASSERT(next_info->cached_pool)
      << "Next info cached pool shouldn't be nullptr";
    auto try_merge_table = table_selection(table, next_info->cached_pool);
    if (try_merge_table) {
      table = try_merge_table;
      auto next_data_ptr = DataPtr{next_info->ptr, next_info->num_bytes, device(), next_info->id};
      auto table_it = next_info->cached_pool->table.find(next_data_ptr);
      HT_ASSERT(table_it != next_info->cached_pool->table.end())
        << "Cannot find " << next_data_ptr << " in the target table";
      // mempool debug use
      // HT_LOG_INFO << "[Merge] remove forever: " << *table_it;
      next_info->cached_pool->table.erase(table_it);
      data_info->num_bytes += next_info->num_bytes;
      data_info->next = next_info->next;
      if (next_info->next != nullptr)
        next_info->next->prev = data_info;
      auto info_it = _data_ptr_info.find(next_info->id);
      HT_ASSERT(info_it != _data_ptr_info.end())
        << "Cannot find " << next_data_ptr << " in the info";
      _data_ptr_info.erase(info_it);
    }
  }
  HT_ASSERT(table)
    << "Table shouldn't be nullptr";
  // 修正status
  if (table == _available_for_all_streams.get()) 
    data_info->status = OccupationStatus::AVAILABLE_FOR_ALL_STREAM;
  else
    data_info->status = OccupationStatus::AVAILABLE_FOR_ALLOC_STREAM;
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
      auto info = it->second;
      if ((--info->free_event_cnt) == 0) {
        // borrow data
        // 直接删除即可
        if (info->deleter) {
          info->deleter(DataPtr{info->ptr, info->num_bytes, device(), data_ptr_id});
          _data_ptr_info.erase(it);
        } 
        // alloc data
        // 考虑放到all stream available table中cache住
        else {
          DataPtrLookupTable* target_table = _available_for_all_streams.get();
          info->refresh();
          if (info->is_split())
            target_table = TryMerge(info, target_table);
          auto data_ptr = DataPtr{info->ptr, info->num_bytes, device(), data_ptr_id};
          HT_ASSERT(data_ptr_id == info->id)
            << "The data ptr id and the info id are mismatched"; 
          info->cached_pool = target_table;
          // mempool debug use
          // HT_LOG_INFO << "[Insert] sync free event then insert to table: " << data_ptr;
          InsertAvailable(data_ptr, *target_table); 
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
  auto info = it->second;
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
    auto info = it->second;
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

size_t ParseMaxSplitSize() {
  const char* max_split_str = std::getenv("HETU_MAX_SPLIT_SIZE_MB");
  size_t max_split_size_mb;
  if (max_split_str != NULL) {
      try {
        max_split_size_mb = std::stoi(max_split_str);
        // TODO: 敲定一个最小值....
      } catch (const std::exception& e) {
        HT_LOG_WARN
          << "Invalid HETU_MAX_SPLIT_SIZE_MB: " << max_split_str << " is set" 
          << ", please provide an integer whose value > 20"
          << ", default value will be used in this process";
      }
  } 
  // 默认设置为最大值
  else {
    max_split_size_mb = std::numeric_limits<size_t>::max() / 1024 / 1024;
  }
  
  return max_split_size_mb * 1024 * 1024;
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
