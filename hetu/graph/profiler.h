#pragma once

#include "hetu/core/device.h"
#include "hetu/impl/memory/CUDACachingMemoryPool.cuh"

namespace hetu {
namespace graph {

enum class MEMORY_PROFILE_LEVEL : int8_t {
  MICRO_BATCH = 0,
  INFO,
  WARN
};

class CUDAMemoryInfo {
  public:
    // 单位都是MiB
    size_t mempool_reserved{0};
    size_t mempool_allocated{0};
    size_t all_reserved{0};
    size_t limit{0};
};

class MicroBatchMemoryInfo {
  public:
    // 单位都是MiB
    bool is_forward;
    size_t stage_id;
    size_t micro_batch_id;
    CUDAMemoryInfo begin_memory_info;
    CUDAMemoryInfo end_memory_info;
};

class CUDAProfiler {
  public:
    CUDAProfiler(const Device& device)
    : _device(device) {
      _mempool =  std::dynamic_pointer_cast<hetu::impl::CUDACachingMemoryPool>(GetMemoryPool(device));  
    }
    
    // profile memory
    CUDAMemoryInfo GetCurrMemoryInfo();
    void PrintCurrMemoryInfo(const std::string& prefix);

    // profile NVLink
    void PrintNvlinkStart();
    void PrintNvlinkEnd();

  protected:
    Device _device;

    // profile memory
    std::shared_ptr<hetu::impl::CUDACachingMemoryPool> _mempool;

    // profile NVLink
    unsigned int _device_count = 0; // 有多少个GPU
    std::vector<unsigned int> _nvlink_counts; // 每个GPU有多少条NVLink
    std::vector<std::vector<unsigned long long>> _nvlink_txs; // 记录执行通信代码片段前每个GPU每条NVLink的Raw Tx
    std::vector<std::vector<unsigned long long>> _nvlink_rxs; // 记录执行通信代码片段前每个GPU每条NVLink的Raw Rx
};

std::shared_ptr<CUDAProfiler> GetCUDAProfiler(const Device& device);

std::ostream& operator<<(std::ostream& os, const CUDAMemoryInfo& memory_info);

std::ostream& operator<<(std::ostream& os, const MicroBatchMemoryInfo& micro_batch_memory_info);

} // namespace graph
} // namespace hetu