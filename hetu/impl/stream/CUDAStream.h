#pragma once

#include "hetu/core/stream.h"
#include "hetu/impl/utils/cuda_utils.h"

namespace hetu {
namespace impl {

// Note: we do not inherit CUDAStream from Stream
// to avoid including the CUDA-related headers in core
class CUDAStream final {
 public:
  CUDAStream(const Stream& stream);

  void Sync() {
    CudaStreamSynchronize(cuda_stream());
  }

  cudaStream_t cuda_stream() const noexcept;

  // Implicit call to `cuda_stream`
  inline operator cudaStream_t() const {
    return cuda_stream();
  }

  inline DeviceIndex device_id() const noexcept {
    return _device_id;
  }

  inline StreamIndex stream_id() const noexcept {
    return _stream_id;
  }

 private:
  const DeviceIndex _device_id;
  const StreamIndex _stream_id;
};

inline CUDAStream GetCUDAStream(int32_t device_id, StreamIndex stream_id) {
  return CUDAStream(Stream(Device(kCUDA, device_id), stream_id));
}

inline CUDAStream GetCUDAComputingStream(int32_t device_id) {
  return GetCUDAStream(device_id, kComputingStream);
}

int GetCUDADeiceCount();
void SynchronizeAllCUDAStreams(const Device& device = {});

class CUDAEvent final : public Event {
 public:
  CUDAEvent(Device device) : Event(device) {
    HT_ASSERT(device.is_cuda())
      << "CUDAEvent should be used with CUDA devices. "
      << "Got " << device;
    hetu::cuda::CUDADeviceGuard guard(_device.index());
    CudaEventCreate(&_event);
  }

  inline void Record(const Stream& stream) {
    hetu::cuda::CUDADeviceGuard guard(stream.device_index());
    CudaEventRecord(_event, CUDAStream(stream));
    _recorded = true;
  }

  inline void Sync() {
    HT_ASSERT(_recorded) << "Event has not been recorded";
    CudaEventSynchronize(_event);
  }

  inline void Block(const Stream& stream) {
    HT_ASSERT(_recorded) << "Event has not been recorded";
    hetu::cuda::CUDADeviceGuard guard(stream.device_index());
    CudaStreamWaitEvent(CUDAStream(stream), _event, 0);
  }

  inline int64_t TimeSince(const Event& event) const {
    const auto& e = reinterpret_cast<const CUDAEvent&>(event);
    HT_ASSERT(e._recorded) << "Start event has not been recorded";
    HT_ASSERT(_recorded) << "Stop event has not been recorded";
    float ms;
    CudaEventElapsedTime(&ms, e._event, _event);
    return static_cast<int64_t>(ms * 1000000);
  }

 private:
  cudaEvent_t _event;
  bool _recorded{false};
};

} // namespace impl
} // namespace hetu
