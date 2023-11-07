#include "hetu/impl/utils/cuda_utils.h"

namespace hetu {
namespace cuda {

NDArray to_int64_ndarray(const std::vector<int64_t>& vec,
                         DeviceIndex device_id) {
  auto ret = NDArray::empty({static_cast<int64_t>(vec.size())},
                            Device(kCUDA, device_id), kInt64, kBlockingStream);
  hetu::cuda::CUDADeviceGuard guard(device_id);
  CudaMemcpy(ret->raw_data_ptr(), vec.data(), vec.size() * sizeof(int64_t),
             cudaMemcpyHostToDevice);
  return ret;
}

NDArray to_int64_ndarray(const int64_t* from, size_t n, DeviceIndex device_id) {
  auto ret = NDArray::empty({static_cast<int64_t>(n)}, Device(kCUDA, device_id),
                            kInt64, kBlockingStream);
  CudaMemcpy(ret->raw_data_ptr(), from, n * sizeof(int64_t),
             cudaMemcpyHostToDevice);
  return ret;
}

} // namespace cuda
} // namespace hetu
