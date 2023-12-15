#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/offset_calculator.cuh"

namespace hetu {
namespace impl {

std::tuple<NDArray, OffsetCalculator*>
AllocOffsetCalculator(const NDArray& arr, const Stream& stream) {
  auto device_id = arr->device().index();
  CUDAStream cuda_stream(stream);
  if (arr->is_contiguous()) {
    NDArray offset_calculator_arr = NDArray::empty({static_cast<int64_t>(sizeof(OffsetCalculator))},
                                                   Device(kCUDA, device_id), kByte, kBlockingStream);
    auto offset_calculator = reinterpret_cast<OffsetCalculator*>(offset_calculator_arr->raw_data_ptr());
    hetu::cuda::CUDADeviceGuard guard(device_id);
    trivial_constructor<<<1, 1, 0, cuda_stream>>>(offset_calculator);
    return {offset_calculator_arr, offset_calculator};
  } else {
    size_t ndim = arr->ndim();
    size_t alloc_size = ndim * sizeof(int64_t);
    auto shape_arr = 
        hetu::cuda::to_int64_ndarray(arr->shape(), device_id);
    auto stride_arr =
        hetu::cuda::to_int64_ndarray(arr->stride(), device_id);
    NDArray offset_calculator_arr = NDArray::empty({static_cast<int64_t>(sizeof(StridedOffsetCalculator))},
                                                   Device(kCUDA, device_id), kByte, kBlockingStream);
    auto offset_calculator = reinterpret_cast<StridedOffsetCalculator*>(offset_calculator_arr->raw_data_ptr());
    hetu::cuda::CUDADeviceGuard guard(device_id);
    strided_constructor<<<1, 1, 0, cuda_stream>>>(offset_calculator, ndim,
                                                  shape_arr->data_ptr<int64_t>(), stride_arr->data_ptr<int64_t>());
    NDArray::MarkUsedBy({shape_arr, stride_arr}, stream);
    return {offset_calculator_arr, offset_calculator};
  }
}

} // namespace impl
} // namespace hetu