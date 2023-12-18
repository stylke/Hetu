#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/offset_calculator.cuh"

namespace hetu {
namespace impl {

namespace {

static std::once_flag offset_calculator_init_flag;
static NDArray trivial_offset_calculator_arr;
static OffsetCalculator* trivial_offset_calculator;
static LFUCache lfu_cache;

static void OffCalcInitOnce(const Stream& stream) {
  std::call_once(offset_calculator_init_flag, [](const Stream& stream) {
    lfu_cache = LFUCache(HT_LFU_CAPACITY);
    CUDAStream cuda_stream(stream);
    trivial_offset_calculator_arr =
      NDArray::empty({static_cast<int64_t>(sizeof(OffsetCalculator))},
                      Device(kCUDA, stream.device_index()), kByte, stream.stream_index());
    trivial_offset_calculator =
      trivial_offset_calculator_arr->data_ptr<OffsetCalculator>();
    hetu::cuda::CUDADeviceGuard guard(stream.device_index());
    trivial_constructor<<<1, 1, 0, cuda_stream>>>(trivial_offset_calculator);
  }, stream);
}

} // namespace

std::tuple<NDArray, OffsetCalculator*>
AllocOffsetCalculator(const NDArray& arr, const Stream& stream) {
  OffCalcInitOnce(stream);
  if (arr->is_contiguous()) {
    return {trivial_offset_calculator_arr, trivial_offset_calculator};
  }
  NDArray offset_calculator_arr;
  StridedOffsetCalculator* offset_calculator;
  auto shape_arr_host = arr->shape();
  auto stride_arr_host = arr->stride();
  std::tie(offset_calculator_arr, offset_calculator) =
      lfu_cache.get(shape_arr_host, stride_arr_host);
  if (!offset_calculator) {
    auto device_id = arr->device().index();
    CUDAStream cuda_stream(stream);
    size_t ndim = arr->ndim();
    size_t alloc_size = ndim * sizeof(int64_t);
    auto shape_arr = 
        hetu::cuda::to_int64_ndarray(arr->shape(), device_id);
    auto stride_arr =
        hetu::cuda::to_int64_ndarray(arr->stride(), device_id);
    offset_calculator_arr = NDArray::empty({static_cast<int64_t>(sizeof(StridedOffsetCalculator))},
                                           Device(kCUDA, device_id), kByte, stream.stream_index());
    offset_calculator = offset_calculator_arr->data_ptr<StridedOffsetCalculator>();
    hetu::cuda::CUDADeviceGuard guard(device_id);
    strided_constructor<<<1, 1, 0, cuda_stream>>>(offset_calculator, ndim,
                                                  shape_arr->data_ptr<int64_t>(), stride_arr->data_ptr<int64_t>());
    NDArray::MarkUsedBy({shape_arr, stride_arr}, stream);
    lfu_cache.put(shape_arr_host, stride_arr_host, offset_calculator_arr, offset_calculator);
  }
  return {offset_calculator_arr, offset_calculator};
}

} // namespace impl
} // namespace hetu