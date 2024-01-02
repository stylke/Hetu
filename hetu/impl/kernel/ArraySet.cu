#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/offset_calculator.cuh"
#include "hetu/impl/kernel/Vectorized.cuh"

namespace hetu {
namespace impl {

void ArraySetCuda(NDArray& data, double value, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(data);
  size_t size = data->numel();
  if (size == 0)
    return;
  bool contiguous = data->is_contiguous();
  if (contiguous) {
    HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
      data->dtype(), spec_t, "ArraySetCuda", [&]() {
        launch_vectorized_set_kernel(size, data->data_ptr<spec_t>(), stream,
                                     [=] __device__ (int /*idx*/) -> spec_t {
                                       return static_cast<spec_t>(value);
                                     });
    });
  } else {
    constexpr int unroll_factor = sizeof(DataType2Size(data->dtype())) >= 4 ? 2 : 4;
    dim3 block(128);
    dim3 grid(DIVUP(size, unroll_factor * block.x));
    NDArray data_offset_calculator_arr;
    OffsetCalculator *data_offset_calculator;
    std::tie(data_offset_calculator_arr, data_offset_calculator) = 
      AllocOffsetCalculator(data, stream);
    CUDAStream cuda_stream(stream);
    HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
      data->dtype(), spec_t, "ArraySetCuda", [&]() {
        set_kernel<128, unroll_factor><<<grid, block, 0, cuda_stream>>>(
          size, data->data_ptr<spec_t>(),
          [=] __device__ (int /*idx*/) -> spec_t {
            return static_cast<spec_t>(value);
          }, data_offset_calculator);
    });
    NDArray::MarkUsedBy({data_offset_calculator_arr}, stream);
  }
  NDArray::MarkUsedBy({data}, stream);
}

} // namespace impl
} // namespace hetu
