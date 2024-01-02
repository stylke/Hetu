#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/kernel/Vectorized.cuh"

namespace hetu {
namespace impl {

void ArangeCuda(double start, double step, NDArray& output, const Stream& stream) {

  size_t size = output->numel();
  if (size == 0)
    return;
  bool contiguous = output->is_contiguous();
  if (contiguous) {
    HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
      output->dtype(), spec_t, "RangeCuda", [&]() {
        launch_vectorized_set_kernel(size, output->data_ptr<spec_t>(), stream,
                                     [=] __device__ (int x) -> spec_t {
                                       return static_cast<spec_t>(start + step * size_t(x));
                                     });
    });
  } else {
    constexpr int unroll_factor = sizeof(DataType2Size(output->dtype())) >= 4 ? 2 : 4;
    dim3 block(128);
    dim3 grid(DIVUP(size, unroll_factor * block.x));
    NDArray out_offset_calculator_arr;
    OffsetCalculator *out_offset_calculator;
    std::tie(out_offset_calculator_arr, out_offset_calculator) = 
      AllocOffsetCalculator(output, stream);
    CUDAStream cuda_stream(stream);
    HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
      output->dtype(), spec_t, "RangeCuda", [&]() {
        set_kernel<128, unroll_factor><<<grid, block, 0, cuda_stream>>>(
          size, output->data_ptr<spec_t>(),
          [=] __device__ (int x) -> spec_t {
            return static_cast<spec_t>(start + step * size_t(x));
          }, out_offset_calculator);
    });
    NDArray::MarkUsedBy({out_offset_calculator_arr}, stream);
  }
  NDArray::MarkUsedBy({output}, stream);
}

} // namespace impl
} // namespace hetu
