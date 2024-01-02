#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/cuda_math.h"
#include "hetu/impl/utils/offset_calculator.cuh"
#include "hetu/impl/kernel/Vectorized.cuh"

namespace hetu {
namespace impl {

void ExpCuda(const NDArray& input, NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_SAME_SHAPE(input, output);

  size_t size = input->numel();
  if (size == 0)
    return;
  bool contiguous = input->is_contiguous() && output->is_contiguous();
  if (contiguous) {
    HT_DISPATCH_FLOATING_TYPES(
      input->dtype(), spec_t, "ExpCuda", [&]() {
        launch_vectorized_unary_kernel(input->data_ptr<spec_t>(), size,
                                       output->data_ptr<spec_t>(), stream,
                                       [=] __device__ (spec_t x) -> spec_t {
                                         return hetu::cuda::cuda_exp(x);
                                       });
    });
  } else {
    constexpr int unroll_factor = sizeof(DataType2Size(output->dtype())) >= 4 ? 2 : 4;
    dim3 block(128);
    dim3 grid(DIVUP(size, unroll_factor * block.x));
    NDArray in_offset_calculator_arr, out_offset_calculator_arr;
    OffsetCalculator *in_offset_calculator, *out_offset_calculator;
    std::tie(in_offset_calculator_arr, in_offset_calculator) =
      AllocOffsetCalculator(input, stream);
    std::tie(out_offset_calculator_arr, out_offset_calculator) = 
      AllocOffsetCalculator(output, stream);
    CUDAStream cuda_stream(stream);
    HT_DISPATCH_FLOATING_TYPES(
      input->dtype(), spec_t, "ExpCuda", [&]() {
        unary_kernel<128, unroll_factor><<<grid, block, 0, cuda_stream>>>(
          input->data_ptr<spec_t>(), size, output->data_ptr<spec_t>(),
          [=] __device__ (spec_t x) -> spec_t {
            return hetu::cuda::cuda_exp(x);
          }, in_offset_calculator, out_offset_calculator);
    });
    NDArray::MarkUsedBy({in_offset_calculator_arr, out_offset_calculator_arr}, stream);
  }
  NDArray::MarkUsedBy({input, output}, stream);
}

} // namespace impl
} // namespace hetu
