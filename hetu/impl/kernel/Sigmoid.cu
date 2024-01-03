#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/cuda_math.h"
#include "hetu/impl/utils/offset_calculator.cuh"
#include "hetu/impl/kernel/Vectorized.cuh"

namespace hetu {
namespace impl {

void SigmoidCuda(const NDArray& input, NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_SAME_SHAPE(input, output);

  size_t size = output->numel();
  if (size == 0)
    return;
  HT_DISPATCH_FLOATING_TYPES(
    input->dtype(), spec_t, "SigmoidCuda", [&]() {
      launch_loop_kernel<spec_t, spec_t>(input, output, size, stream,
                                         [=] __device__ (spec_t x) -> spec_t {
                                           return 1.0 / (1.0 + hetu::cuda::cuda_exp(-x));
                                         });
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

template <typename spec_t>
__global__ void sigmoid_grad_kernel(const spec_t* output_grad, const spec_t* output,
                                    size_t size, spec_t* input_grad,
                                    const OffsetCalculator* out_grad_offset_calculator,
                                    const OffsetCalculator* out_offset_calculator,
                                    const OffsetCalculator* in_grad_offset_calculator) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  auto out_grad_offset = out_grad_offset_calculator->get(idx);
  auto out_offset = out_offset_calculator->get(idx);
  auto in_grad_offset = in_grad_offset_calculator->get(idx);
  input_grad[in_grad_offset] = output_grad[out_grad_offset] * output[out_offset] * (1 - output[out_offset]);
}

void SigmoidGradientCuda(const NDArray& out_grad, const NDArray& output, NDArray& in_grad, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(out_grad);
  HT_ASSERT_SAME_DEVICE(out_grad, output);
  HT_ASSERT_SAME_DEVICE(out_grad, in_grad);
  HT_ASSERT_SAME_SHAPE(out_grad, output);
  HT_ASSERT_SAME_SHAPE(out_grad, in_grad);

  size_t size = output->numel();
  if (size == 0)
    return;
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  NDArray out_grad_offset_calculator_arr, out_offset_calculator_arr,
          in_grad_offset_calculator_arr;
  OffsetCalculator *out_grad_offset_calculator, *out_offset_calculator,
                   *in_grad_offset_calculator;
  std::tie(out_grad_offset_calculator_arr, out_grad_offset_calculator) =
    AllocOffsetCalculator(out_grad, stream);
  std::tie(out_offset_calculator_arr, out_offset_calculator) = 
    AllocOffsetCalculator(output, stream);
  std::tie(in_grad_offset_calculator_arr, in_grad_offset_calculator) = 
    AllocOffsetCalculator(in_grad, stream);
  HT_DISPATCH_FLOATING_TYPES(out_grad->dtype(), spec_t, "SigmoidCuda", [&]() {
    sigmoid_grad_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
      out_grad->data_ptr<spec_t>(), output->data_ptr<spec_t>(),
      size, in_grad->data_ptr<spec_t>(), out_grad_offset_calculator,
      out_offset_calculator, in_grad_offset_calculator);
  });
  NDArray::MarkUsedBy({out_grad, output, in_grad, out_grad_offset_calculator_arr,
                      out_offset_calculator_arr, in_grad_offset_calculator_arr}, stream);
}

} // namespace impl
} // namespace hetu
