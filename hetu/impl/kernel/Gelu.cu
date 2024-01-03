#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/cuda_math.h"
#include "hetu/impl/utils/offset_calculator.cuh"
#include "hetu/impl/kernel/Vectorized.cuh"

#define SQRT_1_2  0.70710678118654757274f
#define pi 3.14159265358979323846f
#define e  2.71828182845904523536f

namespace hetu {
namespace impl {

void GeluCuda(const NDArray& input, NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_EXCHANGABLE(input, output);

  size_t size = output->numel();
  if (size == 0)
    return;
  HT_DISPATCH_FLOATING_TYPES(
    input->dtype(), spec_t, "GeluCuda", [&]() {
      launch_loop_kernel<spec_t, spec_t>(input, output, size, stream,
                                         [=] __device__ (spec_t x) -> spec_t {
                                           return x * 0.5f *
                                              (1.0f + hetu::cuda::cuda_erf(x * SQRT_1_2));
                                         });
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

template <typename spec_t>
__global__ void gelu_gradient_kernel(const spec_t* input, const spec_t* output_grad,
                                     size_t size, spec_t* output,
                                     const OffsetCalculator* in_offset_calculator,
                                     const OffsetCalculator* out_grad_offset_calculator,
                                     const OffsetCalculator* out_offset_calculator) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  auto in_offset = in_offset_calculator->get(idx);
  auto out_grad_offset = out_grad_offset_calculator->get(idx);
  auto out_offset = out_offset_calculator->get(idx);
  output[out_offset] = output_grad[out_grad_offset]*(0.5f + 0.5f * hetu::cuda::cuda_erf(input[in_offset] / hetu::cuda::cuda_sqrt(2.0)) + 
                     0.5f * input[in_offset]*(hetu::cuda::cuda_sqrt(2.0f) * 
                     hetu::cuda::cuda_exp(-0.5f * hetu::cuda::cuda_pow(input[in_offset], spec_t(2.0f))) / hetu::cuda::cuda_sqrt(pi)));
}

void GeluGradientCuda(const NDArray& input, const NDArray& output_grad,
                      NDArray& input_grad, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output_grad);
  HT_ASSERT_SAME_DEVICE(input, input_grad);
  HT_ASSERT_EXCHANGABLE(input, output_grad);
  HT_ASSERT_EXCHANGABLE(input, input_grad);

  size_t size = input_grad->numel();
  if (size == 0)
    return;
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  NDArray in_offset_calculator_arr, out_grad_offset_calculator_arr,
          in_grad_offset_calculator_arr;
  OffsetCalculator *in_offset_calculator, *out_grad_offset_calculator,
                   *in_grad_offset_calculator;
  std::tie(in_offset_calculator_arr, in_offset_calculator) =
    AllocOffsetCalculator(input, stream);
  std::tie(out_grad_offset_calculator_arr, out_grad_offset_calculator) = 
    AllocOffsetCalculator(output_grad, stream);
  std::tie(in_grad_offset_calculator_arr, in_grad_offset_calculator) = 
    AllocOffsetCalculator(input_grad, stream);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "GeluGradientCuda", [&]() {
      gelu_gradient_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        input->data_ptr<spec_t>(), output_grad->data_ptr<spec_t>(), size,
        input_grad->data_ptr<spec_t>(), in_offset_calculator,
        out_grad_offset_calculator, in_grad_offset_calculator);
  });
  NDArray::MarkUsedBy({input, output_grad, input_grad, in_offset_calculator_arr,
                      out_grad_offset_calculator_arr, in_grad_offset_calculator_arr}, stream);
}

} // namespace impl
} // namespace hetu
