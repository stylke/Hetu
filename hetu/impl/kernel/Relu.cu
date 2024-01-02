#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/offset_calculator.cuh"
#include "hetu/impl/kernel/Vectorized.cuh"

namespace hetu {
namespace impl {

void ReluCuda(const NDArray& input, NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_SAME_SHAPE(input, output);

  size_t size = output->numel();
  if (size == 0)
    return;
  bool contiguous = input->is_contiguous() && output->is_contiguous();
  if (contiguous) {
    HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
      input->dtype(), spec_t, "ReluCuda", [&]() {
        launch_vectorized_unary_kernel(input->data_ptr<spec_t>(), size,
                                       output->data_ptr<spec_t>(), stream,
                                       [=] __device__ (spec_t x) -> spec_t {
                                         return (double(x <= 0)) ? spec_t(0) : x;
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
    HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
      input->dtype(), spec_t, "ReluCuda", [&]() {
        unary_kernel<128, unroll_factor><<<grid, block, 0, cuda_stream>>>(
          input->data_ptr<spec_t>(), size, output->data_ptr<spec_t>(),
          [=] __device__ (spec_t x) -> spec_t {
            return (double(x <= 0)) ? spec_t(0) : x;
          }, in_offset_calculator, out_offset_calculator);
    });
    NDArray::MarkUsedBy({in_offset_calculator_arr, out_offset_calculator_arr}, stream);
  }
  NDArray::MarkUsedBy({input, output}, stream);
}

template <typename spec_t>
__global__ void relu_gradient_kernel(const spec_t* input, const spec_t* output_grad,
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
  output[out_offset] = (double(input[in_offset]) <= 0) ? 0 : output_grad[out_grad_offset];
}

void ReluGradientCuda(const NDArray& input, const NDArray& output_grad,
                      NDArray& input_grad, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output_grad);
  HT_ASSERT_SAME_DEVICE(input, input_grad);
  HT_ASSERT_SAME_SHAPE(input, output_grad);
  HT_ASSERT_SAME_SHAPE(input, input_grad);

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
    input->dtype(), spec_t, "ReluGradientCuda", [&]() {
      relu_gradient_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        input->data_ptr<spec_t>(), output_grad->data_ptr<spec_t>(), size,
        input_grad->data_ptr<spec_t>(), in_offset_calculator,
        out_grad_offset_calculator, in_grad_offset_calculator);
    });
  NDArray::MarkUsedBy({input, output_grad, input_grad, in_offset_calculator_arr,
                      out_grad_offset_calculator_arr, in_grad_offset_calculator_arr}, stream);
}

} // namespace impl
} // namespace hetu
