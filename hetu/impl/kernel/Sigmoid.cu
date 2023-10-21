#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/cuda_math.h"

namespace hetu {
namespace impl {

template <typename spec_t>
__global__ void sigmoid_kernel(const spec_t* input, size_t size,
                               spec_t* output) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  output[idx] = 1.0 / (1.0 + hetu::cuda::cuda_exp(-input[idx]));
}

template <typename spec_t>
__global__ void sigmoid_grad_kernel(const spec_t* output_grad, const spec_t* output,
                                    size_t size, spec_t* input_grad) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  input_grad[idx] = output_grad[idx] * output[idx] * (1 - output[idx]);
}

void SigmoidCuda(const NDArray& input, NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_EXCHANGABLE(input, output);

  size_t size = output->numel();
  if (size == 0)
    return;
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  HT_DISPATCH_FLOATING_TYPES(input->dtype(), spec_t, "SigmoidCuda", [&]() {
    sigmoid_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
      input->data_ptr<spec_t>(), size, output->data_ptr<spec_t>());
  });
}

void SigmoidGradientCuda(const NDArray& out_grad, const NDArray& output, NDArray& in_grad, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(out_grad);
  HT_ASSERT_SAME_DEVICE(out_grad, output);
  HT_ASSERT_SAME_DEVICE(out_grad, in_grad);
  HT_ASSERT_EXCHANGABLE(out_grad, output);
  HT_ASSERT_EXCHANGABLE(out_grad, in_grad);

  size_t size = output->numel();
  if (size == 0)
    return;
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  HT_DISPATCH_FLOATING_TYPES(out_grad->dtype(), spec_t, "SigmoidCuda", [&]() {
    sigmoid_grad_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
      out_grad->data_ptr<spec_t>(), output->data_ptr<spec_t>(),
      size, in_grad->data_ptr<spec_t>());
  });
}

} // namespace impl
} // namespace hetu
