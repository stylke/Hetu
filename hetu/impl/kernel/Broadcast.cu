#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"

namespace hetu {
namespace impl {

template <typename spec_t>
__global__ void broadcast_kernel(const spec_t* input, size_t input_size,
                                 size_t size, spec_t* output) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    output[idx] = input[idx % input_size];
}

template <typename spec_t>
__global__ void broadcast_gradient_kernel(const spec_t* input,
                                          size_t input_size, size_t size,
                                          spec_t* output) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    output[idx] = input[idx];
}

void BroadcastCuda(const NDArray& input, NDArray& output,
                   const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);

  size_t size = output->numel();
  size_t input_size = input->numel();
  if (size == 0 || input_size == 0)
    return;
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "BroadcastCuda", [&]() {
      broadcast_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        input->data_ptr<spec_t>(), input_size, size,
        output->data_ptr<spec_t>());
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

void BroadcastGradientCuda(const NDArray& input, NDArray& output,
                           const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);

  size_t size = output->numel();
  size_t input_size = input->numel();
  if (size == 0 || input_size == 0)
    return;
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "BroadcastGradientCuda", [&]() {
      broadcast_gradient_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        input->data_ptr<spec_t>(), input_size, size,
        output->data_ptr<spec_t>());
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

} // namespace impl
} // namespace hetu
