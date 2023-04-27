#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"

namespace hetu {
namespace impl {

template <typename spec_t>
__global__ void clamp_kernel(const spec_t* input, spec_t min_val, spec_t max_val, 
                             size_t size, spec_t* output) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  if (input[idx] < min_val)
    output[idx] = min_val;
  else if (input[idx] > max_val) {
    output[idx] = max_val;
  }
  else 
    output[idx] = input[idx];
}

template <typename spec_t>
__global__ void clamp_elewise_kernel(const spec_t* input, const spec_t* min_val, const spec_t* max_val, 
                                     size_t size, spec_t* output) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  if (input[idx] < min_val[idx])
    output[idx] = min_val[idx];
  else if (input[idx] > max_val[idx]) {
    output[idx] = max_val[idx];
  }
  else 
    output[idx] = input[idx];
}

void ClampCuda(const NDArray& input, double min_val, double max_val, NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_EXCHANGABLE(input, output);

  size_t size = input->numel();
  if (size == 0)
    return;
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "ClampCuda", [&]() {
      clamp_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        input->data_ptr<spec_t>(), min_val, max_val, size, output->data_ptr<spec_t>());
    });
}

void ClampElewiseCuda(const NDArray& input, const NDArray& min_val, const NDArray& max_val, NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_EXCHANGABLE(input, output);

  size_t size = input->numel();
  if (size == 0)
    return;
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "ClampCuda", [&]() {
      clamp_elewise_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        input->data_ptr<spec_t>(), min_val->data_ptr<spec_t>(), max_val->data_ptr<spec_t>(), 
        size, output->data_ptr<spec_t>());
    });
}

} // namespace impl
} // namespace hetu
