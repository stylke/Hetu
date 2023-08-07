#include "hetu/core/ndarray.h"
#include "hetu/impl/cuda/CUDARand.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/random/CUDARandomState.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include <mutex>

namespace hetu {
namespace impl {

template <typename spec_t>
__global__ void dropout_kernel(const spec_t* input, spec_t* output,
                               float drop_rate, size_t size,
                               CUDARandomState rand_state) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  curandStatePhilox4_32_10_t state;
  curand_init(rand_state.seed, idx, rand_state.offset, &state);
  float temp = curand_uniform(&state);
  float keep_mask = (float) (temp >= drop_rate);
  output[idx] = input[idx] * keep_mask / (1 - drop_rate);
}

template <typename spec_t>
__global__ void dropout_gradient_kernel(const spec_t* grad,
                                        const spec_t* fw_output, spec_t* output,
                                        float drop_rate, size_t size) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  spec_t fw_out = fw_output[idx];
  spec_t keep_mask = (spec_t) (fw_out > 1e-20 || fw_out < -1e-20);
  output[idx] = grad[idx] * keep_mask  / (1 - drop_rate);
}

void DropoutCuda(const NDArray& input, double drop_rate, uint64_t seed,
                 NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(output, output);
  HT_ASSERT_EXCHANGABLE(input, output);
  size_t size = input->numel();
  if (size == 0)
    return;
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  HT_DISPATCH_FLOATING_TYPES(input->dtype(), spec_t, "DropoutCuda", [&]() {
    dropout_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
      input->data_ptr<spec_t>(), output->data_ptr<spec_t>(),
      static_cast<float>(drop_rate), size,
      GetCUDARandomState(cuda_stream.device_id(), seed, 4));
  });
}

void DropoutGradientWithRecomputationCuda(const NDArray& grad, double drop_rate,
                                          uint64_t seed, NDArray& output,
                                          const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(grad);
  HT_ASSERT_SAME_DEVICE(grad, output);
  HT_ASSERT_EXCHANGABLE(grad, output);
  size_t size = grad->numel();
  if (size == 0)
    return;
  HT_ASSERT(seed != 0)
    << "Gradient fn of dropout with recomputation "
    << "must be called with an explicitly provided random seed";
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  HT_DISPATCH_FLOATING_TYPES(grad->dtype(), spec_t, "DropoutCuda", [&]() {
    dropout_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
      grad->data_ptr<spec_t>(), output->data_ptr<spec_t>(),
      static_cast<float>(drop_rate), size,
      GetCUDARandomState(cuda_stream.device_id(), seed, 4));
  });
}

void DropoutGradientCuda(const NDArray& grad, const NDArray& fw_output,
                         double drop_rate, NDArray& output,
                         const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(grad);
  HT_ASSERT_SAME_DEVICE(grad, fw_output);
  HT_ASSERT_SAME_DEVICE(grad, output);
  HT_ASSERT_EXCHANGABLE(grad, fw_output);
  HT_ASSERT_EXCHANGABLE(grad, output);
  size_t size = grad->numel();
  if (size == 0)
    return;
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  HT_DISPATCH_FLOATING_TYPES(
    grad->dtype(), spec_t, "DropoutGradientCuda", [&]() {
      dropout_gradient_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        grad->data_ptr<spec_t>(), fw_output->data_ptr<spec_t>(),
        output->data_ptr<spec_t>(), static_cast<float>(drop_rate), size);
    });
}

} // namespace impl
} // namespace hetu
