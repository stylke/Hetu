#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/ndarray_utils.h"
#include "hetu/impl/utils/cuda_utils.h"

namespace hetu {
namespace impl {

template <typename spec_t>
__global__ void sgd_update_kernel(const spec_t* grad, spec_t* param, float lr,
                                  size_t size) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  param[idx] -= lr * grad[idx];
}

template <typename spec_t>
__global__ void momentum_update_kernel(const spec_t* grad, spec_t* param,
                                       spec_t* velocity, float lr,
                                       float momentum, size_t size) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  velocity[idx] = momentum * velocity[idx] - lr * grad[idx];
  param[idx] = param[idx] + velocity[idx];
}

template <typename spec_t>
__global__ void nesterov_momentum_update_kernel(const spec_t* grad,
                                                spec_t* param, spec_t* velocity,
                                                float lr, float momentum,
                                                size_t size) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  float temp = lr * grad[idx];
  velocity[idx] = momentum * (velocity[idx] - temp);
  param[idx] = param[idx] + velocity[idx] - temp;
}

void SGDUpdateCuda(const NDArray& grad, NDArray& param, NDArray& velocity,
                   float lr, float momentum, bool nesterov,
                   const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(grad);
  HT_ASSERT_CUDA_DEVICE(param);
  HT_ASSERT_SAME_DEVICE(grad, param);
  HT_ASSERT_EXCHANGABLE(grad, param);
  if (momentum != 0) {
    HT_ASSERT_CUDA_DEVICE(velocity);
    HT_ASSERT_SAME_DEVICE(velocity, param);
    HT_ASSERT_EXCHANGABLE(velocity, param);
  }
  size_t size = grad->numel();
  if (size == 0)
    return;
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  HT_DISPATCH_FLOATING_TYPES(grad->dtype(), spec_t, "SGDUpdateCuda", [&]() {
    if (momentum == 0) {
      sgd_update_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        grad->data_ptr<spec_t>(), param->data_ptr<spec_t>(), lr, size);
    } else if (!nesterov) {
      momentum_update_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        grad->data_ptr<spec_t>(), param->data_ptr<spec_t>(),
        velocity->data_ptr<spec_t>(), lr, momentum, size);
    } else {
      nesterov_momentum_update_kernel<spec_t>
        <<<blocks, threads, 0, cuda_stream>>>(
          grad->data_ptr<spec_t>(), param->data_ptr<spec_t>(),
          velocity->data_ptr<spec_t>(), lr, momentum, size);
    }
  });
}

} // namespace impl
} // namespace hetu
