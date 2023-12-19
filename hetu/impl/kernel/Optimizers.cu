#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/ndarray_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/cuda_math.h"

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


template <typename spec_t>
__global__ void sgd_update_with_gradscaler_kernel(const spec_t* grad, const float* infinite_count, 
                                                  spec_t* param, float lr, size_t size) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  if (*infinite_count)
    return;
  param[idx] -= lr * grad[idx];
}

template <typename spec_t>
__global__ void momentum_update_with_gradscaler_kernel(const spec_t* grad, const float* infinite_count,
                                                       spec_t* param, spec_t* velocity, float lr,
                                                       float momentum, size_t size) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  if (*infinite_count)
    return;
  velocity[idx] = momentum * velocity[idx] - lr * grad[idx];
  param[idx] = param[idx] + velocity[idx];
}

template <typename spec_t>
__global__ void nesterov_momentum_update_with_gradscaler_kernel(const spec_t* grad, const float* infinite_count,
                                                                spec_t* param, spec_t* velocity,
                                                                float lr, float momentum, size_t size) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  if (*infinite_count)
    return;
  float temp = lr * grad[idx];
  velocity[idx] = momentum * (velocity[idx] - temp);
  param[idx] = param[idx] + velocity[idx] - temp;
}

void SGDUpdateWithGradScalerCuda(const NDArray& grad, const NDArray& infinite_count,
                                 NDArray& param, NDArray& velocity,
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
      sgd_update_with_gradscaler_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        grad->data_ptr<spec_t>(), infinite_count->data_ptr<float>(), 
        param->data_ptr<spec_t>(), lr, size);
    } else if (!nesterov) {
      momentum_update_with_gradscaler_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        grad->data_ptr<spec_t>(), infinite_count->data_ptr<float>(), param->data_ptr<spec_t>(),
        velocity->data_ptr<spec_t>(), lr, momentum, size);
    } else {
      nesterov_momentum_update_with_gradscaler_kernel<spec_t>
        <<<blocks, threads, 0, cuda_stream>>>(
          grad->data_ptr<spec_t>(), infinite_count->data_ptr<float>(), 
          param->data_ptr<spec_t>(), velocity->data_ptr<spec_t>(), lr, momentum, size);
    }
  });
}

template <typename spec_t>
__global__ void adam_update_kernel(const spec_t* grad, spec_t* param, spec_t* mean,
                                   spec_t* variance, int64_t* step, float lr, float beta1, 
                                   float beta2, float eps, float weight_decay, size_t size) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  mean[idx] = mean[idx] * beta1 + grad[idx] * (1 - beta1);
  variance[idx] = variance[idx] * beta2 + grad[idx] * grad[idx] * (1 - beta2);
  spec_t bias1 = spec_t(1 - hetu::cuda::cuda_pow(beta1, float(step[0])));
  spec_t bias2 = hetu::cuda::cuda_sqrt(spec_t(1 - hetu::cuda::cuda_pow(beta2, float(step[0]))));
  param[idx] -= lr * (mean[idx] / bias1) / 
                (hetu::cuda::cuda_sqrt(variance[idx]) / bias2 + eps);
}

void AdamCuda(const NDArray& grad, NDArray& param, NDArray& mean,
              NDArray& variance, NDArray& step, 
              float lr, float beta1, float beta2,
              float eps, float weight_decay,
              const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(grad);
  HT_ASSERT_CUDA_DEVICE(param);
  HT_ASSERT_CUDA_DEVICE(mean);
  HT_ASSERT_CUDA_DEVICE(variance);
  HT_ASSERT_SAME_DEVICE(grad, param);
  HT_ASSERT_SAME_DEVICE(grad, mean);
  HT_ASSERT_SAME_DEVICE(grad, variance);
  HT_ASSERT_EXCHANGABLE(grad, param);
  HT_ASSERT_EXCHANGABLE(grad, mean);
  HT_ASSERT_EXCHANGABLE(grad, variance);
  size_t size = grad->numel();
  if (size == 0)
    return;
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  HT_DISPATCH_FLOATING_TYPES(grad->dtype(), spec_t, "AdamUpdateCuda", [&]() {
    adam_update_kernel<spec_t>
        <<<blocks, threads, 0, cuda_stream>>>(
          grad->data_ptr<spec_t>(), param->data_ptr<spec_t>(), 
          mean->data_ptr<spec_t>(), variance->data_ptr<spec_t>(), 
          step->data_ptr<int64_t>(), lr, beta1, beta2, eps, weight_decay, size);
  });
  NDArray::add(step, 1, kBlockingStream, step);
  NDArray::MarkUsedBy({grad, param, mean, variance, step}, stream);
}

} // namespace impl
} // namespace hetu
