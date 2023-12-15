#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/ndarray_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/cuda_math.h"
#include "hetu/impl/utils/offset_calculator.cuh"

namespace hetu {
namespace impl {

template <typename spec_t>
__global__ void sgd_update_kernel(const spec_t* grad, spec_t* param, float lr,
                                  size_t size,
                                  const OffsetCalculator* grad_offset_calculator,
                                  const OffsetCalculator* param_offset_calculator) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  auto grad_offset = grad_offset_calculator->get(idx);
  auto param_offset = param_offset_calculator->get(idx);
  param[param_offset] -= lr * grad[grad_offset];
}

template <typename spec_t>
__global__ void momentum_update_kernel(const spec_t* grad, spec_t* param,
                                       spec_t* velocity, float lr,
                                       float momentum, size_t size,
                                       const OffsetCalculator* grad_offset_calculator,
                                       const OffsetCalculator* param_offset_calculator,
                                       const OffsetCalculator* vel_offset_calculator) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  auto grad_offset = grad_offset_calculator->get(idx);
  auto param_offset = param_offset_calculator->get(idx);
  auto vel_offset = vel_offset_calculator->get(idx);
  velocity[vel_offset] = momentum * velocity[vel_offset] - lr * grad[grad_offset];
  param[param_offset] += velocity[vel_offset];
}

template <typename spec_t>
__global__ void nesterov_momentum_update_kernel(const spec_t* grad,
                                                spec_t* param, spec_t* velocity,
                                                float lr, float momentum, size_t size,
                                                const OffsetCalculator* grad_offset_calculator,
                                                const OffsetCalculator* param_offset_calculator,
                                                const OffsetCalculator* vel_offset_calculator) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  auto grad_offset = grad_offset_calculator->get(idx);
  auto param_offset = param_offset_calculator->get(idx);
  auto vel_offset = vel_offset_calculator->get(idx);
  float temp = lr * grad[grad_offset];
  velocity[vel_offset] = momentum * (velocity[vel_offset] - temp);
  param[param_offset] += (velocity[vel_offset] - temp);
}

void SGDUpdateCuda(const NDArray& grad, NDArray& param, NDArray& velocity,
                   float lr, float momentum, bool nesterov,
                   const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(grad);
  HT_ASSERT_CUDA_DEVICE(param);
  HT_ASSERT_SAME_DEVICE(grad, param);
  HT_ASSERT_SAME_SHAPE(grad, param);
  if (momentum != 0) {
    HT_ASSERT_CUDA_DEVICE(velocity);
    HT_ASSERT_SAME_DEVICE(velocity, param);
    HT_ASSERT_SAME_SHAPE(velocity, param);
  }
  size_t size = grad->numel();
  if (size == 0)
    return;
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  NDArray grad_offset_calculator_arr, param_offset_calculator_arr;
  OffsetCalculator *grad_offset_calculator, *param_offset_calculator;
  std::tie(grad_offset_calculator_arr, grad_offset_calculator) =
    AllocOffsetCalculator(grad, stream);
  std::tie(param_offset_calculator_arr, param_offset_calculator) = 
    AllocOffsetCalculator(param, stream);
  HT_DISPATCH_FLOATING_TYPES(grad->dtype(), spec_t, "SGDUpdateCuda", [&]() {
    if (momentum == 0) {
      sgd_update_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        grad->data_ptr<spec_t>(), param->data_ptr<spec_t>(), lr, size,
        grad_offset_calculator, param_offset_calculator);
    } else {
      NDArray vel_offset_calculator_arr;
      OffsetCalculator *vel_offset_calculator;
      std::tie(vel_offset_calculator_arr, vel_offset_calculator) =
        AllocOffsetCalculator(velocity, stream);
      if (!nesterov) {
        momentum_update_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
          grad->data_ptr<spec_t>(), param->data_ptr<spec_t>(),
          velocity->data_ptr<spec_t>(), lr, momentum, size,
          grad_offset_calculator, param_offset_calculator,
          vel_offset_calculator);
      } else {
        nesterov_momentum_update_kernel<spec_t>
          <<<blocks, threads, 0, cuda_stream>>>(
            grad->data_ptr<spec_t>(), param->data_ptr<spec_t>(),
            velocity->data_ptr<spec_t>(), lr, momentum, size,
            grad_offset_calculator, param_offset_calculator,
            vel_offset_calculator);
      }
      NDArray::MarkUsedBy({vel_offset_calculator_arr}, stream);
    }
  });
  NDArray::MarkUsedBy({grad, param, velocity, grad_offset_calculator_arr,
                       param_offset_calculator_arr}, stream);
}


template <typename spec_t>
__global__ void sgd_update_with_gradscaler_kernel(const spec_t* grad, const float* infinite_count, 
                                                  spec_t* param, float lr, size_t size,
                                                  const OffsetCalculator* grad_offset_calculator,
                                                  const OffsetCalculator* param_offset_calculator) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  if (*infinite_count)
    return;
  auto grad_offset = grad_offset_calculator->get(idx);
  auto param_offset = param_offset_calculator->get(idx);
  param[param_offset] -= lr * grad[grad_offset];
}

template <typename spec_t>
__global__ void momentum_update_with_gradscaler_kernel(const spec_t* grad, const float* infinite_count,
                                                       spec_t* param, spec_t* velocity, float lr,
                                                       float momentum, size_t size,
                                                       const OffsetCalculator* grad_offset_calculator,
                                                       const OffsetCalculator* param_offset_calculator,
                                                       const OffsetCalculator* vel_offset_calculator) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  if (*infinite_count)
    return;
  auto grad_offset = grad_offset_calculator->get(idx);
  auto param_offset = param_offset_calculator->get(idx);
  auto vel_offset = vel_offset_calculator->get(idx);
  velocity[vel_offset] = momentum * velocity[vel_offset] - lr * grad[grad_offset];
  param[param_offset] += velocity[vel_offset];
}

template <typename spec_t>
__global__ void nesterov_momentum_update_with_gradscaler_kernel(const spec_t* grad, const float* infinite_count,
                                                                spec_t* param, spec_t* velocity,
                                                                float lr, float momentum, size_t size,
                                                                const OffsetCalculator* grad_offset_calculator,
                                                                const OffsetCalculator* param_offset_calculator,
                                                                const OffsetCalculator* vel_offset_calculator) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  if (*infinite_count)
    return;
  auto grad_offset = grad_offset_calculator->get(idx);
  auto param_offset = param_offset_calculator->get(idx);
  auto vel_offset = vel_offset_calculator->get(idx);
  float temp = lr * grad[grad_offset];
  velocity[vel_offset] = momentum * (velocity[vel_offset] - temp);
  param[param_offset] += (velocity[vel_offset] - temp);
}

void SGDUpdateWithGradScalerCuda(const NDArray& grad, const NDArray& infinite_count,
                                 NDArray& param, NDArray& velocity,
                                 float lr, float momentum, bool nesterov,
                                 const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(grad);
  HT_ASSERT_CUDA_DEVICE(param);
  HT_ASSERT_SAME_DEVICE(grad, param);
  HT_ASSERT_SAME_SHAPE(grad, param);
  if (momentum != 0) {
    HT_ASSERT_CUDA_DEVICE(velocity);
    HT_ASSERT_SAME_DEVICE(velocity, param);
    HT_ASSERT_SAME_SHAPE(velocity, param);
  }
  size_t size = grad->numel();
  if (size == 0)
    return;
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  NDArray grad_offset_calculator_arr, param_offset_calculator_arr;
  OffsetCalculator *grad_offset_calculator, *param_offset_calculator;
  std::tie(grad_offset_calculator_arr, grad_offset_calculator) =
    AllocOffsetCalculator(grad, stream);
  std::tie(param_offset_calculator_arr, param_offset_calculator) = 
    AllocOffsetCalculator(param, stream);
  HT_DISPATCH_FLOATING_TYPES(grad->dtype(), spec_t, "SGDUpdateCuda", [&]() {
    if (momentum == 0) {
      sgd_update_with_gradscaler_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        grad->data_ptr<spec_t>(), infinite_count->data_ptr<float>(), 
        param->data_ptr<spec_t>(), lr, size, grad_offset_calculator,
        param_offset_calculator);
    } else {
      NDArray vel_offset_calculator_arr;
      OffsetCalculator *vel_offset_calculator;
      std::tie(vel_offset_calculator_arr, vel_offset_calculator) =
        AllocOffsetCalculator(velocity, stream);
      if (!nesterov) {
        momentum_update_with_gradscaler_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
          grad->data_ptr<spec_t>(), infinite_count->data_ptr<float>(), param->data_ptr<spec_t>(),
          velocity->data_ptr<spec_t>(), lr, momentum, size, grad_offset_calculator,
          param_offset_calculator, vel_offset_calculator);
      } else {
        nesterov_momentum_update_with_gradscaler_kernel<spec_t>
          <<<blocks, threads, 0, cuda_stream>>>(
            grad->data_ptr<spec_t>(), infinite_count->data_ptr<float>(), 
            param->data_ptr<spec_t>(), velocity->data_ptr<spec_t>(), lr, momentum, size,
            grad_offset_calculator, param_offset_calculator, vel_offset_calculator);
      }
      NDArray::MarkUsedBy({vel_offset_calculator_arr}, stream);
    }
  });
  NDArray::MarkUsedBy({grad, param, velocity, grad_offset_calculator_arr,
                      param_offset_calculator_arr}, stream);
}

template <typename spec_t>
__global__ void adam_update_kernel(const spec_t* grad, spec_t* param, spec_t* mean,
                                   spec_t* variance, int64_t* step, float lr, float beta1, 
                                   float beta2, float eps, float weight_decay, size_t size,
                                   const OffsetCalculator* grad_offset_calculator,
                                   const OffsetCalculator* param_offset_calculator,
                                   const OffsetCalculator* mean_offset_calculator,
                                   const OffsetCalculator* var_offset_calculator) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  auto grad_offset = grad_offset_calculator->get(idx);
  auto param_offset = param_offset_calculator->get(idx);
  auto mean_offset = mean_offset_calculator->get(idx);
  auto var_offset = var_offset_calculator->get(idx);
  mean[mean_offset] = mean[mean_offset] * beta1 + grad[grad_offset] * (1 - beta1);
  variance[var_offset] = variance[var_offset] * beta2 + grad[grad_offset] * grad[grad_offset] * (1 - beta2);
  spec_t bias1 = spec_t(1 - hetu::cuda::cuda_pow(beta1, float(step[0])));
  spec_t bias2 = hetu::cuda::cuda_sqrt(spec_t(1 - hetu::cuda::cuda_pow(beta2, float(step[0]))));
  param[param_offset] -= lr * (mean[mean_offset] / bias1) / 
                         (hetu::cuda::cuda_sqrt(variance[var_offset]) / bias2 + eps);
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
  NDArray grad_offset_calculator_arr, param_offset_calculator_arr,
          mean_offset_calculator_arr, variance_offset_calculator_arr;
  OffsetCalculator *grad_offset_calculator, *param_offset_calculator,
                   *mean_offset_calculator, *variance_offset_calculator;
  std::tie(grad_offset_calculator_arr, grad_offset_calculator) =
    AllocOffsetCalculator(grad, stream);
  std::tie(param_offset_calculator_arr, param_offset_calculator) = 
    AllocOffsetCalculator(param, stream);
  std::tie(mean_offset_calculator_arr, mean_offset_calculator) = 
    AllocOffsetCalculator(mean, stream);
  std::tie(variance_offset_calculator_arr, variance_offset_calculator) = 
    AllocOffsetCalculator(variance, stream);
  HT_DISPATCH_FLOATING_TYPES(grad->dtype(), spec_t, "AdamUpdateCuda", [&]() {
    adam_update_kernel<spec_t>
        <<<blocks, threads, 0, cuda_stream>>>(
          grad->data_ptr<spec_t>(), param->data_ptr<spec_t>(), 
          mean->data_ptr<spec_t>(), variance->data_ptr<spec_t>(), 
          step->data_ptr<int64_t>(), lr, beta1, beta2, eps, weight_decay, size,
          grad_offset_calculator, param_offset_calculator,
          mean_offset_calculator, variance_offset_calculator);
  });
  NDArray::add(step, 1, kBlockingStream, step);
  NDArray::MarkUsedBy({grad, param, mean, variance, step, grad_offset_calculator_arr,
                      param_offset_calculator_arr, mean_offset_calculator_arr,
                      variance_offset_calculator_arr}, stream);
}

} // namespace impl
} // namespace hetu
