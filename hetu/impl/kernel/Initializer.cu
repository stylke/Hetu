#include "hetu/core/ndarray.h"
#include "hetu/impl/cuda/CUDARand.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/random/CUDARandomState.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"

namespace hetu {
namespace impl {

template <typename spec_t>
__global__ void init_normal_kernel(spec_t* arr, size_t size, spec_t mean,
                                   spec_t stddev, CUDARandomState rand_state) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  curandStatePhilox4_32_10_t state;
  curand_init(rand_state.seed, idx, rand_state.offset, &state);
  arr[idx] = curand_normal(&state) * stddev + mean;
}

template <typename spec_t>
__global__ void init_uniform_kernel(spec_t* arr, size_t size, spec_t lb,
                                    spec_t ub, CUDARandomState rand_state) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  curandStatePhilox4_32_10_t state;
  curand_init(rand_state.seed, idx, rand_state.offset, &state);
  arr[idx] = curand_uniform(&state) * (ub - lb) + lb;
}

template <typename spec_t>
__global__ void init_truncated_normal_kernel(spec_t* arr, size_t size,
                                             spec_t mean, spec_t stddev,
                                             spec_t lb, spec_t ub,
                                             CUDARandomState rand_state) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  curandStatePhilox4_32_10_t state;
  curand_init(rand_state.seed, idx, rand_state.offset, &state);
  do {
    arr[idx] = curand_normal(&state) * stddev + mean;
  } while (arr[idx] < lb || arr[idx] > ub);
}

void NormalInitsCuda(NDArray& data, double mean, double stddev, uint64_t seed,
                     const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(data);
  size_t size = data->numel();
  if (size == 0)
    return;
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    data->dtype(), spec_t, "NormalInitsCuda", [&]() {
      init_normal_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        data->data_ptr<spec_t>(), size, static_cast<spec_t>(mean),
        static_cast<spec_t>(stddev),
        GetCUDARandomState(cuda_stream.device_id(), seed, 4));
    });
  NDArray::MarkUsedBy({data}, stream);
}

void UniformInitsCuda(NDArray& data, double lb, double ub, uint64_t seed,
                      const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(data);
  HT_ASSERT(lb < ub) << "Invalid range for uniform random init: "
                     << "[" << lb << ", " << ub << ").";
  size_t size = data->numel();
  if (size == 0)
    return;
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    data->dtype(), spec_t, "UniformInitCuda", [&]() {
      init_uniform_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        data->data_ptr<spec_t>(), size, static_cast<spec_t>(lb),
        static_cast<spec_t>(ub),
        GetCUDARandomState(cuda_stream.device_id(), seed, 4));
    });
  NDArray::MarkUsedBy({data}, stream);
}

void TruncatedNormalInitsCuda(NDArray& data, double mean, double stddev,
                              double lb, double ub, uint64_t seed,
                              const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(data);
  size_t size = data->numel();
  if (size == 0)
    return;
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    data->dtype(), spec_t, "TruncatedNormalInitsCuda", [&]() {
      init_truncated_normal_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        data->data_ptr<spec_t>(), size, static_cast<spec_t>(mean),
        static_cast<spec_t>(stddev), static_cast<spec_t>(lb),
        static_cast<spec_t>(ub),
        GetCUDARandomState(cuda_stream.device_id(), seed, 32));
    });
  NDArray::MarkUsedBy({data}, stream);
}

} // namespace impl
} // namespace hetu
