#include "hetu/core/ndarray.h"
#include "hetu/core/memory_pool.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"

namespace hetu {
namespace impl {

template <typename spec_t>
__global__ void broadcast_shape_kernel(const spec_t* input, spec_t* output,
                                       uint* out_strides, uint* in_dims,
                                       size_t ndims, size_t size) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  size_t i_ind = 0;
  size_t temp = idx;
  for (int i = 0; i < ndims; ++i) {
    i_ind *= in_dims[i];
    i_ind += (in_dims[i] > 1) * temp / out_strides[i];
    temp %= out_strides[i];
  }
  output[idx] = input[i_ind];
}

void BroadcastShapeCuda(const NDArray& input, NDArray& output,
                        const HTShape& add_axes, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);

  size_t size = output->numel();
  size_t input_size = input->numel();

  int input_dim = input->ndim();
  int output_dim = output->ndim();
  size_t allocated = output_dim * sizeof(uint);
  uint* out_strides = (uint*) malloc(allocated);
  uint* in_dims = (uint*) malloc(allocated);

  size_t output_size = 1;
  size_t diff = output_dim - input_dim;

  if (add_axes.empty()) {
    for (int i = output_dim - 1; i >= 0; --i) {
      out_strides[i] = output_size;
      output_size *= output->shape(i);
      if (i < diff) {
        in_dims[i] = 1;
      } else {
        in_dims[i] = input->shape(i - diff);
      }
    }
  } else {
    for (int i = output_dim - 1; i >= 0; --i) {
      out_strides[i] = output_size;
      output_size *= output->shape(i);
      in_dims[i] = 0;
    }
    for (int i = 0; i < diff; ++i) {
      in_dims[add_axes[i]] = 1;
    }
    int o_ind = 0;
    for (int i = 0; i < input->ndim(); ++i) {
      while (in_dims[o_ind++] == 1)
        ;
      in_dims[o_ind - 1] = input->shape(i);
    }
  }

  DataPtr gpu_strides_ptr = AllocFromMemoryPool(input->device(), allocated);
  uint* gpu_strides = (uint*) gpu_strides_ptr.ptr;
  DataPtr gpu_dims_ptr = AllocFromMemoryPool(input->device(), allocated);
  uint* gpu_dims = (uint*) gpu_dims_ptr.ptr;

  if (size == 0 || input_size == 0)
    return;
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());

  CudaMemcpyAsync(gpu_strides, out_strides, allocated, cudaMemcpyHostToDevice,
                  cuda_stream);
  CudaMemcpyAsync(gpu_dims, in_dims, allocated, cudaMemcpyHostToDevice,
                  cuda_stream);

  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "BroadcastShapeCuda", [&]() {
      broadcast_shape_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        input->data_ptr<spec_t>(), output->data_ptr<spec_t>(), gpu_strides,
        gpu_dims, output_dim, size);
    });

  FreeToMemoryPool(gpu_strides_ptr);
  FreeToMemoryPool(gpu_dims_ptr);
  free(out_strides);
  free(in_dims);
}

template <typename spec_t>
__global__ void broadcast_shape_mul_kernel(const spec_t* input,
                                           spec_t const_value, spec_t* output,
                                           uint* out_strides, uint* in_dims,
                                           size_t ndims, size_t size) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  size_t i_ind = 0;
  size_t temp = idx;
  for (int i = 0; i < ndims; ++i) {
    i_ind *= in_dims[i];
    i_ind += (in_dims[i] > 1) * temp / out_strides[i];
    temp %= out_strides[i];
  }
  output[idx] = input[i_ind] * const_value;
}

void BroadcastShapeMulCuda(const NDArray& input, double const_value,
                           NDArray& output, const HTShape& add_axes,
                           const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);

  size_t size = output->numel();
  size_t input_size = input->numel();

  int input_dim = input->ndim();
  int output_dim = output->ndim();
  size_t allocated = output_dim * sizeof(uint);
  uint* out_strides = (uint*) malloc(allocated);
  uint* in_dims = (uint*) malloc(allocated);

  size_t output_size = 1;
  size_t diff = output_dim - input_dim;

  if (add_axes.empty()) {
    for (int i = output_dim - 1; i >= 0; --i) {
      out_strides[i] = output_size;
      output_size *= output->shape(i);
      if (i < diff) {
        in_dims[i] = 1;
      } else {
        in_dims[i] = input->shape(i - diff);
      }
    }
  } else {
    for (int i = output_dim - 1; i >= 0; --i) {
      out_strides[i] = output_size;
      output_size *= output->shape(i);
      in_dims[i] = 0;
    }
    for (int i = 0; i < diff; ++i) {
      in_dims[add_axes[i]] = 1;
    }
    int o_ind = 0;
    for (int i = 0; i < input->ndim(); ++i) {
      while (in_dims[o_ind++] == 1)
        ;
      in_dims[o_ind - 1] = input->shape(i);
    }
  }

  DataPtr gpu_strides_ptr = AllocFromMemoryPool(input->device(), allocated);
  uint* gpu_strides = (uint*) gpu_strides_ptr.ptr;
  DataPtr gpu_dims_ptr = AllocFromMemoryPool(input->device(), allocated);
  uint* gpu_dims = (uint*) gpu_dims_ptr.ptr;

  if (size == 0 || input_size == 0)
    return;
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());

  CudaMemcpyAsync(gpu_strides, out_strides, allocated, cudaMemcpyHostToDevice,
                  cuda_stream);
  CudaMemcpyAsync(gpu_dims, in_dims, allocated, cudaMemcpyHostToDevice,
                  cuda_stream);

  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "BroadcastShapeMulCuda", [&]() {
      broadcast_shape_mul_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        input->data_ptr<spec_t>(), static_cast<spec_t>(const_value),
        output->data_ptr<spec_t>(), gpu_strides, gpu_dims, output_dim, size);
    });

  FreeToMemoryPool(gpu_strides_ptr);
  FreeToMemoryPool(gpu_dims_ptr);
  free(out_strides);
  free(in_dims);
}

} // namespace impl
} // namespace hetu
