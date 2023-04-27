#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/cuda_math.h"

namespace hetu {
namespace impl {

template <typename spec_t>
__global__ void repeat_kernel(const spec_t *input, spec_t *output, size_t size,
                              int64_t *stride_in, int64_t *stride_out,
                              int64_t *dim, int ndim) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  int index = 0;
  size_t ind = idx;
  for (int i = 0; i < ndim; i++) {
    int tmp_index = ind / stride_out[i];
    index += (tmp_index % dim[i]) * stride_in[i];
    ind -= tmp_index * stride_out[i];
  }
  output[idx] = input[index];
}


void RepeatCuda(const NDArray& input, NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);

  size_t size = output->numel();
  int ndim = output->ndim();
  int64_t *stride_tmp = new int64_t[ndim];
  int64_t *shape_tmp = new int64_t[ndim];
  for (int i = 0; i < ndim; i++) {
      if (i < (ndim - input->ndim())) {
          stride_tmp[i] = input->stride()[0];
          shape_tmp[i] = 1;
      } else {
          stride_tmp[i] = input->stride()[i - (ndim - input->ndim())];
          shape_tmp[i] = input->shape(i - (ndim - input->ndim()));
      }
  }
  int64_t *stride_in = NULL;
  int64_t *stride_out = NULL;
  int64_t *dim = NULL;
  size_t buf_size = 3 * ndim * sizeof(int64_t);
  DataPtr stride_in_ptr = AllocFromMemoryPool(input->device(), buf_size);
  stride_in = (int64_t*) stride_in_ptr.ptr;
  DataPtr stride_out_ptr = AllocFromMemoryPool(input->device(), buf_size);
  stride_out = (int64_t*) stride_out_ptr.ptr;
  DataPtr dim_ptr = AllocFromMemoryPool(input->device(), buf_size);
  dim = (int64_t*) dim_ptr.ptr;
  if (size == 0)
    return;
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  CudaMemcpyAsync(stride_in, (void*) stride_tmp, buf_size, cudaMemcpyHostToDevice, cuda_stream);
  CudaMemcpyAsync(stride_out, (void*) output->stride().data(), buf_size, cudaMemcpyHostToDevice, cuda_stream);
  CudaMemcpyAsync(dim, (void*) shape_tmp, buf_size, cudaMemcpyHostToDevice, cuda_stream);
  HT_DISPATCH_FLOATING_TYPES(
    input->dtype(), spec_t, "RepeatCuda", [&]() {
      repeat_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        input->data_ptr<spec_t>(), output->data_ptr<spec_t>(), size, stride_in, stride_out, dim, ndim);
    });
  FreeToMemoryPool(stride_in_ptr);
  FreeToMemoryPool(stride_out_ptr);
  FreeToMemoryPool(dim_ptr);
  free(stride_tmp);
  free(shape_tmp);
}

template <typename spec_t>
extern __global__ void array_zero_set_kernel(spec_t* input, size_t size);

template <typename spec_t>
__global__ void repeat_gradient_kernel(const spec_t *input, spec_t *output, size_t size,
                                       int64_t *stride_in, int64_t *stride_out,
                                       int64_t *dim, int ndim) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  int index = 0;
  size_t ind = idx;
  for (int i = 0; i < ndim; i++) {
    int tmp_index = ind / stride_out[i];
    index += (tmp_index % dim[i]) * stride_in[i];
    ind -= tmp_index * stride_out[i];
  }
  hetu::cuda::AtomicAdd(&output[index], input[idx]);
}

void RepeatGradientCuda(const NDArray& output, NDArray& input, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);

  size_t size = output->numel();
  int ndim = output->ndim();
  int64_t *stride_tmp = new int64_t[ndim];
  int64_t *shape_tmp = new int64_t[ndim];
  for (int i = 0; i < ndim; i++) {
      if (i < (ndim - input->ndim())) {
          stride_tmp[i] = input->stride()[0];
          shape_tmp[i] = 1;
      } else {
          stride_tmp[i] = input->stride()[i - (ndim - input->ndim())];
          shape_tmp[i] = input->shape(i - (ndim - input->ndim()));
      }
  }
  int64_t *stride_in = NULL;
  int64_t *stride_out = NULL;
  int64_t *dim = NULL;
  size_t buf_size = 3 * ndim * sizeof(int64_t);
  DataPtr stride_in_ptr = AllocFromMemoryPool(input->device(), buf_size);
  stride_in = (int64_t*) stride_in_ptr.ptr;
  DataPtr stride_out_ptr = AllocFromMemoryPool(input->device(), buf_size);
  stride_out = (int64_t*) stride_out_ptr.ptr;
  DataPtr dim_ptr = AllocFromMemoryPool(input->device(), buf_size);
  dim = (int64_t*) dim_ptr.ptr;
  if (size == 0)
    return;
  dim3 blocks, threads;
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  threads.x = MIN(input->numel(), HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(input->numel(), HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  HT_DISPATCH_FLOATING_TYPES(
    output->dtype(), spec_t, "ArraySetZeroCuda", [&]() {
      array_zero_set_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        input->data_ptr<spec_t>(), input->numel());
    });
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CudaMemcpyAsync(stride_in, (void*) stride_tmp, buf_size, cudaMemcpyHostToDevice, cuda_stream);
  CudaMemcpyAsync(stride_out, (void*) output->stride().data(), buf_size, cudaMemcpyHostToDevice, cuda_stream);
  CudaMemcpyAsync(dim, (void*) shape_tmp, buf_size, cudaMemcpyHostToDevice, cuda_stream);
  HT_DISPATCH_FLOATING_TYPES(
    input->dtype(), spec_t, "RepeatGradientCuda", [&]() {
      repeat_gradient_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        output->data_ptr<spec_t>(), input->data_ptr<spec_t>(), size, stride_in, stride_out, dim, ndim);
    });
  FreeToMemoryPool(stride_in_ptr);
  FreeToMemoryPool(stride_out_ptr);
  FreeToMemoryPool(dim_ptr);
  free(stride_tmp);
  free(shape_tmp);
}


} // namespace impl
} // namespace hetu
