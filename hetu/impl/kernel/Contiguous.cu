#include "hetu/core/ndarray.h"
#include "hetu/core/memory_pool.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include <chrono>

namespace hetu {
namespace impl {

template <typename spec_t>
__global__ void contiguous_kernel(const spec_t* input, spec_t* output,
                                  const int64_t* stride, const int64_t* new_stride,
                                  const uint ndims, size_t size) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  int64_t i_idx = 0;
  int64_t t = idx;
  for (int i = 0; i < ndims; ++i) {
    int64_t ratio = t / new_stride[i];
    t -= ratio * new_stride[i];
    i_idx += ratio * stride[i];
  }
  output[idx] = input[i_idx];
}

template <typename spec_t>
__global__ void contiguous_gradient_kernel(const spec_t* input, spec_t* output,
                                           const int64_t* stride, const int64_t* new_stride,
                                           const uint ndims, size_t size) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  int64_t i_idx = 0;
  int64_t t = idx;
#pragma unroll
  for (int i = 0; i < ndims; ++i) {
    const int64_t ratio = t / stride[i];
    t -= ratio * stride[i];
    i_idx += ratio * new_stride[i];
  }
  output[i_idx] = input[idx];
}

void ContiguousCuda(const NDArray& input, NDArray& output,
                    const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT(input->numel() == output->numel());

  HTStride stride = input->stride();
  HTStride new_stride = output->stride();
  size_t size = output->numel();
  int ndim = input->ndim();
    
  if (size == 0)
    return;
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  NDArrayMeta meta;
  HTShape alloc = {ndim};
  size_t alloc_size = ndim * sizeof(int64_t);
  meta.set_device(kCPU)
      .set_dtype(DataType::INT64)
      .set_shape(alloc);
  NDArray stride_buf_ptr = NDArray::cuda(NDArray(meta, std::make_shared<NDArrayStorage>(
                            (void*)input->stride().data(), alloc_size, Device(kCPU), 
                            [](DataPtr ptr) { 
                            })), input->device().index(), stream.stream_index());
  int64_t* stride_buf = stride_buf_ptr->data_ptr<int64_t>();
  NDArray new_stride_buf_ptr = NDArray::cuda(NDArray(meta, std::make_shared<NDArrayStorage>(
                            (void*)output->stride().data(), alloc_size, Device(kCPU), 
                            [](DataPtr ptr) { 
                            })), input->device().index(), stream.stream_index());
  int64_t* new_stride_buf = new_stride_buf_ptr->data_ptr<int64_t>();
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "Contiguous", [&]() {
      contiguous_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        input->data_ptr<spec_t>(), output->data_ptr<spec_t>(), stride_buf,
        new_stride_buf, ndim, size);
    });
}

void ContiguousGradientCuda(const NDArray& input, NDArray& output,
                            const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT(input->numel() == output->numel());

  HTStride stride = input->stride();
  HTStride new_stride = output->stride();
  size_t size = output->numel();
  int ndim = input->ndim();
    
  if (size == 0)
    return;
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  NDArrayMeta meta;
  HTShape alloc = {ndim};
  size_t alloc_size = ndim * sizeof(int64_t);
  meta.set_device(kCPU)
      .set_dtype(DataType::INT64)
      .set_shape(alloc);
  NDArray stride_buf_ptr = NDArray::cuda(NDArray(meta, std::make_shared<NDArrayStorage>(
                            (void*)input->stride().data(), alloc_size, Device(kCPU), 
                            [](DataPtr ptr) { 
                            })), input->device().index(), stream.stream_index());
  int64_t* stride_buf = stride_buf_ptr->data_ptr<int64_t>();
  NDArray new_stride_buf_ptr = NDArray::cuda(NDArray(meta, std::make_shared<NDArrayStorage>(
                            (void*)output->stride().data(), alloc_size, Device(kCPU), 
                            [](DataPtr ptr) { 
                            })), input->device().index(), stream.stream_index());
  int64_t* new_stride_buf = new_stride_buf_ptr->data_ptr<int64_t>();
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "ContiguousGradient", [&]() {
      contiguous_gradient_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        input->data_ptr<spec_t>(), output->data_ptr<spec_t>(), stride_buf,
        new_stride_buf, ndim, size);
    });
}

} // namespace impl
} // namespace hetu
