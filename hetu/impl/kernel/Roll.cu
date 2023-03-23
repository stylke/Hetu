#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/cuda_math.h"

namespace hetu {
namespace impl {

template <typename spec_t>
__global__ void roll_kernel(const spec_t *input, spec_t *output, size_t size, int rank,
                            uint *shifts, uint *strides, uint *sizes) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  int output_idx = idx;
  int new_dim_idx = 0;

#pragma unroll
  for (int i = 0; i < rank; i++) {
    new_dim_idx = (idx / strides[i]) % sizes[i] + shifts[i];
    if (new_dim_idx >= sizes[i])
      output_idx += (shifts[i] - sizes[i]) * strides[i];
    else
      output_idx += shifts[i] * strides[i];
  }
  output[output_idx] = input[idx];
}


void RollCuda(const NDArray& input, const HTShape& shift, const HTAxes& axis,
              NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);

  size_t len = input->numel();
  int64_t nums = shift.size();
  int64_t n_dims = input->ndim();

  int *stride_dim = new int[n_dims];
  stride_dim[n_dims - 1] = 1;
  for (int i = 0; i < n_dims; i++) {
    if (i > 0)
      stride_dim[n_dims - i - 1] =
        input->shape(n_dims - i) * stride_dim[n_dims - i];
  }

  int *strides = new int[nums];
  int *sizes = new int[nums];
  int *shifts = new int[nums];

  if (axis.size() == 0) {
    strides[0] = 1;
    sizes[0] = len;
    shifts[0] = (shift[0] % len + len) % len;
  } else {
    for (int i = 0; i < nums; i++) {
      int dim = axis[i] >= 0 ? axis[i] : axis[i] + n_dims;
      int size = input->shape(dim);
      if (size != 0) {
        strides[i] = stride_dim[dim];
        sizes[i] = size;
        shifts[i] = (shift[i] % size + size) % size;
      }
    }
  }

  uint *shifts_buf = NULL;
  uint *strides_buf = NULL;
  uint *sizes_buf = NULL;
  size_t buf_size = nums * sizeof(uint);
  DataPtr shifts_buf_ptr = AllocFromMemoryPool(input->device(), buf_size);
  shifts_buf = (uint*) shifts_buf_ptr.ptr;
  DataPtr strides_buf_ptr = AllocFromMemoryPool(input->device(), buf_size);
  strides_buf = (uint*) strides_buf_ptr.ptr;
  DataPtr sizes_buf_ptr = AllocFromMemoryPool(input->device(), buf_size);
  sizes_buf = (uint*) sizes_buf_ptr.ptr;

  dim3 blocks, threads;
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  threads.x = MIN(len, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(len, HT_DEFAULT_NUM_THREADS_PER_BLOCK);

  CudaMemcpyAsync(shifts_buf, (void *)shifts, buf_size,
                  cudaMemcpyHostToDevice, cuda_stream);
  CudaMemcpyAsync(strides_buf, (void *)strides, buf_size,
                  cudaMemcpyHostToDevice, cuda_stream);
  CudaMemcpyAsync(sizes_buf, (void *)sizes, buf_size,
                  cudaMemcpyHostToDevice, cuda_stream);

  HT_DISPATCH_FLOATING_TYPES(
    input->dtype(), spec_t, "RollCuda", [&]() {
      roll_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        input->data_ptr<spec_t>(), output->data_ptr<spec_t>(), 
        len, nums, shifts_buf, strides_buf, sizes_buf);
    });
  FreeToMemoryPool(shifts_buf_ptr);
  FreeToMemoryPool(strides_buf_ptr);
  FreeToMemoryPool(sizes_buf_ptr);
  free(shifts);
  free(strides);
  free(sizes);
}

} // namespace impl
} // namespace hetu
