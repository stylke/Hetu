#include "hetu/core/ndarray.h"
#include "hetu/core/memory_pool.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"

namespace hetu {
namespace impl {

template <typename spec_t>
__global__ void div_elewise_kernel(const spec_t* inputA, const spec_t* inputB,
                                   size_t size, spec_t* output) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    output[idx] = inputA[idx] / inputB[idx];
}

template <typename spec_t>
__global__ void
div_elewise_broadcast_kernel(const spec_t* inputA, const spec_t* inputB,
                             size_t size, spec_t* output, uint* A_dims,
                             uint* B_dims, size_t A_ndims, size_t B_ndims,
                             uint* out_strides, size_t out_dims) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  size_t A_ind = 0;
  size_t temp = idx;
  for (int i = 0; i < out_dims; ++i) {
    A_ind *= A_dims[i];
    A_ind += (A_dims[i] > 1) * temp / out_strides[i];
    temp %= out_strides[i];
  }
  size_t B_ind = 0;
  temp = idx;
  for (int i = 0; i < out_dims; ++i) {
    B_ind *= B_dims[i];
    B_ind += (B_dims[i] > 1) * temp / out_strides[i];
    temp %= out_strides[i];
  }
  output[idx] = inputA[A_ind] / inputB[B_ind];
}

void DivElewiseCuda(const NDArray& inputA, const NDArray& inputB,
                    NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(inputA);
  HT_ASSERT_SAME_DEVICE(inputA, output);
  HT_ASSERT_SAME_DEVICE(inputB, output);

  size_t size;
  size_t sizeA = inputA->numel();
  size_t sizeB = inputB->numel();
  if (sizeA == sizeB) {
    size = sizeA;
    dim3 blocks, threads;
    threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
    blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
    CUDAStream cuda_stream(stream);
    hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
    HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
      inputA->dtype(), spec_t, "DivElewiseCuda", [&]() {
        div_elewise_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
          inputA->data_ptr<spec_t>(), inputB->data_ptr<spec_t>(), size,
          output->data_ptr<spec_t>());
      });
  } else {
    size_t allocated = output->ndim() * sizeof(uint);
    uint* A_dims = (uint*) malloc(allocated);
    uint* B_dims = (uint*) malloc(allocated);
    uint* out_strides = (uint*) malloc(allocated);
    size_t output_dim = output->ndim();
    size_t output_size = 1;
    size_t diff = output_dim - inputA->ndim();
    for (int i = output_dim - 1; i >= 0; --i) {
      out_strides[i] = output_size;
      output_size *= output->shape(i);
      if (i < diff) {
        A_dims[i] = 1;
      } else {
        A_dims[i] = inputA->shape(i - diff);
      }
    }
    diff = output_dim - inputB->ndim();
    for (int i = output_dim - 1; i >= 0; --i) {
      if (i < diff) {
        B_dims[i] = 1;
      } else {
        B_dims[i] = inputB->shape(i - diff);
      }
    }
    CUDAStream cuda_stream(stream);
    hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
    int dev_id = cuda_stream.device_id();
    size = output->numel();
    dim3 blocks, threads;
    threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
    blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);

    DataPtr gpu_dimsA_ptr = AllocFromMemoryPool(inputA->device(), allocated);
    uint* gpu_dimsA = (uint*) gpu_dimsA_ptr.ptr;
    DataPtr gpu_dimsB_ptr = AllocFromMemoryPool(inputA->device(), allocated);
    uint* gpu_dimsB = (uint*) gpu_dimsB_ptr.ptr;
    DataPtr gpu_output_strides_ptr =
      AllocFromMemoryPool(inputA->device(), allocated);
    uint* gpu_output_strides = (uint*) gpu_output_strides_ptr.ptr;

    CUDA_CALL(cudaMemcpyAsync(gpu_dimsA, A_dims, allocated,
                              cudaMemcpyHostToDevice, cuda_stream));
    CUDA_CALL(cudaMemcpyAsync(gpu_dimsB, B_dims, allocated,
                              cudaMemcpyHostToDevice, cuda_stream));
    CUDA_CALL(cudaMemcpyAsync(gpu_output_strides, out_strides, allocated,
                              cudaMemcpyHostToDevice, cuda_stream));
    HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
      inputA->dtype(), spec_t, "DivElewiseCuda", [&]() {
        div_elewise_broadcast_kernel<spec_t>
          <<<blocks, threads, 0, cuda_stream>>>(
            inputA->data_ptr<spec_t>(), inputB->data_ptr<spec_t>(), size,
            output->data_ptr<spec_t>(), gpu_dimsA, gpu_dimsB,
            (size_t) inputA->ndim(), (size_t) inputB->ndim(),
            gpu_output_strides, output_dim);
      });
    FreeToMemoryPool(gpu_dimsA_ptr);
    FreeToMemoryPool(gpu_dimsB_ptr);
    FreeToMemoryPool(gpu_output_strides_ptr);
    free(A_dims);
    free(B_dims);
    free(out_strides);
  }
}

} // namespace impl
} // namespace hetu
