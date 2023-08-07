#include "hetu/core/ndarray.h"
#include "hetu/core/memory_pool.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/cuda/CUDADnn.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/cuda_math.h"

namespace hetu {
namespace impl {

// template <typename spec_t>
// __forceinline__ __device__ spec_t WarpReduceSum(spec_t val) {
//   unsigned int mask = __ballot_sync(0xFFFFFFFF, true);
//   for (unsigned int k = (warpSize >> 1); k > 0; k >>= 1)
//     val += __shfl_down_sync(mask, val, k, warpSize);
//   return val;
// }

// template <typename spec_t>
// __forceinline__ __device__ void BlockReduceSum(spec_t& val, spec_t* shared) {
//   int tid = threadIdx.x % warpSize;
//   int wid = threadIdx.x / warpSize;

//   val = WarpReduceSum(val);

//   __syncthreads();
//   if (tid == 0)
//     shared[wid] = val;

//   __syncthreads();
//   val = (threadIdx.x < blockDim.x / warpSize) ? shared[tid] : 0;

//   if (wid == 0)
//     val = WarpReduceSum(val);
// }

template <typename spec_t>
__global__ void
reduce_mean_kernel(const spec_t* input, spec_t* output, int ndim_input,
                   int ndim_rest, int ndim_reduce, size_t* strides,
                   size_t* strides_reduce, size_t* stride_rest,
                   size_t* shape_in, size_t* shape_rest, size_t* shape_reduce,
                   int* reduce_dims, int* rest_dims, int reduce_num) {
  __shared__ spec_t shared_sum[32];

  size_t start_index = threadIdx.x;
  size_t end_index = reduce_num;
  size_t output_ptr = blockIdx.x;
  if (start_index >= end_index)
    return;

  size_t ptr_fix = 0, tmp = output_ptr, k;
  for (int i = 0; i < ndim_rest; ++i) {
    k = tmp / stride_rest[i];
    ptr_fix += k * strides[rest_dims[i]];
    tmp -= k * stride_rest[i];
  }

  spec_t sum_thread = 0;
  for (size_t i = start_index, ptr; i < end_index; i += blockDim.x) {
    ptr = ptr_fix, tmp = i;
    for (int j = 0; j < ndim_reduce; ++j) {
      k = tmp / strides_reduce[j];
      ptr += k * strides[reduce_dims[j]];
      tmp -= k * strides_reduce[j];
    }
    sum_thread += input[ptr];
  }

  hetu::cuda::BlockReduceSum(sum_thread, shared_sum);
  if (threadIdx.x == 0)
    output[output_ptr] = sum_thread / reduce_num;
}

template <typename spec_t>
__global__ void reduce_mean_single_kernel(const spec_t* input, spec_t* output,
                                          size_t befor_dim_size,
                                          size_t reduce_dim_size,
                                          size_t after_dim_size) {
  __shared__ spec_t shared_sum[32];

  size_t x = blockIdx.x / after_dim_size;
  size_t y = blockIdx.x % after_dim_size;
  size_t start_ptr, end_ptr, stride;
  if (after_dim_size > 1) {
    stride = after_dim_size * blockDim.x;
    start_ptr =
      x * reduce_dim_size * after_dim_size + y + threadIdx.x * after_dim_size;
    end_ptr = x * reduce_dim_size * after_dim_size + y +
      reduce_dim_size * after_dim_size;
  } else {
    size_t cols_per_thread = (reduce_dim_size + blockDim.x - 1) / blockDim.x;
    size_t block_end_ptr = x * reduce_dim_size * after_dim_size + y +
      reduce_dim_size * after_dim_size;
    start_ptr = x * reduce_dim_size * after_dim_size + y +
      threadIdx.x * cols_per_thread * after_dim_size;
    end_ptr = min(start_ptr + cols_per_thread * after_dim_size, block_end_ptr);
    stride = after_dim_size;
  }
  size_t output_ptr = x * after_dim_size + y;
  if (start_ptr >= end_ptr)
    return;

  spec_t sum_thread = 0;
  for (size_t ptr = start_ptr; ptr < end_ptr; ptr += stride)
    sum_thread += input[ptr];

  hetu::cuda::BlockReduceSum(sum_thread, shared_sum);
  if (threadIdx.x == 0)
    output[output_ptr] = sum_thread / reduce_dim_size;
}

void ReduceMeanCuda(const NDArray& in_arr, NDArray& out_arr,
                    const int64_t* axes, int64_t num_ax, const Stream& stream) {
  if (num_ax <= 0)
    return;
  for (int i = 0; i < num_ax; ++i)
    HT_ASSERT(axes[i] >= 0 && axes[i] < in_arr->ndim());
  int64_t* reduce_axes = (int64_t*) malloc(num_ax * sizeof(int64_t));
  memcpy(reduce_axes, axes, num_ax * sizeof(int64_t));
  std::sort(reduce_axes, reduce_axes + num_ax);
  num_ax = std::unique(reduce_axes, reduce_axes + num_ax) - reduce_axes;

  int* reduce_dims = (int*) malloc(num_ax * sizeof(int));
  int* rest_dims = (int*) malloc((in_arr->ndim() - num_ax) * sizeof(int));

  size_t* shape_in = (size_t*) malloc(in_arr->ndim() * sizeof(size_t));
  size_t* shape_reduce = (size_t*) malloc(num_ax * sizeof(size_t));
  size_t* shape_rest =
    (size_t*) malloc((in_arr->ndim() - num_ax) * sizeof(size_t));

  // merge continuous reduce_dims
  int reduce_num = 1, rest_num = 1;
  int ndim_input = 0, ndim_reduce = 0, ndim_rest = 0;
  for (int i = 0, p = 0; i < in_arr->ndim();) {
    while (p < num_ax && reduce_axes[p] < i)
      ++p;
    if (p < num_ax && reduce_axes[p] == i) {
      int reduce_size = 1;
      for (; p < num_ax && reduce_axes[p] == i; ++i, ++p)
        reduce_size *= in_arr->shape(i);
      reduce_dims[ndim_reduce] = ndim_input;
      shape_reduce[ndim_reduce++] = reduce_size;
      shape_in[ndim_input++] = reduce_size;
      reduce_num *= reduce_size;
    } else {
      rest_dims[ndim_rest] = ndim_input;
      shape_rest[ndim_rest++] = in_arr->shape(i);
      shape_in[ndim_input++] = in_arr->shape(i);
      rest_num *= in_arr->shape(i);
      ++i;
    }
  }

  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());

  if (ndim_reduce == 1) {
    size_t befor_dim_size, reduce_dim_size, after_dim_size;
    befor_dim_size = reduce_dim_size = after_dim_size = 1;
    for (int i = 0; i < ndim_input; ++i) {
      if (i < reduce_dims[0])
        befor_dim_size *= shape_in[i];
      else if (i == reduce_dims[0])
        reduce_dim_size = shape_in[i];
      else
        after_dim_size *= shape_in[i];
    }

    int blocks = befor_dim_size * after_dim_size;
    int threads = hetu::impl::GetThreadNum(reduce_dim_size);
    HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
      in_arr->dtype(), spec_t, "ReduceMeanCuda", [&]() {
        reduce_mean_single_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
          in_arr->data_ptr<spec_t>(), out_arr->data_ptr<spec_t>(),
          befor_dim_size, reduce_dim_size, after_dim_size);
      });
    // CudaStreamSynchronize(cuda_stream);
  } else {
    size_t* strides = (size_t*) malloc(ndim_input * sizeof(size_t));
    size_t* strides_rest = (size_t*) malloc(ndim_rest * sizeof(size_t));
    size_t* strides_reduce = (size_t*) malloc(ndim_reduce * sizeof(size_t));

    strides[ndim_input - 1] = strides_reduce[ndim_reduce - 1] =
      strides_rest[ndim_rest - 1] = 1;
    for (int i = ndim_input - 2; i >= 0; --i)
      strides[i] = strides[i + 1] * shape_in[i + 1];
    for (int i = ndim_reduce - 2; i >= 0; --i)
      strides_reduce[i] = strides_reduce[i + 1] * shape_reduce[i + 1];
    for (int i = ndim_rest - 2; i >= 0; --i)
      strides_rest[i] = strides_rest[i + 1] * shape_rest[i + 1];

    DataPtr reduce_dims_cu_ptr =
      AllocFromMemoryPool(in_arr->device(), ndim_reduce * sizeof(int));
    int* reduce_dims_cu = (int*) reduce_dims_cu_ptr.ptr;
    DataPtr rest_dims_cu_ptr =
      AllocFromMemoryPool(in_arr->device(), ndim_rest * sizeof(int));
    int* rest_dims_cu = (int*) rest_dims_cu_ptr.ptr;
    CUDA_CALL(cudaMemcpyAsync(reduce_dims_cu, reduce_dims,
                              ndim_reduce * sizeof(int), cudaMemcpyHostToDevice,
                              cuda_stream));
    CUDA_CALL(cudaMemcpyAsync(rest_dims_cu, rest_dims, ndim_rest * sizeof(int),
                              cudaMemcpyHostToDevice, cuda_stream));

    DataPtr shape_in_cu_ptr =
      AllocFromMemoryPool(in_arr->device(), ndim_input * sizeof(size_t));
    size_t* shape_in_cu = (size_t*) shape_in_cu_ptr.ptr;
    DataPtr shape_reduce_cu_ptr =
      AllocFromMemoryPool(in_arr->device(), ndim_reduce * sizeof(size_t));
    size_t* shape_reduce_cu = (size_t*) shape_reduce_cu_ptr.ptr;
    DataPtr shape_rest_cu_ptr =
      AllocFromMemoryPool(in_arr->device(), ndim_rest * sizeof(int));
    size_t* shape_rest_cu = (size_t*) shape_rest_cu_ptr.ptr;
    DataPtr strides_cu_ptr =
      AllocFromMemoryPool(in_arr->device(), ndim_input * sizeof(size_t));
    size_t* strides_cu = (size_t*) strides_cu_ptr.ptr;
    DataPtr strides_reduce_cu_ptr =
      AllocFromMemoryPool(in_arr->device(), ndim_reduce * sizeof(size_t));
    size_t* strides_reduce_cu = (size_t*) strides_reduce_cu_ptr.ptr;
    DataPtr strides_rest_cu_ptr =
      AllocFromMemoryPool(in_arr->device(), ndim_rest * sizeof(size_t));
    size_t* strides_rest_cu = (size_t*) strides_rest_cu_ptr.ptr;

    CUDA_CALL(cudaMemcpyAsync(shape_in_cu, shape_in,
                              ndim_input * sizeof(size_t),
                              cudaMemcpyHostToDevice, cuda_stream));
    CUDA_CALL(cudaMemcpyAsync(shape_rest_cu, shape_rest,
                              ndim_rest * sizeof(size_t),
                              cudaMemcpyHostToDevice, cuda_stream));
    CUDA_CALL(cudaMemcpyAsync(shape_reduce_cu, shape_reduce,
                              ndim_reduce * sizeof(size_t),
                              cudaMemcpyHostToDevice, cuda_stream));
    CUDA_CALL(cudaMemcpyAsync(strides_cu, strides, ndim_input * sizeof(size_t),
                              cudaMemcpyHostToDevice, cuda_stream));
    CUDA_CALL(cudaMemcpyAsync(strides_reduce_cu, strides_reduce,
                              ndim_reduce * sizeof(size_t),
                              cudaMemcpyHostToDevice, cuda_stream));
    CUDA_CALL(cudaMemcpyAsync(strides_rest_cu, strides_rest,
                              ndim_rest * sizeof(size_t),
                              cudaMemcpyHostToDevice, cuda_stream));

    int blocks = rest_num;
    int threads = hetu::impl::GetThreadNum(reduce_num);
    HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
      in_arr->dtype(), spec_t, "ReduceMeanCuda", [&]() {
        reduce_mean_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
          in_arr->data_ptr<spec_t>(), out_arr->data_ptr<spec_t>(), ndim_input,
          ndim_rest, ndim_reduce, strides_cu, strides_reduce_cu,
          strides_rest_cu, shape_in_cu, shape_rest_cu, shape_reduce_cu,
          reduce_dims_cu, rest_dims_cu, reduce_num);
      });
    FreeToMemoryPool(rest_dims_cu_ptr);
    FreeToMemoryPool(reduce_dims_cu_ptr);
    FreeToMemoryPool(shape_in_cu_ptr);
    FreeToMemoryPool(shape_rest_cu_ptr);
    FreeToMemoryPool(shape_reduce_cu_ptr);
    FreeToMemoryPool(strides_cu_ptr);
    FreeToMemoryPool(strides_rest_cu_ptr);
    FreeToMemoryPool(strides_reduce_cu_ptr);
    free(strides);
    free(strides_rest);
    free(strides_reduce);
    // CudaStreamSynchronize(cuda_stream);
  }
  free(rest_dims);
  free(reduce_dims);
  free(shape_in);
  free(shape_rest);
  free(shape_reduce);
  free(reduce_axes);
  // CudaStreamSynchronize(cuda_stream);
  return;
}

} // namespace impl
} // namespace hetu
