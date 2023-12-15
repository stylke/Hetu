#include "hetu/core/ndarray.h"
#include "hetu/core/memory_pool.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/cuda/CUDADnn.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/cuda_math.h"

namespace hetu {
namespace impl {

template <typename spec_t>
__forceinline__ __device__ void WarpReduceArgmin(spec_t& val) {
  spec_t tmp_val;
  unsigned int mask = __ballot_sync(0xFFFFFFFF, true);
  for (unsigned int k = (warpSize >> 1); k > 0; k >>= 1) {
    tmp_val = __shfl_down_sync(mask, val, k, warpSize);
    if (tmp_val < val) {
      val = tmp_val;
    }
  }
}

template <>
__forceinline__ __device__ void WarpReduceArgmin(bfloat16& val) {
  bfloat16 tmp_val;
  unsigned int mask = __ballot_sync(0xFFFFFFFF, true);
  #if defined(__CUDACC__) && (__CUDA_ARCH__ >= 800)
  for (unsigned int k = (warpSize >> 1); k > 0; k >>= 1) {
    tmp_val = __shfl_down_sync(mask, val, k, warpSize);
    if (tmp_val < val) {
      val = tmp_val;
    }
  }
  #endif
}

template <typename spec_t>
__forceinline__ __device__ void BlockReduceArgmin(spec_t& val,
                                                  spec_t* shared_value) {
  int tid = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  WarpReduceArgmin(val);

  __syncthreads();
  if (tid == 0) {
    shared_value[wid] = val;
  }

  __syncthreads();
  val = (threadIdx.x < blockDim.x / warpSize) ? shared_value[tid] : SIZE_MAX;

  if (wid == 0)
    WarpReduceArgmin(val);
}

template <typename spec_t>
__global__ void
reduce_min_kernel(const spec_t* input, spec_t* output, int ndim_input,
                  int ndim_rest, int ndim_reduce, 
                  const hetu::cuda::Int64Buffer<HT_MAX_NDIM> strides,
                  const hetu::cuda::Int64Buffer<HT_MAX_NDIM> strides_reduce,
                  const hetu::cuda::Int64Buffer<HT_MAX_NDIM> stride_rest,
                  const hetu::cuda::Int64Buffer<HT_MAX_NDIM> shape_in,
                  const hetu::cuda::Int64Buffer<HT_MAX_NDIM> shape_rest,
                  const hetu::cuda::Int64Buffer<HT_MAX_NDIM> shape_reduce,
                  const hetu::cuda::Int64Buffer<HT_MAX_NDIM> reduce_dims,
                  const hetu::cuda::Int64Buffer<HT_MAX_NDIM> rest_dims,
                  int reduce_num) {
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

  spec_t sum_thread = SIZE_MAX;
  for (size_t i = start_index, ptr; i < end_index; i += blockDim.x) {
    ptr = ptr_fix, tmp = i;
    for (int j = 0; j < ndim_reduce; ++j) {
      k = tmp / strides_reduce[j];
      ptr += k * strides[reduce_dims[j]];
      tmp -= k * strides_reduce[j];
    }
    sum_thread = hetu::cuda::cuda_min(sum_thread, input[ptr]);
  }
  BlockReduceArgmin(sum_thread, shared_sum);
  if (threadIdx.x == 0)
    output[output_ptr] = sum_thread;
}

template <typename spec_t>
__global__ void reduce_min_single_kernel(const spec_t* input, spec_t* output,
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

  spec_t sum_thread = SIZE_MAX;
  for (size_t ptr = start_ptr; ptr < end_ptr; ptr += stride)
    sum_thread = hetu::cuda::cuda_min(sum_thread, input[ptr]);

  BlockReduceArgmin(sum_thread, shared_sum);
  if (threadIdx.x == 0)
    output[output_ptr] = sum_thread;
}

void ReduceMinCuda(const NDArray& in_arr, NDArray& out_arr, const int64_t* axes,
                   int64_t num_ax, const Stream& stream) {
  if (num_ax <= 0)
    return;
  for (int i = 0; i < num_ax; ++i)
    HT_ASSERT(axes[i] >= 0 && axes[i] < in_arr->ndim());
  
  HTShape reduce_axes(axes, axes + num_ax);
  std::sort(reduce_axes.begin(), reduce_axes.end());
  num_ax =
    std::unique(reduce_axes.begin(), reduce_axes.end()) - reduce_axes.begin();

  hetu::cuda::Int64Buffer<HT_MAX_NDIM> reduce_dims;
  hetu::cuda::Int64Buffer<HT_MAX_NDIM> rest_dims;
  hetu::cuda::Int64Buffer<HT_MAX_NDIM> shape_in;
  hetu::cuda::Int64Buffer<HT_MAX_NDIM> shape_reduce;
  hetu::cuda::Int64Buffer<HT_MAX_NDIM> shape_rest;

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
    CUDAStream cuda_stream(stream);
    hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
    HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
      in_arr->dtype(), spec_t, "ReduceMinCuda", [&]() {
        reduce_min_single_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
          in_arr->data_ptr<spec_t>(), out_arr->data_ptr<spec_t>(),
          befor_dim_size, reduce_dim_size, after_dim_size);
      });
  } else {
    hetu::cuda::Int64Buffer<HT_MAX_NDIM> strides;
    hetu::cuda::Int64Buffer<HT_MAX_NDIM> strides_rest;
    hetu::cuda::Int64Buffer<HT_MAX_NDIM> strides_reduce;

    strides[ndim_input - 1] = strides_reduce[ndim_reduce - 1] =
      strides_rest[ndim_rest - 1] = 1;
    for (int i = ndim_input - 2; i >= 0; --i)
      strides[i] = strides[i + 1] * shape_in[i + 1];
    for (int i = ndim_reduce - 2; i >= 0; --i)
      strides_reduce[i] = strides_reduce[i + 1] * shape_reduce[i + 1];
    for (int i = ndim_rest - 2; i >= 0; --i)
      strides_rest[i] = strides_rest[i + 1] * shape_rest[i + 1];

    CUDAStream cuda_stream(stream);
    hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
    int blocks = rest_num;
    int threads = hetu::impl::GetThreadNum(reduce_num);
    HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
      in_arr->dtype(), spec_t, "ReduceMinCuda", [&]() {
        reduce_min_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
          in_arr->data_ptr<spec_t>(), out_arr->data_ptr<spec_t>(), ndim_input,
          ndim_rest, ndim_reduce, strides, strides_reduce, strides_rest,
          shape_in, shape_rest, shape_reduce, reduce_dims, rest_dims,
          reduce_num);
      });
  }
  NDArray::MarkUsedBy({in_arr, out_arr}, stream);
}

} // namespace impl
} // namespace hetu
