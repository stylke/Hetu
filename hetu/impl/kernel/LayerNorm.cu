#include "hetu/core/ndarray.h"
#include "hetu/core/memory_pool.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/cuda/CUDADnn.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"

namespace hetu {
namespace impl {

template <typename spec_t>
__forceinline__ __device__ spec_t WarpReduceSum(spec_t val) {
  unsigned int mask = __ballot_sync(0xFFFFFFFF, true);
  for (unsigned int k = (warpSize >> 1); k > 0; k >>= 1)
    val += __shfl_down_sync(mask, val, k, warpSize);
  return val;
}

template <typename spec_t>
__forceinline__ __device__ void BlockReduceSum(spec_t& val, spec_t* shared) {
  int tid = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = WarpReduceSum(val);

  __syncthreads();
  if (tid == 0)
    shared[wid] = val;

  __syncthreads();
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[tid] : 0;

  if (wid == 0)
    val = WarpReduceSum(val);
}

template <typename spec_t>
__global__ void layer_norm_kernel(const spec_t* x, const spec_t* scale,
                                  const spec_t* bias, spec_t* y, spec_t* mean,
                                  spec_t* var, const float eps,
                                  const int last_dim) {
  __shared__ spec_t var_share;
  __shared__ spec_t mean_share;
  __shared__ spec_t shared_var[32];
  __shared__ spec_t shared_mean[32];

  int begin = blockIdx.x * last_dim + threadIdx.x;
  int end = (blockIdx.x + 1) * last_dim;

  spec_t mean_thread = 0, var_thread = 0;
  for (int i = begin; i < end; i += blockDim.x) {
    mean_thread += x[i];
    var_thread += (x[i] * x[i]);
  }

  BlockReduceSum(mean_thread, shared_mean);
  BlockReduceSum(var_thread, shared_var);
  if (threadIdx.x == 0) {
    mean[blockIdx.x] = mean_share = mean_thread / last_dim;
    var_share = var_thread / last_dim - mean_share * mean_share;
    if (var_share < 0)
      var_share = 0;
    var[blockIdx.x] = var_share;
  }
  __syncthreads();

  mean_thread = mean_share;
  var_thread = var_share;
  spec_t tmp = 1.0f / sqrtf(var_thread + eps);
  for (int i = begin, j = threadIdx.x; i < end;
       i += blockDim.x, j += blockDim.x)
    y[i] = (x[i] - mean_thread) * tmp * scale[j] + bias[j];
}

void LayerNormCuda(const NDArray& in_arr, const NDArray& ln_scale,
                   const NDArray& ln_bias, NDArray& mean_arr, NDArray& var_arr,
                   NDArray& out_arr, float eps, const Stream& stream) {
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  cudnnHandle_t handle = hetu::impl::GetCudnnHandle(cuda_stream.device_id());

  int ndim = in_arr->ndim();
  int base_dim = 1, last_dim = in_arr->shape(ndim - 1);
  for (int i = 0; i < ndim - 1; ++i)
    base_dim *= in_arr->shape(i);
  // int BlockDim = (last_dim >= 1024 ? 1024: 64);
  dim3 blocks, threads;
  threads.x = (last_dim >= 1024 ? 1024 : 64);
  blocks.x = base_dim;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    in_arr->dtype(), spec_t, "LayerNormCuda", [&]() {
      layer_norm_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        in_arr->data_ptr<spec_t>(), ln_scale->data_ptr<spec_t>(),
        ln_bias->data_ptr<spec_t>(), out_arr->data_ptr<spec_t>(),
        mean_arr->data_ptr<spec_t>(), var_arr->data_ptr<spec_t>(), eps,
        last_dim);
    });
  return;
}

template <typename spec_t>
__global__ void process_kernel1(const spec_t* grads, const spec_t* in_arr,
                                const spec_t* mean_arr, const spec_t* var_arr,
                                const spec_t* ln_scale, spec_t* ws1,
                                spec_t* ws2, spec_t* out_arr, spec_t eps,
                                int last_dim, size_t size) {
  size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
  if (ind >= size)
    return;
  int mo_ind = ind / last_dim;
  spec_t std = sqrtf(var_arr[mo_ind] + eps);
  spec_t x_centered = in_arr[ind] - mean_arr[mo_ind];
  spec_t x_norm = x_centered / std;
  spec_t gscale = grads[ind] * x_norm;
  ws1[ind] = gscale;
  int ln_ind = ind % last_dim;
  if (ln_ind == 0) {
    ws2[mo_ind] = std;
  }
  spec_t dx_norm = grads[ind] * ln_scale[ln_ind];
  spec_t dvar_temp = dx_norm * x_centered;
  out_arr[ind] = dvar_temp;
}

template <typename spec_t>
__global__ void process_kernel2(const spec_t* grads, const spec_t* in_arr,
                                const spec_t* mean_arr, const spec_t* var_arr,
                                const spec_t* ln_scale, spec_t* ws1,
                                spec_t* ws2, spec_t* ws3, spec_t eps,
                                int last_dim, size_t size) {
  size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
  if (ind >= size)
    return;
  int ln_ind = ind % last_dim;
  spec_t dx_norm = grads[ind] * ln_scale[ln_ind];
  int mo_ind = ind / last_dim;
  spec_t dx_mu_1 = dx_norm / ws2[mo_ind];
  spec_t dvar = ws3[mo_ind] * -0.5 / ws2[mo_ind] / (var_arr[mo_ind] + eps);
  spec_t x_centered = in_arr[ind] - mean_arr[mo_ind];
  spec_t dx_mu_2 = dvar * 2 * x_centered / last_dim;
  spec_t dx1 = dx_mu_1 + dx_mu_2;
  ws1[ind] = dx1;
}

template <typename spec_t>
__global__ void process_kernel3(const spec_t* ws1, const spec_t* ws3,
                                spec_t* out_arr, int last_dim, size_t size) {
  size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
  if (ind >= size)
    return;
  int mo_ind = ind / last_dim;
  out_arr[ind] = ws1[ind] - ws3[mo_ind] / last_dim;
}

void LayerNormGradientCuda(const NDArray& out_grads, const NDArray& in_arr,
                           const NDArray& ln_scale, NDArray& grad_arr,
                           NDArray& grad_scale, NDArray& grad_bias,
                           const NDArray& mean_arr, const NDArray& var_arr,
                           float eps, const Stream& stream) {
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  cudnnHandle_t handle = hetu::impl::GetCudnnHandle(cuda_stream.device_id());

  int ori_ndim = out_grads->ndim();
  int last_dim = out_grads->shape(ori_ndim - 1);
  int ndim = std::max(ori_ndim, 4);
  size_t cpu_mem = ndim * sizeof(int);
  int* dimA = (int*) malloc(cpu_mem);
  int* strideA = (int*) malloc(cpu_mem);
  int* dimC = (int*) malloc(cpu_mem);
  int* strideC = (int*) malloc(cpu_mem);

  int temp_strideA = 1;
  int temp_strideC = 1;

  for (int i = ndim - 1; i >= 0; --i) {
    dimA[i] = i < ori_ndim ? (int) out_grads->shape(i) : 1;
    dimC[i] = i == ori_ndim - 1 ? (int) out_grads->shape(i) : 1;
    strideA[i] = temp_strideA;
    strideC[i] = temp_strideC;
    temp_strideA *= dimA[i];
    temp_strideC *= dimC[i];
  }

  int dev_id = cuda_stream.device_id();

  cudnnDataType_t datatype;
  cudnnIndicesType_t indicetype;
  if (out_grads->dtype() == DataType::FLOAT32) {
    datatype = CUDNN_DATA_FLOAT;
    indicetype = CUDNN_32BIT_INDICES;
  } else if (out_grads->dtype() == DataType::FLOAT64) {
    datatype = CUDNN_DATA_DOUBLE;
    indicetype = CUDNN_64BIT_INDICES;
  }

  dim3 blocks, threads;
  threads.x = MIN(temp_strideA, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(temp_strideA, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    in_arr->dtype(), spec_t, "LayerNormGradientCuda", [&]() {
      size_t size = temp_strideA * sizeof(spec_t);
      size_t size1 = temp_strideA * sizeof(spec_t);
      size_t size2 = size1 / last_dim;

      spec_t one = 1.0;
      spec_t zero = 0.0;

      cudnnReduceTensorDescriptor_t rtd;
      CUDNN_CALL(cudnnCreateReduceTensorDescriptor(&rtd));
      CUDNN_CALL(cudnnSetReduceTensorDescriptor(
        rtd, CUDNN_REDUCE_TENSOR_ADD, datatype, CUDNN_PROPAGATE_NAN,
        CUDNN_REDUCE_TENSOR_NO_INDICES, indicetype));

      cudnnTensorDescriptor_t adesc;
      cudnnTensorDescriptor_t cdesc;
      CUDNN_CALL(cudnnCreateTensorDescriptor(&adesc));
      CUDNN_CALL(cudnnCreateTensorDescriptor(&cdesc));
      CUDNN_CALL(
        cudnnSetTensorNdDescriptor(adesc, datatype, ndim, dimA, strideA));
      CUDNN_CALL(
        cudnnSetTensorNdDescriptor(cdesc, datatype, ndim, dimC, strideC));
      CUDNN_CALL(cudnnReduceTensor(
        handle, rtd, NULL, 0, (void*) grad_arr->data_ptr<void>(), size, &one,
        adesc, (const void*) out_grads->data_ptr<void>(), &zero, cdesc,
        (void*) grad_bias->data_ptr<void>()));
      DataPtr ws1_ptr = AllocFromMemoryPool(out_grads->device(), size1);
      spec_t* ws1 = (spec_t*) ws1_ptr.ptr;
      DataPtr ws2_ptr = AllocFromMemoryPool(out_grads->device(), size2);
      spec_t* ws2 = (spec_t*) ws2_ptr.ptr;
      DataPtr ws3_ptr = AllocFromMemoryPool(out_grads->device(), size2);
      spec_t* ws3 = (spec_t*) ws3_ptr.ptr;

      process_kernel1<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        out_grads->data_ptr<spec_t>(), in_arr->data_ptr<spec_t>(),
        mean_arr->data_ptr<spec_t>(), var_arr->data_ptr<spec_t>(),
        ln_scale->data_ptr<spec_t>(), ws1, ws2, grad_arr->data_ptr<spec_t>(),
        eps, last_dim, (size_t) temp_strideA);

      CUDNN_CALL(cudnnReduceTensor(handle, rtd, NULL, 0, (void*) ws1, size,
                                   &one, adesc, (const void*) ws1, &zero, cdesc,
                                   (void*) grad_scale->data_ptr<void>()));
      temp_strideC = 1;

      for (int i = ndim - 1; i >= 0; --i) {
        dimC[i] = i < ori_ndim - 1 ? (int) out_grads->shape(i) : 1;
        strideC[i] = temp_strideC;
        temp_strideC *= dimC[i];
      }
      CUDNN_CALL(
        cudnnSetTensorNdDescriptor(cdesc, datatype, ndim, dimC, strideC));
      CUDNN_CALL(cudnnReduceTensor(
        handle, rtd, NULL, 0, (void*) grad_arr->data_ptr<void>(), size, &one,
        adesc, (const void*) grad_arr->data_ptr<void>(), &zero, cdesc,
        (void*) ws3));

      process_kernel2<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        out_grads->data_ptr<spec_t>(), in_arr->data_ptr<spec_t>(),
        mean_arr->data_ptr<spec_t>(), var_arr->data_ptr<spec_t>(),
        ln_scale->data_ptr<spec_t>(), ws1, ws2, ws3, eps, last_dim,
        (size_t) temp_strideA);

      CUDNN_CALL(cudnnReduceTensor(
        handle, rtd, NULL, 0, (void*) grad_arr->data_ptr<void>(), size, &one,
        adesc, (const void*) ws1, &zero, cdesc, (void*) ws3));

      process_kernel3<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        ws1, ws3, grad_arr->data_ptr<spec_t>(), last_dim, temp_strideA);

      FreeToMemoryPool(ws1_ptr);
      FreeToMemoryPool(ws2_ptr);
      FreeToMemoryPool(ws3_ptr);

      CUDNN_CALL(cudnnDestroyTensorDescriptor(adesc));
      CUDNN_CALL(cudnnDestroyTensorDescriptor(cdesc));
      CUDNN_CALL(cudnnDestroyReduceTensorDescriptor(rtd));
    });

  free(dimA);
  free(dimC);
  free(strideA);
  free(strideC);
}

} // namespace impl
} // namespace hetu
