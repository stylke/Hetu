#include "hetu/core/ndarray.h"
#include "hetu/core/memory_pool.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/cuda/CUDADnn.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/kernel/Binary.cuh"
#include "hetu/impl/utils/cuda_math.h"

namespace hetu {
namespace impl {

template <typename spec_a_t, typename spec_b_t, typename Operator>
extern __global__ void binary_elewise_kernel(const spec_a_t* inputA, const spec_b_t* inputB,
                                             size_t size, Operator op, spec_a_t* output);

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
//   val = (threadIdx.x < blockDim.x / warpSize) ? spec_t(shared[tid]) : spec_t(0);

//   if (wid == 0)
//     val = WarpReduceSum(val);
// }

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

  hetu::cuda::BlockReduceSum(mean_thread, shared_mean);
  hetu::cuda::BlockReduceSum(var_thread, shared_var);
  if (threadIdx.x == 0) {
    mean[blockIdx.x] = mean_share = mean_thread / last_dim;
    var_share = var_thread / last_dim - mean_share * mean_share;
    if (double(var_share) < 0)
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
                   NDArray& out_arr, int64_t reduce_dims, 
                   float eps, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(in_arr);
  HT_ASSERT_SAME_DEVICE(in_arr, ln_scale);
  HT_ASSERT_SAME_DEVICE(in_arr, ln_bias);
  HT_ASSERT_SAME_DEVICE(in_arr, mean_arr); 
  HT_ASSERT_SAME_DEVICE(in_arr, var_arr); 
  HT_ASSERT_SAME_DEVICE(in_arr, out_arr);

  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  cudnnHandle_t handle = hetu::impl::GetCudnnHandle(cuda_stream.device_id());

  int ndim = in_arr->ndim();
  int base_dim = 1, last_dim = 1;
  for (int i = 0; i < ndim - reduce_dims; ++i)
    base_dim *= in_arr->shape(i);
  for (int i = ndim - reduce_dims; i < ndim; ++i)
    last_dim *= in_arr->shape(i);
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
__global__ void calculate_gscale(const spec_t* grads, const spec_t* in_arr,
                                 const spec_t* mean_arr, const spec_t* var_arr,
                                 spec_t* grad_scale, float eps,
                                 int last_dim, size_t size) {
  size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
  if (ind >= size)
    return;
  int mo_ind = ind / last_dim;
  spec_t std = hetu::cuda::cuda_sqrt(var_arr[mo_ind] + eps);
  spec_t x_centered = in_arr[ind] - mean_arr[mo_ind];
  spec_t x_norm = x_centered / std;
  grad_scale[ind] = grads[ind] * x_norm;
}

template <>
__global__ void calculate_gscale<float16>(const float16* grads, const float16* in_arr,
                                          const float16* mean_arr, const float16* var_arr,
                                          float16* grad_scale, float eps,
                                          int last_dim, size_t size) {
  size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
  if (ind >= size)
    return;
  int mo_ind = ind / last_dim;
  float16 std = hetu::cuda::cuda_sqrt(var_arr[mo_ind] + eps);
  float16 x_centered = in_arr[ind] - mean_arr[mo_ind];
  float16 x_norm = x_centered / std;
  grad_scale[ind] = grads[ind] * x_norm;
}

template <typename spec_t>
__global__ void calculate_grad_kernel_layer(const spec_t* out_grads,
                                      const spec_t* in_arr,
                                      const spec_t* mean_arr,
                                      const spec_t* var_arr, 
                                      spec_t* ds, spec_t* db,
                                      spec_t* grad_arr,
                                      size_t lastdim, float eps, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  size_t mo_idx = idx / lastdim;
  // float y = (in_arr[idx] - mean_arr[mo_idx]) / sqrtf(var_arr[mo_idx] + eps);
  spec_t tmp = (db[mo_idx] * mean_arr[mo_idx] - ds[mo_idx]) * (in_arr[idx] - mean_arr[mo_idx]) /
                (var_arr[mo_idx] + eps);
  grad_arr[idx] = (out_grads[idx] + (tmp - db[mo_idx]) / (spec_t)lastdim) / 
    hetu::cuda::cuda_sqrt(var_arr[mo_idx] + eps);
}

template <>
__global__ void calculate_grad_kernel_layer<float16>(const float16* out_grads,
                                      const float16* in_arr,
                                      const float16* mean_arr,
                                      const float16* var_arr, 
                                      float16* ds, float16* db,
                                      float16* grad_arr,
                                      size_t lastdim, float eps, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  size_t mo_idx = idx / lastdim;
  // float y = (in_arr[idx] - mean_arr[mo_idx]) / sqrtf(var_arr[mo_idx] + eps);
  float16 tmp = (db[mo_idx] * mean_arr[mo_idx] - ds[mo_idx]) * (in_arr[idx] - mean_arr[mo_idx]) /
                (var_arr[mo_idx] + eps);
  grad_arr[idx] = (out_grads[idx] + (tmp - db[mo_idx]) / (float16)lastdim) / 
    hetu::cuda::cuda_sqrt(var_arr[mo_idx] + eps);
}

// void LayerNormGradientCuda(const NDArray& out_grads, const NDArray& in_arr,
//                            const NDArray& ln_scale, NDArray& grad_arr,
//                            NDArray& grad_scale, NDArray& grad_bias,
//                            const NDArray& mean_arr, const NDArray& var_arr,
//                            int64_t reduce_dims, float eps, const Stream& stream) {
//   HT_ASSERT_CUDA_DEVICE(out_grads);
//   HT_ASSERT_SAME_DEVICE(out_grads, ln_scale);
//   HT_ASSERT_SAME_DEVICE(out_grads, in_arr);
//   HT_ASSERT_SAME_DEVICE(out_grads, mean_arr); 
//   HT_ASSERT_SAME_DEVICE(out_grads, var_arr); 
//   HT_ASSERT_SAME_DEVICE(out_grads, grad_scale);
//   HT_ASSERT_SAME_DEVICE(out_grads, grad_arr);
//   HT_ASSERT_SAME_DEVICE(out_grads, grad_bias);

//   CUDAStream cuda_stream(stream);
//   hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
//   cudnnHandle_t handle = hetu::impl::GetCudnnHandle(cuda_stream.device_id());

//   int ndim = out_grads->ndim();
// //   HT_ASSERT(ndim == 4);
//   size_t total_elements = 1;

//   cudnnDataType_t datatype;
//   cudnnIndicesType_t indicetype;
//   if (in_arr->dtype() == DataType::FLOAT32) {
//     datatype = CUDNN_DATA_FLOAT;
//     indicetype = CUDNN_32BIT_INDICES;
//   } else if (in_arr->dtype() == DataType::FLOAT64) {
//     datatype = CUDNN_DATA_DOUBLE;
//     indicetype = CUDNN_64BIT_INDICES;
//   } else if (in_arr->dtype() == DataType::FLOAT16) {
//     datatype = CUDNN_DATA_HALF;
//     indicetype = CUDNN_32BIT_INDICES;
//   }
//   #if defined(CUDNN_VERSION) && CUDNN_VERSION >= 8200
//   else if (in_arr->dtype() == DataType::BFLOAT16) {
//     datatype = CUDNN_DATA_BFLOAT16;
//     indicetype = CUDNN_32BIT_INDICES;
//   }
//   #endif
//   else {
//     HT_LOG_INFO << "UNSUPPORTED TYPE:" << in_arr->dtype();
//   }

//   int last_2dim = in_arr->shape(ndim - 1) * in_arr->shape(ndim - 2);
//   size_t cpu_mem = ndim * sizeof(int);
//   int* dimA = (int*) malloc(cpu_mem);
//   int* strideA = (int*) malloc(cpu_mem);
//   int* dimB = (int*) malloc(cpu_mem);
//   int* strideB = (int*) malloc(cpu_mem);
//   int* dimC = (int*) malloc(cpu_mem);
//   int* strideC = (int*) malloc(cpu_mem);

//   int temp_strideA = 1;
//   int temp_strideB = 1;
//   int temp_strideC = 1;

//   for (int i = ndim - 1; i >= 0; --i) {
//     dimA[i] = (int) in_arr->shape(i);
//     dimB[i] = i >= in_arr->ndim() - reduce_dims ? (int) in_arr->shape(i) : 1;
//     dimC[i] = i < in_arr->ndim() - reduce_dims ? (int) in_arr->shape(i) : 1;
//     strideA[i] = temp_strideA;
//     strideB[i] = temp_strideB;
//     strideC[i] = temp_strideC;
//     temp_strideA *= dimA[i];
//     temp_strideB *= dimB[i];
//     temp_strideC *= dimC[i];
//   }

//   for (int i = 0; i < ndim; ++i)
//     total_elements *= out_grads->shape(i);
//   int lastdim = 1;
//   for (size_t i = 0; i < reduce_dims; ++i) {
//     lastdim *= out_grads->shape(ndim - 1 -i);
//   }

//   size_t size = total_elements;
//   if (size == 0)
//     return;
//   dim3 blocks, threads;
//   threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
//   blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
//   HT_DISPATCH_FLOATING_TYPES(
//     in_arr->dtype(), spec_t, "CauculateGradCuda", [&]() {
//       spec_t* ds = NULL;
//       DataPtr ds_ptr = AllocFromMemoryPool(in_arr->device(), temp_strideC * sizeof(spec_t));
//       ds = (spec_t*) ds_ptr.ptr;

//       spec_t* db = NULL;
//       DataPtr db_ptr = AllocFromMemoryPool(in_arr->device(), temp_strideC * sizeof(spec_t));
//       db = (spec_t*) db_ptr.ptr;

//       spec_t* dy_mul_x = NULL;
//       DataPtr dy_mul_x_ptr = AllocFromMemoryPool(in_arr->device(), temp_strideA * sizeof(spec_t));
//       dy_mul_x = (spec_t*) dy_mul_x_ptr.ptr;

//       DataPtr gscale_ptr = AllocFromMemoryPool(out_grads->device(), temp_strideA * sizeof(spec_t));
//       spec_t* gscale = (spec_t*) gscale_ptr.ptr;

//       DataPtr workspace_ptr;
//       workspace_ptr = in_arr->dtype() == DataType::FLOAT16 || in_arr->dtype() == DataType::BFLOAT16 
//                                          ? AllocFromMemoryPool(out_grads->device(), temp_strideA * sizeof(float))
//                                          : AllocFromMemoryPool(out_grads->device(), temp_strideA * sizeof(spec_t));
//       spec_t* workspace = (spec_t*) workspace_ptr.ptr;

//       spec_t one = 1.0f;
//       spec_t zero = 0.0f;

//       float one_f = 1.0f;
//       float zero_f = 0.0f;

//       cudnnReduceTensorDescriptor_t rtd;
//       CUDNN_CALL(cudnnCreateReduceTensorDescriptor(&rtd));
//       CUDNN_CALL(cudnnSetReduceTensorDescriptor(
//         rtd, CUDNN_REDUCE_TENSOR_ADD, in_arr->dtype() == DataType::FLOAT16 || in_arr->dtype() == DataType::BFLOAT16 ? CUDNN_DATA_FLOAT : datatype, CUDNN_PROPAGATE_NAN,
//         CUDNN_REDUCE_TENSOR_NO_INDICES, indicetype));

//       cudnnTensorDescriptor_t adesc;
//       cudnnTensorDescriptor_t bdesc;
//       cudnnTensorDescriptor_t cdesc;
//       CUDNN_CALL(cudnnCreateTensorDescriptor(&adesc));
//       CUDNN_CALL(cudnnCreateTensorDescriptor(&bdesc));
//       CUDNN_CALL(cudnnCreateTensorDescriptor(&cdesc));

//       CUDNN_CALL(
//         cudnnSetTensorNdDescriptor(adesc, datatype, ndim, dimA, strideA));
//       CUDNN_CALL(
//         cudnnSetTensorNdDescriptor(bdesc, datatype, ndim, dimB, strideB));
//       CUDNN_CALL(
//         cudnnSetTensorNdDescriptor(cdesc, datatype, ndim, dimC, strideC));

      
//       if (in_arr->dtype() == DataType::FLOAT16 || in_arr->dtype() == DataType::BFLOAT16) {
//         CUDNN_CALL(cudnnReduceTensor(
//           handle, rtd, NULL, 0, (void*) workspace, temp_strideA * sizeof(float), &one_f,
//           adesc, (const void*) out_grads->data_ptr<void>(), &zero_f, bdesc,
//           (void*) grad_bias->data_ptr<void>()));
//       }
//       else {
//         CUDNN_CALL(cudnnReduceTensor(
//           handle, rtd, NULL, 0, (void*) workspace, temp_strideA * sizeof(spec_t), &one,
//           adesc, (const void*) out_grads->data_ptr<void>(), &zero, bdesc,
//           (void*) grad_bias->data_ptr<void>()));
//       }

//       calculate_gscale<spec_t><<<blocks, threads, 0, cuda_stream>>>(
//         out_grads->data_ptr<spec_t>(), in_arr->data_ptr<spec_t>(),
//         mean_arr->data_ptr<spec_t>(), var_arr->data_ptr<spec_t>(),
//         gscale, eps, lastdim, (size_t) temp_strideA);
      

//       if (in_arr->dtype() == DataType::FLOAT16 || in_arr->dtype() == DataType::BFLOAT16) {  
//         CUDNN_CALL(cudnnReduceTensor(
//           handle, rtd, NULL, 0, (void*) workspace, temp_strideA * sizeof(float), &one_f,
//           adesc, (const void*) gscale, &zero_f, bdesc,
//           (void*) grad_scale->data_ptr<void>())); 
//         CUDNN_CALL(cudnnReduceTensor(
//           handle, rtd, NULL, 0, (void*) workspace, temp_strideA * sizeof(float), &one_f,
//           adesc, (const void*) out_grads->data_ptr<void>(), &zero_f, cdesc,
//           (void*) db));  
//       }   
//       else {
//         CUDNN_CALL(cudnnReduceTensor(
//           handle, rtd, NULL, 0, (void*) workspace, temp_strideA * sizeof(spec_t), &one,
//           adesc, (const void*) gscale, &zero, bdesc,
//           (void*) grad_scale->data_ptr<void>())); 
//         CUDNN_CALL(cudnnReduceTensor(
//           handle, rtd, NULL, 0, (void*) workspace, temp_strideA * sizeof(spec_t), &one,
//           adesc, (const void*) out_grads->data_ptr<void>(), &zero, cdesc,
//           (void*) db));  
//       } 

//       auto op = kmultiplies<spec_t, spec_t>();

//       binary_elewise_kernel<spec_t, spec_t><<<blocks, threads, 0, cuda_stream>>>(
//         out_grads->data_ptr<spec_t>(), in_arr->data_ptr<spec_t>(),
//         size, op, dy_mul_x);
      
//       if (in_arr->dtype() == DataType::FLOAT16 || in_arr->dtype() == DataType::BFLOAT16) {
//         CUDNN_CALL(cudnnReduceTensor(
//           handle, rtd, NULL, 0, (void*) workspace, temp_strideA * sizeof(float), &one_f,
//           adesc, (const void*) dy_mul_x, &zero_f, cdesc,
//           (void*) ds));  
//       }
//       else {
//         CUDNN_CALL(cudnnReduceTensor(
//           handle, rtd, NULL, 0, (void*) workspace, temp_strideA * sizeof(spec_t), &one,
//           adesc, (const void*) dy_mul_x, &zero, cdesc,
//           (void*) ds));  
//       }
        

//       calculate_grad_kernel_layer<spec_t><<<blocks, threads, 0, cuda_stream>>>(
//         out_grads->data_ptr<spec_t>(), in_arr->data_ptr<spec_t>(),
//         mean_arr->data_ptr<spec_t>(), var_arr->data_ptr<spec_t>(),
//         ds, db,
//         grad_arr->data_ptr<spec_t>(), lastdim, eps, size);
//       FreeToMemoryPool(ds_ptr);
//       FreeToMemoryPool(db_ptr);
//       FreeToMemoryPool(dy_mul_x_ptr);
//       FreeToMemoryPool(gscale_ptr);
//       FreeToMemoryPool(workspace_ptr);
//     });
//     free(dimA);
//     free(strideA);
//     free(dimB);
//     free(strideB);
//     free(dimC);
//     free(strideC);
// }

void LayerNormGradientCuda(const NDArray& out_grads, const NDArray& in_arr,
                           const NDArray& ln_scale, NDArray& grad_arr,
                           NDArray& grad_scale, NDArray& grad_bias,
                           const NDArray& mean_arr, const NDArray& var_arr,
                           int64_t reduce_dims, float eps, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(out_grads);
  HT_ASSERT_SAME_DEVICE(out_grads, ln_scale);
  HT_ASSERT_SAME_DEVICE(out_grads, in_arr);
  HT_ASSERT_SAME_DEVICE(out_grads, mean_arr); 
  HT_ASSERT_SAME_DEVICE(out_grads, var_arr); 
  HT_ASSERT_SAME_DEVICE(out_grads, grad_scale);
  HT_ASSERT_SAME_DEVICE(out_grads, grad_arr);
  HT_ASSERT_SAME_DEVICE(out_grads, grad_bias);

  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  cudnnHandle_t handle = hetu::impl::GetCudnnHandle(cuda_stream.device_id());

  int ndim = out_grads->ndim();
//   HT_ASSERT(ndim == 4);
  size_t total_elements = 1;

  int last_2dim = in_arr->shape(ndim - 1) * in_arr->shape(ndim - 2);

  HTAxes reduce_axes_before = {}, reduce_axes_after = {};
  for (int i = 0; i < ndim; ++i) {
    if (i < ndim - reduce_dims)
      reduce_axes_before.emplace_back(i);
    else
      reduce_axes_after.emplace_back(i);
  }

  for (int i = 0; i < ndim; ++i)
    total_elements *= out_grads->shape(i);
  int lastdim = 1;
  for (size_t i = 0; i < reduce_dims; ++i) {
    lastdim *= out_grads->shape(ndim - 1 -i);
  }

  size_t size = total_elements;
  if (size == 0)
    return;
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  HT_DISPATCH_FLOATING_TYPES(
    in_arr->dtype(), spec_t, "CauculateGradCuda", [&]() {
    

      NDArray grad_bias_ = NDArray::sum(out_grads, reduce_axes_before, true, stream.stream_index());

      NDArray gscale_ = NDArray::empty_like(in_arr);

      calculate_gscale<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        out_grads->data_ptr<spec_t>(), in_arr->data_ptr<spec_t>(),
        mean_arr->data_ptr<spec_t>(), var_arr->data_ptr<spec_t>(),
        gscale_->data_ptr<spec_t>(), eps, lastdim, in_arr->numel());
      
      NDArray::sum(gscale_, reduce_axes_before, true, stream.stream_index(), grad_scale);

      NDArray db_ = NDArray::sum(out_grads, reduce_axes_after, true, stream.stream_index());

      NDArray dy_mul_x_ = NDArray::mul(out_grads, in_arr, stream.stream_index());

      NDArray ds_ = NDArray::sum(dy_mul_x_, reduce_axes_after, true, stream.stream_index());

      calculate_grad_kernel_layer<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        out_grads->data_ptr<spec_t>(), in_arr->data_ptr<spec_t>(),
        mean_arr->data_ptr<spec_t>(), var_arr->data_ptr<spec_t>(),
        ds_->data_ptr<spec_t>(), db_->data_ptr<spec_t>(),
        grad_arr->data_ptr<spec_t>(), lastdim, eps, size);
    });
}

} // namespace impl
} // namespace hetu
