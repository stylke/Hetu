#include "hetu/core/ndarray.h"
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

template <typename spec_t>
__global__ void minus_mean_n_square_kernel1(const spec_t* in_arr,
                                            const spec_t* mean, spec_t* out_arr,
                                            int last_2dim, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  spec_t temp = in_arr[idx] - mean[idx / last_2dim];
  out_arr[idx] = temp * temp;
}

template <typename spec_t>
__global__ void std_normal_transform(const spec_t* in_arr,
                                     const spec_t* mean_arr,
                                     const spec_t* var_arr, spec_t* out_arr,
                                     int last_2dim, float eps, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  size_t mo_idx = idx / last_2dim;
  out_arr[idx] =
    (in_arr[idx] - mean_arr[mo_idx]) / hetu::cuda::cuda_sqrt(var_arr[mo_idx] + eps);
}

template <typename spec_t>
__global__ void instance_norm_kernel(const spec_t* x,  spec_t* y, spec_t* mean,
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
    y[i] = (x[i] - mean_thread) * tmp;
}

void InstanceNormCuda(const NDArray& in_arr, NDArray& mean_arr,
                      NDArray& var_arr, NDArray& out_arr, float eps,
                      const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(in_arr);
  HT_ASSERT_SAME_DEVICE(in_arr, mean_arr); 
  HT_ASSERT_SAME_DEVICE(in_arr, var_arr); 
  HT_ASSERT_SAME_DEVICE(in_arr, out_arr);   

  int ndim = in_arr->ndim();
  HT_ASSERT(ndim == 4);
  int last_2dim = in_arr->shape(ndim - 1) * in_arr->shape(ndim - 2);
  int base_dim = in_arr->shape(0) * in_arr->shape(1);

  auto device_id = in_arr->device().index();
  hetu::cuda::CUDADeviceGuard guard(device_id);
  CUDAStream cuda_stream(stream);
  dim3 blocks, threads;
  threads.x = (last_2dim >= 1024 ? 1024 : 64);
  blocks.x = base_dim;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    in_arr->dtype(), spec_t, "InstanceNormCuda", [&]() {
      instance_norm_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        in_arr->data_ptr<spec_t>(), out_arr->data_ptr<spec_t>(),
        mean_arr->data_ptr<spec_t>(), var_arr->data_ptr<spec_t>(), 
        eps, last_2dim);
    });
  NDArray::MarkUsedBy({in_arr, mean_arr, var_arr, out_arr}, stream);
}

template <typename spec_t>
__global__ void calculate_grad_kernel(const spec_t* out_grads,
                                      const spec_t* in_arr,
                                      const spec_t* mean_arr,
                                      const spec_t* var_arr, 
                                      spec_t* ds, spec_t* dbias,
                                      spec_t* grad_arr,
                                      size_t last2dim, float eps, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  size_t mo_idx = idx / last2dim;
  spec_t y = (in_arr[idx] - mean_arr[mo_idx]) / sqrtf(var_arr[mo_idx] + eps);
  spec_t tmp = (dbias[mo_idx] * mean_arr[mo_idx] - ds[mo_idx]) * (in_arr[idx] - mean_arr[mo_idx]) /
                (var_arr[mo_idx] + eps);
  grad_arr[idx] = out_grads[idx] / hetu::cuda::cuda_sqrt(var_arr[mo_idx] + eps) +
    ((tmp - dbias[mo_idx]) / (spec_t)last2dim) / 
    hetu::cuda::cuda_sqrt(var_arr[mo_idx] + eps);
}

template <>
__global__ void calculate_grad_kernel<float16>(const float16* out_grads,
                                      const float16* in_arr,
                                      const float16* mean_arr,
                                      const float16* var_arr, 
                                      float16* ds, float16* dbias,
                                      float16* grad_arr,
                                      size_t last2dim, float eps, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  size_t mo_idx = idx / last2dim;
  float16 y = (in_arr[idx] - mean_arr[mo_idx]) / sqrtf(var_arr[mo_idx] + eps);
  float16 tmp = (dbias[mo_idx] * mean_arr[mo_idx] - ds[mo_idx]) * (in_arr[idx] - mean_arr[mo_idx]) /
                (var_arr[mo_idx] + eps);
  grad_arr[idx] = out_grads[idx] / hetu::cuda::cuda_sqrt(var_arr[mo_idx] + eps) +
    ((tmp - dbias[mo_idx]) / (float16)last2dim) / 
    hetu::cuda::cuda_sqrt(var_arr[mo_idx] + eps);
}

template <>
__global__ void calculate_grad_kernel<bfloat16>(const bfloat16* out_grads,
                                      const bfloat16* in_arr,
                                      const bfloat16* mean_arr,
                                      const bfloat16* var_arr, 
                                      bfloat16* ds, bfloat16* dbias,
                                      bfloat16* grad_arr,
                                      size_t last2dim, float eps, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  size_t mo_idx = idx / last2dim;
  float16 y = (in_arr[idx] - mean_arr[mo_idx]) / sqrtf(var_arr[mo_idx] + eps);
  float16 tmp = (dbias[mo_idx] * mean_arr[mo_idx] - ds[mo_idx]) * (in_arr[idx] - mean_arr[mo_idx]) /
                (var_arr[mo_idx] + eps);
  grad_arr[idx] = out_grads[idx] / hetu::cuda::cuda_sqrt(var_arr[mo_idx] + eps) +
    ((tmp - dbias[mo_idx]) / (float16)last2dim) / 
    hetu::cuda::cuda_sqrt(var_arr[mo_idx] + eps);
}

// void InstanceNormGradientCuda(const NDArray& out_grads, const NDArray& in_arr,
//                               NDArray& grad_arr, const NDArray& mean_arr,
//                               const NDArray& var_arr, float eps,
//                               const Stream& stream) {
//   HT_ASSERT_CUDA_DEVICE(out_grads);
//   HT_ASSERT_SAME_DEVICE(out_grads, in_arr); 
//   HT_ASSERT_SAME_DEVICE(out_grads, grad_arr); 
//   HT_ASSERT_SAME_DEVICE(out_grads, mean_arr);   
//   HT_ASSERT_SAME_DEVICE(out_grads, var_arr); 

//   CUDAStream cuda_stream(stream);
//   hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
//   cudnnHandle_t handle = hetu::impl::GetCudnnHandle(cuda_stream.device_id());

//   int ndim = out_grads->ndim();
//   HT_ASSERT(ndim == 4);
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

//   HT_ASSERT(ndim == 4);
//   int last_2dim = in_arr->shape(ndim - 1) * in_arr->shape(ndim - 2);
//   size_t cpu_mem = ndim * sizeof(int);
//   int* dimA = (int*) malloc(cpu_mem);
//   int* strideA = (int*) malloc(cpu_mem);
//   int* dimC = (int*) malloc(cpu_mem);
//   int* strideC = (int*) malloc(cpu_mem);

//   int temp_strideA = 1;
//   int temp_strideC = 1;

//   for (int i = ndim - 1; i >= 0; --i) {
//     dimA[i] = (int) in_arr->shape(i);
//     dimC[i] = i < in_arr->ndim() - 2 ? (int) in_arr->shape(i) : 1;
//     strideA[i] = temp_strideA;
//     strideC[i] = temp_strideC;
//     temp_strideA *= dimA[i];
//     temp_strideC *= dimC[i];
//   }

//   for (int i = 0; i < ndim; ++i)
//     total_elements *= out_grads->shape(i);
//   int last2dim = out_grads->shape(ndim - 1) * out_grads->shape(ndim - 2);

//   size_t size = total_elements;
//   if (size == 0)
//     return;
//   dim3 blocks, threads;
//   threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
//   blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
//   HT_DISPATCH_FLOATING_TYPES(
//     in_arr->dtype(), spec_t, "CauculateGradCuda", [&]() {
//       #if defined(CUDNN_VERSION) && CUDNN_VERSION < 8200
//       if (in_arr->dtype() == DataType::BFLOAT16)
//         return;
//       #endif
//       spec_t* dscale = NULL;
//       DataPtr dscale_ptr = AllocFromMemoryPool(in_arr->device(), temp_strideC * sizeof(spec_t));
//       dscale = (spec_t*) dscale_ptr.ptr;

//       spec_t* dbias = NULL;
//       DataPtr dbias_ptr = AllocFromMemoryPool(in_arr->device(), temp_strideC * sizeof(spec_t));
//       dbias = (spec_t*) dbias_ptr.ptr;

//       spec_t* dy_mul_x = NULL;
//       DataPtr dy_mul_x_ptr = AllocFromMemoryPool(in_arr->device(), temp_strideA * sizeof(spec_t));
//       dy_mul_x = (spec_t*) dy_mul_x_ptr.ptr;

//       void* workspace = NULL;
//       DataPtr workspace_ptr = in_arr->dtype() == DataType::FLOAT16 ? AllocFromMemoryPool(in_arr->device(), temp_strideA * sizeof(float))
//                                                                    : AllocFromMemoryPool(in_arr->device(), temp_strideA * sizeof(spec_t));
//       workspace = (void*) workspace_ptr.ptr;

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
//       cudnnTensorDescriptor_t cdesc;
//       CUDNN_CALL(cudnnCreateTensorDescriptor(&adesc));
//       CUDNN_CALL(cudnnCreateTensorDescriptor(&cdesc));

//       CUDNN_CALL(
//         cudnnSetTensorNdDescriptor(adesc, datatype, ndim, dimA, strideA));
//       CUDNN_CALL(
//         cudnnSetTensorNdDescriptor(cdesc, datatype, ndim, dimC, strideC));

//       if (in_arr->dtype() == DataType::FLOAT16) {
//         CUDNN_CALL(cudnnReduceTensor(
//           handle, rtd, NULL, 0, (void*) workspace, temp_strideA * sizeof(float), &one_f,
//           adesc, (const void*) out_grads->data_ptr<void>(), &zero_f, cdesc,
//           (void*) dbias));     
//       }
//       else {
//         CUDNN_CALL(cudnnReduceTensor(
//           handle, rtd, NULL, 0, (void*) workspace, temp_strideA * sizeof(spec_t), &one,
//           adesc, (const void*) out_grads->data_ptr<void>(), &zero, cdesc,
//           (void*) dbias));      
//       }

//       auto op = kmultiplies<spec_t, spec_t>();

//       binary_elewise_kernel<spec_t, spec_t><<<blocks, threads, 0, cuda_stream>>>(
//         out_grads->data_ptr<spec_t>(), in_arr->data_ptr<spec_t>(),
//         size, op, dy_mul_x);
//       if (in_arr->dtype() == DataType::FLOAT16 || in_arr->dtype() == DataType::BFLOAT16) {  
//         CUDNN_CALL(cudnnReduceTensor(
//           handle, rtd, NULL, 0, (void*) workspace, temp_strideA * sizeof(float), &one_f,
//           adesc, (const void*) dy_mul_x, &zero_f, cdesc,
//           (void*) dscale));  
//       }
//       else {
//         CUDNN_CALL(cudnnReduceTensor(
//           handle, rtd, NULL, 0, (void*) workspace, temp_strideA * sizeof(spec_t), &one,
//           adesc, (const void*) dy_mul_x, &zero, cdesc,
//           (void*) dscale));  
//       }
//       calculate_grad_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
//         out_grads->data_ptr<spec_t>(), in_arr->data_ptr<spec_t>(),
//         mean_arr->data_ptr<spec_t>(), var_arr->data_ptr<spec_t>(),
//         dscale, dbias,
//         grad_arr->data_ptr<spec_t>(), last2dim, eps, size);
//       FreeToMemoryPool(dscale_ptr);
//       FreeToMemoryPool(dbias_ptr);
//       FreeToMemoryPool(dy_mul_x_ptr);
//       FreeToMemoryPool(workspace_ptr);
//     });
//   free(dimA);
//   free(strideA);
//   free(dimC);
//   free(strideC);
// }

void InstanceNormGradientCuda(const NDArray& out_grads, const NDArray& in_arr,
                              NDArray& grad_arr, const NDArray& mean_arr,
                              const NDArray& var_arr, float eps,
                              const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(out_grads);
  HT_ASSERT_SAME_DEVICE(out_grads, in_arr); 
  HT_ASSERT_SAME_DEVICE(out_grads, grad_arr); 
  HT_ASSERT_SAME_DEVICE(out_grads, mean_arr);   
  HT_ASSERT_SAME_DEVICE(out_grads, var_arr); 

  int ndim = out_grads->ndim();
  HT_ASSERT(ndim == 4);
  size_t total_elements = 1;


  HT_ASSERT(ndim == 4);
  int last_2dim = in_arr->shape(ndim - 1) * in_arr->shape(ndim - 2);

  for (int i = 0; i < ndim; ++i)
    total_elements *= out_grads->shape(i);
  int last2dim = out_grads->shape(ndim - 1) * out_grads->shape(ndim - 2);

  size_t size = total_elements;
  if (size == 0)
    return;
  
  auto device_id = out_grads->device().index();
  hetu::cuda::CUDADeviceGuard guard(device_id);
  CUDAStream cuda_stream(stream);
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  NDArray dbias_arr = NDArray::sum(out_grads, {2, 3}, true, stream.stream_index());
  NDArray dy_mul_x_arr = NDArray::mul(out_grads, in_arr, stream.stream_index());
  NDArray dscale_arr = NDArray::sum(dy_mul_x_arr, {2, 3}, true, stream.stream_index());
  HT_DISPATCH_FLOATING_TYPES(
    in_arr->dtype(), spec_t, "CauculateGradCuda", [&]() {
      calculate_grad_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        out_grads->data_ptr<spec_t>(), in_arr->data_ptr<spec_t>(),
        mean_arr->data_ptr<spec_t>(), var_arr->data_ptr<spec_t>(),
        dscale_arr->data_ptr<spec_t>(), dbias_arr->data_ptr<spec_t>(),
        grad_arr->data_ptr<spec_t>(), last2dim, eps, size);
    });
  NDArray::MarkUsedBy({out_grads, in_arr, grad_arr, mean_arr, var_arr,
                       dbias_arr, dy_mul_x_arr, dscale_arr},
                      stream);
}

} // namespace impl
} // namespace hetu
