#include "hetu/core/ndarray.h"
#include "hetu/core/memory_pool.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/cuda/CUDADnn.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"

namespace hetu {
namespace impl {

template <typename spec_t>
__global__ void softmax_cross_entropy_kernel(const spec_t* logsoftmax,
                                             const spec_t* label,
                                             spec_t* output, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  output[idx] = -logsoftmax[idx] * label[idx];
}

void SoftmaxCrossEntropyCuda(const NDArray& input, const NDArray& label,
                             NDArray& output, const Stream& stream) {
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  cudnnHandle_t handle = hetu::impl::GetCudnnHandle(cuda_stream.device_id());
  size_t indim = input->ndim();
  HT_ASSERT(indim == label->ndim() && indim == output->ndim() + 1)
    << "Indim is " << indim << ", Label dim is " << label->ndim()
    << ", Output dim is " << output->ndim();
  int n_ = 1;
  for (int i = 0; i < indim - 1; ++i) {
    n_ *= input->shape(i);
  }
  int c_ = input->shape(indim - 1);
  size_t size = n_ * c_;

  if (size == 0)
    return;

  int dev_id = cuda_stream.device_id();
  cudnnDataType_t datatype;
  cudnnIndicesType_t indicetype;
  if (input->dtype() == DataType::FLOAT32) {
    datatype = CUDNN_DATA_FLOAT;
    indicetype = CUDNN_32BIT_INDICES;
  } else if (input->dtype() == DataType::FLOAT64) {
    datatype = CUDNN_DATA_DOUBLE;
    indicetype = CUDNN_64BIT_INDICES;
  }

  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "SoftmaxCrossEntropyCuda", [&]() {
      spec_t alpha = 1.0;
      spec_t beta = 0.0;
      cudnnTensorDescriptor_t desc;
      CUDNN_CALL(cudnnCreateTensorDescriptor(&desc));
      CUDNN_CALL(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, datatype,
                                            n_, c_, 1, 1));
      DataPtr temp_data_ptr =
        AllocFromMemoryPool(input->device(), size * sizeof(spec_t));
      void* temp_data = temp_data_ptr.ptr;

      CUDNN_CALL(cudnnSoftmaxForward(
        handle, CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, desc,
        (const void*) input->data_ptr<spec_t>(), &beta, desc, temp_data));

      softmax_cross_entropy_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        (const spec_t*) temp_data, label->data_ptr<spec_t>(),
        (spec_t*) temp_data, size);

      cudnnReduceTensorDescriptor_t rtd;
      CUDNN_CALL(cudnnCreateReduceTensorDescriptor(&rtd));
      CUDNN_CALL(cudnnSetReduceTensorDescriptor(
        rtd, CUDNN_REDUCE_TENSOR_ADD, datatype, CUDNN_PROPAGATE_NAN,
        CUDNN_REDUCE_TENSOR_NO_INDICES, indicetype));

      cudnnTensorDescriptor_t new_desc;
      CUDNN_CALL(cudnnCreateTensorDescriptor(&new_desc));
      CUDNN_CALL(cudnnSetTensor4dDescriptor(new_desc, CUDNN_TENSOR_NCHW,
                                            datatype, n_, 1, 1, 1));
      CUDNN_CALL(cudnnReduceTensor(
        handle, rtd, NULL, 0, temp_data, size * sizeof(spec_t), &alpha, desc,
        (const void*) temp_data, &beta, new_desc, output->data_ptr<spec_t>()));

      CUDNN_CALL(cudnnDestroyReduceTensorDescriptor(rtd));
      CUDNN_CALL(cudnnDestroyTensorDescriptor(new_desc));
      CUDNN_CALL(cudnnDestroyTensorDescriptor(desc));
      FreeToMemoryPool(temp_data_ptr);
    });
}

template <typename spec_t>
__global__ void softmax_cross_entropy_gradient_kernel(
  const spec_t* pred, const spec_t* y_, const spec_t* grad_data,
  spec_t* output_data, int last_dim, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  output_data[idx] = (pred[idx] - y_[idx]) * grad_data[idx / last_dim];
}

void SoftmaxCrossEntropyGradientCuda(const NDArray& input_y,
                                     const NDArray& label, const NDArray& grad,
                                     NDArray& output, const Stream& stream) {
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  cudnnHandle_t handle = hetu::impl::GetCudnnHandle(cuda_stream.device_id());
  size_t indim = input_y->ndim();
  HT_ASSERT(indim == label->ndim() && indim == output->ndim() &&
            indim == grad->ndim() + 1)
    << "Indim is " << indim << ", Label dim is " << label->ndim()
    << ", Output dim is " << output->ndim();
  int n_ = 1;
  for (int i = 0; i < indim - 1; ++i) {
    n_ *= input_y->shape(i);
  }
  int c_ = input_y->shape(indim - 1);
  size_t size = n_ * c_;

  cudnnDataType_t datatype;
  if (input_y->dtype() == DataType::FLOAT32) {
    datatype = CUDNN_DATA_FLOAT;
  } else if (input_y->dtype() == DataType::FLOAT64) {
    datatype = CUDNN_DATA_DOUBLE;
  }

  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);

  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input_y->dtype(), spec_t, "SoftmaxCrossEntropyCuda", [&]() {
      int dev_id = cuda_stream.device_id();

      DataPtr temp_data_ptr =
        AllocFromMemoryPool(grad->device(), size * sizeof(spec_t));
      void* temp_data = temp_data_ptr.ptr;

      spec_t alpha = 1.0;
      spec_t beta = 0.0;
      cudnnTensorDescriptor_t desc;
      CUDNN_CALL(cudnnCreateTensorDescriptor(&desc));
      CUDNN_CALL(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, datatype,
                                            n_, c_, 1, 1));
      CUDNN_CALL(cudnnSoftmaxForward(
        handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha,
        desc, input_y->data_ptr<spec_t>(), &beta, desc, temp_data));

      softmax_cross_entropy_gradient_kernel<spec_t>
        <<<blocks, threads, 0, cuda_stream>>>(
          (const spec_t*) temp_data, label->data_ptr<spec_t>(),
          grad->data_ptr<spec_t>(), output->data_ptr<spec_t>(), c_, size);

      CUDNN_CALL(cudnnDestroyTensorDescriptor(desc));
      FreeToMemoryPool(temp_data_ptr);
    });
}

} // namespace impl
} // namespace hetu
