#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/cuda/CUDADnn.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"

namespace hetu {
namespace impl {

void SoftmaxCuda(const NDArray& input, NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);

  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  cudnnHandle_t handle = hetu::impl::GetCudnnHandle(cuda_stream.device_id());
  size_t indim = input->ndim();
  HT_ASSERT(indim == output->ndim());
  int n_ = 1;
  for (int i = 0; i < indim - 1; ++i) {
    n_ *= input->shape(i);
  }
  int c_ = input->shape(indim - 1);

  cudnnDataType_t datatype;
  if (input->dtype() == DataType::FLOAT32) {
    datatype = CUDNN_DATA_FLOAT;
  } else if (input->dtype() == DataType::FLOAT64) {
    datatype = CUDNN_DATA_DOUBLE;
  }

  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "SoftmaxCuda", [&]() {
      const spec_t* input_data = (const spec_t*) (input->data_ptr<spec_t>());
      spec_t* output_data = (spec_t*) (output->data_ptr<spec_t>());
      spec_t alpha = 1.0;
      spec_t beta = 0.0;
      cudnnTensorDescriptor_t desc;
      CUDNN_CALL(cudnnCreateTensorDescriptor(&desc));
      CUDNN_CALL(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, datatype,
                                            n_, c_, 1, 1));
      CUDNN_CALL(cudnnSoftmaxForward(
        handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha,
        desc, (const void*) input_data, &beta, desc, (void*) output_data));
      CUDNN_CALL(cudnnDestroyTensorDescriptor(desc));
    });
}

void SoftmaxGradientCuda(const NDArray& input_Y, const NDArray& output_grad,
                         NDArray& input_grad, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input_Y);
  HT_ASSERT_SAME_DEVICE(input_Y, output_grad);
  HT_ASSERT_SAME_DEVICE(input_Y, input_grad);

  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  cudnnHandle_t handle = hetu::impl::GetCudnnHandle(cuda_stream.device_id());
  size_t indim = input_Y->ndim();
  HT_ASSERT(indim == output_grad->ndim() && indim == input_grad->ndim());
  int n_ = 1;
  for (int i = 0; i < indim - 1; ++i) {
    n_ *= input_Y->shape(i);
  }
  int c_ = input_Y->shape(indim - 1);

  cudnnDataType_t datatype;
  if (input_Y->dtype() == DataType::FLOAT32) {
    datatype = CUDNN_DATA_FLOAT;
  } else if (input_Y->dtype() == DataType::FLOAT64) {
    datatype = CUDNN_DATA_DOUBLE;
  }

  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input_Y->dtype(), spec_t, "SoftmaxGradientCuda", [&]() {
      const spec_t* y_data = (const spec_t*) (input_Y->data_ptr<spec_t>());
      const spec_t* output_grad_data =
        (const spec_t*) (output_grad->data_ptr<spec_t>());
      spec_t* input_grad_data = (spec_t*) (input_grad->data_ptr<spec_t>());
      spec_t alpha = 1.0;
      spec_t beta = 0.0;
      cudnnTensorDescriptor_t desc;
      CUDNN_CALL(cudnnCreateTensorDescriptor(&desc));
      CUDNN_CALL(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, datatype,
                                            n_, c_, 1, 1));
      CUDNN_CALL(cudnnSoftmaxBackward(
        handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha,
        desc, (const void*) y_data, desc, (const void*) output_grad_data, &beta,
        desc, (void*) input_grad_data));
      CUDNN_CALL(cudnnDestroyTensorDescriptor(desc));
    });
}

} // namespace impl
} // namespace hetu
