#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/cuda/CUDADnn.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"

namespace hetu {
namespace impl {

void MaxPoolCuda(const NDArray& input, const size_t kernel_H,
                 const size_t kernel_W, NDArray& output, const size_t padding,
                 const size_t stride, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  cudnnHandle_t handle = hetu::impl::GetCudnnHandle(cuda_stream.device_id());
  // input
  size_t input_N = input->shape(0);
  size_t input_C = input->shape(1);
  size_t input_H = input->shape(2);
  size_t input_W = input->shape(3);

  // output
  size_t output_H = output->shape(2);
  size_t output_W = output->shape(3);

  cudnnDataType_t datatype;
  if (input->dtype() == DataType::FLOAT32) {
    datatype = CUDNN_DATA_FLOAT;
  } else if (input->dtype() == DataType::FLOAT64) {
    datatype = CUDNN_DATA_DOUBLE;
  }

  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "MaxPoolCuda", [&]() {
      const spec_t* input_data = (const spec_t*) input->data_ptr<spec_t>();
      spec_t* output_data = (spec_t*) output->data_ptr<spec_t>();
      // pooling descriptor
      cudnnPoolingDescriptor_t maxpool_desc;
      CUDNN_CALL(cudnnCreatePoolingDescriptor(&maxpool_desc));
      CUDNN_CALL(cudnnSetPooling2dDescriptor(
        maxpool_desc, CUDNN_POOLING_MAX_DETERMINISTIC, CUDNN_PROPAGATE_NAN,
        kernel_H, kernel_W, padding, padding, stride, stride));

      // input descriptor
      cudnnTensorDescriptor_t input_desc;
      CUDNN_CALL(cudnnCreateTensorDescriptor(&input_desc));
      CUDNN_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW,
                                            datatype, input_N, input_C, input_H,
                                            input_W));

      // output descriptor
      cudnnTensorDescriptor_t output_desc;
      CUDNN_CALL(cudnnCreateTensorDescriptor(&output_desc));
      CUDNN_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW,
                                            datatype, input_N, input_C,
                                            output_H, output_W));

      spec_t alpha = 1.0;
      spec_t beta = 0.0;

      CUDNN_CALL(cudnnPoolingForward(handle, maxpool_desc, &alpha, input_desc,
                                     input_data, &beta, output_desc,
                                     output_data));

      CUDNN_CALL(cudnnDestroyTensorDescriptor(input_desc));
      CUDNN_CALL(cudnnDestroyTensorDescriptor(output_desc));
      CUDNN_CALL(cudnnDestroyPoolingDescriptor(maxpool_desc));
    });
}

void MaxPoolGradientCuda(const NDArray& output_Y, const NDArray& gradient_Y,
                         const NDArray& input_X, const size_t kernel_H,
                         const size_t kernel_W, NDArray& gradient_X,
                         const size_t padding, const size_t stride,
                         const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(output_Y);
  HT_ASSERT_SAME_DEVICE(output_Y, gradient_Y);
  HT_ASSERT_SAME_DEVICE(output_Y, input_X);
  HT_ASSERT_SAME_DEVICE(output_Y, gradient_X);

  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  cudnnHandle_t handle = hetu::impl::GetCudnnHandle(cuda_stream.device_id());

  // input
  size_t input_N = input_X->shape(0);
  size_t input_C = input_X->shape(1);
  size_t input_H = input_X->shape(2);
  size_t input_W = input_X->shape(3);
  // output
  size_t output_H = output_Y->shape(2);
  size_t output_W = output_Y->shape(3);

  cudnnDataType_t datatype;
  if (output_Y->dtype() == DataType::FLOAT32) {
    datatype = CUDNN_DATA_FLOAT;
  } else if (output_Y->dtype() == DataType::FLOAT64) {
    datatype = CUDNN_DATA_DOUBLE;
  }

  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    output_Y->dtype(), spec_t, "MaxPoolGradientCuda", [&]() {
      const spec_t* input_data = (const spec_t*) input_X->data_ptr<spec_t>();
      spec_t* gradient_x_data = (spec_t*) gradient_X->data_ptr<spec_t>();
      const spec_t* output_data = (const spec_t*) output_Y->data_ptr<spec_t>();
      const spec_t* gradient_Y_data =
        (const spec_t*) gradient_Y->data_ptr<spec_t>();
      // pooling descriptor
      cudnnPoolingDescriptor_t maxpool_desc;
      CUDNN_CALL(cudnnCreatePoolingDescriptor(&maxpool_desc));
      CUDNN_CALL(cudnnSetPooling2dDescriptor(
        maxpool_desc, CUDNN_POOLING_MAX_DETERMINISTIC, CUDNN_PROPAGATE_NAN,
        kernel_H, kernel_W, padding, padding, stride, stride));

      // input descriptor
      cudnnTensorDescriptor_t input_desc;
      CUDNN_CALL(cudnnCreateTensorDescriptor(&input_desc));
      CUDNN_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW,
                                            datatype, input_N, input_C, input_H,
                                            input_W));

      // output descriptor
      cudnnTensorDescriptor_t output_desc;
      CUDNN_CALL(cudnnCreateTensorDescriptor(&output_desc));
      CUDNN_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW,
                                            datatype, input_N, input_C,
                                            output_H, output_W));

      spec_t alpha = 1.0;
      spec_t beta = 0.0;

      CUDNN_CALL(cudnnPoolingBackward(handle, maxpool_desc, &alpha, output_desc,
                                      output_data, output_desc, gradient_Y_data,
                                      input_desc, input_data, &beta, input_desc,
                                      gradient_x_data));
      CUDNN_CALL(cudnnDestroyTensorDescriptor(input_desc));
      CUDNN_CALL(cudnnDestroyTensorDescriptor(output_desc));
      CUDNN_CALL(cudnnDestroyPoolingDescriptor(maxpool_desc));
    });
}

} // namespace impl
} // namespace hetu
