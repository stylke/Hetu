#include "hetu/core/ndarray.h"
#include "hetu/core/memory_pool.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/cuda/CUDADnn.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include <chrono>

namespace hetu {
namespace impl {

void Conv2dCuda(const NDArray& input_x, const NDArray& input_f, NDArray& output,
                const int padding_h, const int padding_w, const int stride_h,
                const int stride_w, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input_x);
  HT_ASSERT_SAME_DEVICE(input_x, input_f);
  HT_ASSERT_SAME_DEVICE(input_x, output);

  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  cudnnHandle_t handle = hetu::impl::GetCudnnHandle(cuda_stream.device_id());

  cudnnDataType_t datatype;
  if (input_x->dtype() == DataType::FLOAT32) {
    datatype = CUDNN_DATA_FLOAT;
  } else if (input_x->dtype() == DataType::FLOAT64) {
    datatype = CUDNN_DATA_DOUBLE;
  }

  size_t input_N = input_x->shape(0);
  size_t input_C = input_x->shape(1);
  size_t input_H = input_x->shape(2);
  size_t input_W = input_x->shape(3);

  size_t filter_N = input_f->shape(0);
  size_t filter_C = input_f->shape(1);
  size_t filter_H = input_f->shape(2);
  size_t filter_W = input_f->shape(3);

  size_t out_N = output->shape(0);
  size_t out_C = output->shape(1);
  size_t out_H = output->shape(2);
  size_t out_W = output->shape(3);

  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input_x->dtype(), spec_t, "Conv2dCuda", [&]() {
      const spec_t* input_data = (const spec_t*) input_x->data_ptr<spec_t>();

      // input
      cudnnTensorDescriptor_t input_desc;
      CUDNN_CALL(cudnnCreateTensorDescriptor(&input_desc));
      CUDNN_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW,
                                            datatype, input_N, input_C, input_H,
                                            input_W));

      const spec_t* filter_data = (const spec_t*) input_f->data_ptr<spec_t>();

      // filter
      cudnnFilterDescriptor_t filter_desc;
      CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_desc));
      CUDNN_CALL(cudnnSetFilter4dDescriptor(filter_desc, datatype,
                                            CUDNN_TENSOR_NCHW, filter_N,
                                            filter_C, filter_H, filter_W));

      // convolution
      cudnnConvolutionDescriptor_t conv_desc;
      CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
      CUDNN_CALL(cudnnSetConvolution2dDescriptor(
        conv_desc, padding_h, padding_w, stride_h, stride_w, 1, 1,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
      // output
      cudnnTensorDescriptor_t out_desc;
      CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
      CUDNN_CALL(cudnnSetTensor4dDescriptor(
        out_desc, CUDNN_TENSOR_NCHW, datatype, out_N, out_C, out_H, out_W));
      spec_t* output_data = (spec_t*) output->data_ptr<spec_t>();
      // algorithm
      cudnnConvolutionFwdAlgo_t algo;
      size_t workspace_size;

#if defined(CUDNN_MAJOR) && ((CUDNN_MAJOR >= 8))
      // workaround here
      // TODO: using cudnnFindConvolutionForwardAlgorithm in CuDNN 8 instead
      int return_algo_cnt = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
      cudnnConvolutionFwdAlgoPerf_t
        perf_results[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
      CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm_v7(
        handle, input_desc, filter_desc, conv_desc, out_desc,
        CUDNN_CONVOLUTION_FWD_ALGO_COUNT, &return_algo_cnt, perf_results));

      void* tmp_work_data = nullptr;
      for (int i = 0; i < return_algo_cnt; ++i) {
        CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
          handle, input_desc, filter_desc, conv_desc, out_desc,
          perf_results[i].algo, &workspace_size));
        if (cudaMalloc(&tmp_work_data, workspace_size) == cudaSuccess) {
          algo = perf_results[i].algo;
          CudaFree(tmp_work_data);
          break;
        }
      }
    // algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

#else
        CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm(
            handle, input_desc, filter_desc, conv_desc, out_desc, 
            CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));
#endif
      CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
        handle, input_desc, filter_desc, conv_desc, out_desc, algo,
        &workspace_size));

      DataPtr work_data_ptr =
        AllocFromMemoryPool(input_x->device(), workspace_size);
      void* work_data = work_data_ptr.ptr;

      spec_t alpha = 1.0;
      spec_t beta = 0.0;
      CUDNN_CALL(cudnnConvolutionForward(handle, &alpha, input_desc, input_data,
                                         filter_desc, filter_data, conv_desc,
                                         algo, work_data, workspace_size, &beta,
                                         out_desc, output_data));
      FreeToMemoryPool(work_data_ptr);
      CUDNN_CALL(cudnnDestroyTensorDescriptor(out_desc));
      CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
      CUDNN_CALL(cudnnDestroyFilterDescriptor(filter_desc));
      CUDNN_CALL(cudnnDestroyTensorDescriptor(input_desc));
    });
  return;
}

void Conv2dGradientofFilterCuda(const NDArray& input_x,
                                const NDArray& gradient_y, NDArray& gradient_f,
                                const int padding_h, const int padding_w,
                                const int stride_h, const int stride_w,
                                const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input_x);
  HT_ASSERT_SAME_DEVICE(input_x, gradient_y);
  HT_ASSERT_SAME_DEVICE(input_x, gradient_f);

  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  cudnnHandle_t handle = hetu::impl::GetCudnnHandle(cuda_stream.device_id());

  cudnnDataType_t datatype;
  if (input_x->dtype() == DataType::FLOAT32) {
    datatype = CUDNN_DATA_FLOAT;
  } else if (input_x->dtype() == DataType::FLOAT64) {
    datatype = CUDNN_DATA_DOUBLE;
  }

  // input
  size_t input_N = input_x->shape(0);
  size_t input_C = input_x->shape(1);
  size_t input_H = input_x->shape(2);
  size_t input_W = input_x->shape(3);
  // dy
  size_t dy_N = gradient_y->shape(0);
  size_t dy_C = gradient_y->shape(1);
  size_t dy_H = gradient_y->shape(2);
  size_t dy_W = gradient_y->shape(3);
  // dw
  size_t df_N = gradient_f->shape(0);
  size_t df_C = gradient_f->shape(1);
  size_t df_H = gradient_f->shape(2);
  size_t df_W = gradient_f->shape(3);

  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input_x->dtype(), spec_t, "Conv2dGradientofFilterCuda", [&]() {
      // input
      const spec_t* input_data = (const spec_t*) input_x->data_ptr<spec_t>();
      cudnnTensorDescriptor_t input_desc;
      CUDNN_CALL(cudnnCreateTensorDescriptor(&input_desc));
      CUDNN_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW,
                                            datatype, input_N, input_C, input_H,
                                            input_W));

      // dy
      const spec_t* dy_data = (const spec_t*) gradient_y->data_ptr<spec_t>();
      cudnnTensorDescriptor_t dy_desc;
      CUDNN_CALL(cudnnCreateTensorDescriptor(&dy_desc));
      CUDNN_CALL(cudnnSetTensor4dDescriptor(dy_desc, CUDNN_TENSOR_NCHW,
                                            datatype, dy_N, dy_C, dy_H, dy_W));

      // conv2d
      cudnnConvolutionDescriptor_t conv_desc;
      CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
      CUDNN_CALL(cudnnSetConvolution2dDescriptor(
        conv_desc, padding_h, padding_w, stride_h, stride_w, 1, 1,
        CUDNN_CROSS_CORRELATION, datatype));

      // dw
      spec_t* df_data = (spec_t*) gradient_f->data_ptr<spec_t>();
      cudnnFilterDescriptor_t df_desc;
      CUDNN_CALL(cudnnCreateFilterDescriptor(&df_desc));
      CUDNN_CALL(cudnnSetFilter4dDescriptor(
        df_desc, datatype, CUDNN_TENSOR_NCHW, df_N, df_C, df_H, df_W));
      // algo
      cudnnConvolutionBwdFilterAlgo_t algo;
      size_t workspace_size;

#if defined(CUDNN_MAJOR) && ((CUDNN_MAJOR >= 8))
      // TODO: using cudnnFindConvolutionBackwardFilterAlgorithm in CuDNN 8
      // instead algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT;
      int return_algo_cnt = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT;
      cudnnConvolutionBwdFilterAlgoPerf_t
        perf_results[CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT];
      CUDNN_CALL(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
        handle, input_desc, dy_desc, conv_desc, df_desc,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT, &return_algo_cnt,
        perf_results));

      void* tmp_work_data = nullptr;
      for (int i = 0; i < return_algo_cnt; ++i) {
        CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
          handle, input_desc, dy_desc, conv_desc, df_desc, perf_results[i].algo,
          &workspace_size));
        if (cudaMalloc(&tmp_work_data, workspace_size) == cudaSuccess) {
          algo = perf_results[i].algo;
          CudaFree(tmp_work_data);
          break;
        }
      }
#else
        CUDNN_CALL(cudnnGetConvolutionBackwardFilterAlgorithm(
            handle, input_desc, dy_desc, conv_desc, df_desc,
            CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &algo));
#endif
      CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        handle, input_desc, dy_desc, conv_desc, df_desc, algo,
        &workspace_size));
      DataPtr work_data_ptr =
        AllocFromMemoryPool(input_x->device(), workspace_size);
      void* work_data = work_data_ptr.ptr;
      spec_t alpha = 1.0;
      spec_t beta = 0.0;
      CUDNN_CALL(cudnnConvolutionBackwardFilter(
        handle, &alpha, input_desc, input_data, dy_desc, dy_data, conv_desc,
        algo, work_data, workspace_size, &beta, df_desc, df_data));
      FreeToMemoryPool(work_data_ptr);
      CUDNN_CALL(cudnnDestroyTensorDescriptor(dy_desc));
      CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
      CUDNN_CALL(cudnnDestroyFilterDescriptor(df_desc));
      CUDNN_CALL(cudnnDestroyTensorDescriptor(input_desc));
    });
  return;
}

void Conv2dGradientofDataCuda(const NDArray& input_f, const NDArray& gradient_y,
                              NDArray& gradient_x, const int padding_h,
                              const int padding_w, const int stride_h,
                              const int stride_w, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input_f);
  HT_ASSERT_SAME_DEVICE(input_f, gradient_y);
  HT_ASSERT_SAME_DEVICE(input_f, gradient_x);

  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  cudnnHandle_t handle = hetu::impl::GetCudnnHandle(cuda_stream.device_id());

  cudnnDataType_t datatype;
  if (input_f->dtype() == DataType::FLOAT32) {
    datatype = CUDNN_DATA_FLOAT;
  } else if (input_f->dtype() == DataType::FLOAT64) {
    datatype = CUDNN_DATA_DOUBLE;
  }
  // filter
  size_t filter_N = input_f->shape(0);
  size_t filter_C = input_f->shape(1);
  size_t filter_H = input_f->shape(2);
  size_t filter_W = input_f->shape(3);
  // dy
  size_t dy_N = gradient_y->shape(0);
  size_t dy_C = gradient_y->shape(1);
  size_t dy_H = gradient_y->shape(2);
  size_t dy_W = gradient_y->shape(3);
  // dx
  size_t dx_N = gradient_x->shape(0);
  size_t dx_C = gradient_x->shape(1);
  size_t dx_H = gradient_x->shape(2);
  size_t dx_W = gradient_x->shape(3);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input_f->dtype(), spec_t, "Conv2dGradientofDataCuda", [&]() {
      // filter
      const spec_t* filter_data = (const spec_t*) input_f->data_ptr<spec_t>();
      cudnnFilterDescriptor_t filter_desc;
      CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_desc));
      CUDNN_CALL(cudnnSetFilter4dDescriptor(filter_desc, datatype,
                                            CUDNN_TENSOR_NCHW, filter_N,
                                            filter_C, filter_H, filter_W));
      // dy
      const spec_t* dy_data = (const spec_t*) gradient_y->data_ptr<spec_t>();
      cudnnTensorDescriptor_t dy_desc;
      CUDNN_CALL(cudnnCreateTensorDescriptor(&dy_desc));
      CUDNN_CALL(cudnnSetTensor4dDescriptor(dy_desc, CUDNN_TENSOR_NCHW,
                                            datatype, dy_N, dy_C, dy_H, dy_W));
      // conv2d
      cudnnConvolutionDescriptor_t conv_desc;
      CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
      CUDNN_CALL(cudnnSetConvolution2dDescriptor(
        conv_desc, padding_h, padding_w, stride_h, stride_w, 1, 1,
        CUDNN_CROSS_CORRELATION, datatype));
      // dx
      spec_t* dx_data = (spec_t*) gradient_x->data_ptr<spec_t>();
      cudnnTensorDescriptor_t dx_desc;
      CUDNN_CALL(cudnnCreateTensorDescriptor(&dx_desc));
      CUDNN_CALL(cudnnSetTensor4dDescriptor(dx_desc, CUDNN_TENSOR_NCHW,
                                            datatype, dx_N, dx_C, dx_H, dx_W));

      // algo
      cudnnConvolutionBwdDataAlgo_t algo;
      size_t workspace_size;

#if defined(CUDNN_MAJOR) && ((CUDNN_MAJOR >= 8))
      // TODO: using cudnnFindConvolutionBackwardDataAlgorithm in CuDNN 8
      // instead
      int return_algo_cnt = CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT;
      cudnnConvolutionBwdDataAlgoPerf_t
        perf_results[CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT];
      CUDNN_CALL(cudnnGetConvolutionBackwardDataAlgorithm_v7(
        handle, filter_desc, dy_desc, conv_desc, dx_desc,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT, &return_algo_cnt, perf_results));

      void* tmp_work_data = nullptr;
      for (int i = 0; i < return_algo_cnt; ++i) {
        CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
          handle, filter_desc, dy_desc, conv_desc, dx_desc,
          perf_results[i].algo, &workspace_size));
        if (cudaMalloc(&tmp_work_data, workspace_size) == cudaSuccess) {
          algo = perf_results[i].algo;
          CudaFree(tmp_work_data);
          break;
        }
      }
#else
        CUDNN_CALL(cudnnGetConvolutionBackwardDataAlgorithm(
            handle, filter_desc, dy_desc, conv_desc, dx_desc,
            CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &algo));
#endif
      CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
        handle, filter_desc, dy_desc, conv_desc, dx_desc, algo,
        &workspace_size));

      DataPtr work_data_ptr =
        AllocFromMemoryPool(input_f->device(), workspace_size);
      void* work_data = work_data_ptr.ptr;

      spec_t alpha = 1.0;
      spec_t beta = 0.0;
      CUDNN_CALL(cudnnConvolutionBackwardData(
        handle, &alpha, filter_desc, filter_data, dy_desc, dy_data, conv_desc,
        algo, work_data, workspace_size, &beta, dx_desc, dx_data));

      FreeToMemoryPool(work_data_ptr);
      CUDNN_CALL(cudnnDestroyTensorDescriptor(dy_desc));
      CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
      CUDNN_CALL(cudnnDestroyTensorDescriptor(dx_desc));
      CUDNN_CALL(cudnnDestroyFilterDescriptor(filter_desc));
    });
  return;
}

template <typename spec_t>
__global__ void conv2d_add_bias_kernel(const spec_t* input, spec_t* output,
                                       size_t input_size, size_t output_size,
                                       size_t size) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  size_t input_idx = idx % input_size / output_size;
  output[idx] += input[input_idx];
}

void Conv2dAddBiasCuda(const NDArray& input_x, const NDArray& input_f,
                       const NDArray& bias, NDArray& output,
                       const int padding_h, const int padding_w,
                       const int stride_h, const int stride_w,
                       const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input_x);
  HT_ASSERT_SAME_DEVICE(input_x, input_f);
  HT_ASSERT_SAME_DEVICE(input_x, bias);
  HT_ASSERT_SAME_DEVICE(input_x, output);

  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  cudnnHandle_t handle = hetu::impl::GetCudnnHandle(cuda_stream.device_id());

  cudnnDataType_t datatype;
  if (input_f->dtype() == DataType::FLOAT32) {
    datatype = CUDNN_DATA_FLOAT;
  } else if (input_f->dtype() == DataType::FLOAT64) {
    datatype = CUDNN_DATA_DOUBLE;
  }

  size_t input_N = input_x->shape(0);
  size_t input_C = input_x->shape(1);
  size_t input_H = input_x->shape(2);
  size_t input_W = input_x->shape(3);

  size_t filter_N = input_f->shape(0);
  size_t filter_C = input_f->shape(1);
  size_t filter_H = input_f->shape(2);
  size_t filter_W = input_f->shape(3);

  size_t out_N = output->shape(0);
  size_t out_C = output->shape(1);
  size_t out_H = output->shape(2);
  size_t out_W = output->shape(3);

  // add bias
  size_t size = out_N * out_C * out_H * out_W;
  size_t bias_output_size = out_H * out_W;
  size_t bias_input_size = out_C * bias_output_size;
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input_x->dtype(), spec_t, "Conv2dAddBiasCuda", [&]() {
      const spec_t* input_data = (const spec_t*) input_x->data_ptr<spec_t>();
      // input
      cudnnTensorDescriptor_t input_desc;
      CUDNN_CALL(cudnnCreateTensorDescriptor(&input_desc));
      CUDNN_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW,
                                            datatype, input_N, input_C, input_H,
                                            input_W));

      const spec_t* filter_data = (const spec_t*) input_f->data_ptr<spec_t>();
      // filter
      cudnnFilterDescriptor_t filter_desc;
      CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_desc));
      CUDNN_CALL(cudnnSetFilter4dDescriptor(filter_desc, datatype,
                                            CUDNN_TENSOR_NCHW, filter_N,
                                            filter_C, filter_H, filter_W));

      // convolution
      cudnnConvolutionDescriptor_t conv_desc;
      CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
      CUDNN_CALL(cudnnSetConvolution2dDescriptor(
        conv_desc, padding_h, padding_w, stride_h, stride_w, 1, 1,
        CUDNN_CROSS_CORRELATION, datatype));

      // output
      cudnnTensorDescriptor_t out_desc;
      CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
      CUDNN_CALL(cudnnSetTensor4dDescriptor(
        out_desc, CUDNN_TENSOR_NCHW, datatype, out_N, out_C, out_H, out_W));
      spec_t* output_data = (spec_t*) output->data_ptr<spec_t>();
      // algorithm
      cudnnConvolutionFwdAlgo_t algo;
      size_t workspace_size;

#if defined(CUDNN_MAJOR) && ((CUDNN_MAJOR >= 8))
      // TODO: using cudnnFindConvolutionForwardAlgorithm in CuDNN 8 instead
      int return_algo_cnt = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
      cudnnConvolutionFwdAlgoPerf_t
        perf_results[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
      CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm_v7(
        handle, input_desc, filter_desc, conv_desc, out_desc,
        CUDNN_CONVOLUTION_FWD_ALGO_COUNT, &return_algo_cnt, perf_results));

      void* tmp_work_data = nullptr;
      for (int i = 0; i < return_algo_cnt; ++i) {
        CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
          handle, input_desc, filter_desc, conv_desc, out_desc,
          perf_results[i].algo, &workspace_size));
        if (cudaMalloc(&tmp_work_data, workspace_size) == cudaSuccess) {
          algo = perf_results[i].algo;
          CudaFree(tmp_work_data);
          break;
        }
      }
#else
        CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm(
            handle, input_desc, filter_desc, conv_desc, out_desc,
            CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));
#endif
      CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
        handle, input_desc, filter_desc, conv_desc, out_desc, algo,
        &workspace_size));

      DataPtr work_data_ptr =
        AllocFromMemoryPool(input_x->device(), workspace_size);
      void* work_data = work_data_ptr.ptr;

      spec_t alpha = 1.0;
      spec_t beta = 0.0;
      CUDNN_CALL(cudnnConvolutionForward(handle, &alpha, input_desc, input_data,
                                         filter_desc, filter_data, conv_desc,
                                         algo, work_data, workspace_size, &beta,
                                         out_desc, output_data));
      FreeToMemoryPool(work_data_ptr);
      CUDNN_CALL(cudnnDestroyTensorDescriptor(out_desc));
      CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
      CUDNN_CALL(cudnnDestroyFilterDescriptor(filter_desc));
      CUDNN_CALL(cudnnDestroyTensorDescriptor(input_desc));

      conv2d_add_bias_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        bias->data_ptr<spec_t>(), output->data_ptr<spec_t>(), bias_input_size,
        bias_output_size, size);
    });
  return;
}

} // namespace impl
} // namespace hetu
