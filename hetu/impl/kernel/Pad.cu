#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"

namespace hetu {
namespace impl {

template <typename spec_t>
__global__ void
pad_kernel(const spec_t* input_data, spec_t* output_data, size_t begin_N,
           size_t end_N, size_t N, size_t begin_C, size_t end_C, size_t C,
           size_t begin_H, size_t end_H, size_t H, size_t begin_W, size_t end_W,
           size_t W, spec_t constant_value) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N * C * H * W)
    return;
  size_t idx_N = idx / (C * H * W);
  size_t idx_C = idx % (C * H * W) / (H * W);
  size_t idx_H = idx % (H * W) / W;
  size_t idx_W = idx % W;
  if (idx_N >= begin_N && idx_N < end_N && idx_C >= begin_C && idx_C < end_C &&
      idx_H >= begin_H && idx_H < end_H && idx_W >= begin_W && idx_W < end_W) {
    output_data[idx] =
      input_data[(((idx_N - begin_N) * (end_C - begin_C) + idx_C - begin_C) *
                    (end_H - begin_H) +
                  idx_H - begin_H) *
                   (end_W - begin_W) +
                 idx_W - begin_W];
  } else {
    output_data[idx] = constant_value;
  }
}

template <typename spec_t>
__global__ void
pad_gradient_kernel(const spec_t* output_grad, spec_t* input_grad, size_t N,
                    size_t C, size_t H, size_t W, size_t begin_N,
                    size_t begin_C, size_t begin_H, size_t begin_W,
                    size_t out_N, size_t out_C, size_t out_H, size_t out_W) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N * C * H * W)
    return;
  size_t idx_N = idx / (C * H * W);
  size_t idx_C = idx % (C * H * W) / (H * W);
  size_t idx_H = idx % (H * W) / W;
  size_t idx_W = idx % W;
  input_grad[idx] = output_grad[(
    (((idx_N + begin_N) * out_C + idx_C + begin_C) * out_H + idx_H + begin_H) *
      out_W +
    idx_W + begin_W)];
}

void PadCuda(const NDArray& input, NDArray& output, const HTShape& paddings,
             const Stream& stream, std::string mode = "constant",
             double constant_values = 0) {
  HT_ASSERT(input->is_cuda()) << "Input is not on a host device.";
  HT_ASSERT(output->is_cuda()) << "Output is not on a host device.";
  HT_ASSERT(input->device() == output->device())
    << "Input and output are not on the same host device. "
    << "Devices: (input) " << input->device() << " vs. (output) "
    << output->device();
  size_t pad_len = paddings.size();
  size_t len = pad_len;
  size_t endpoint[8];
  for (int i = 0; i < 4; i++) {
    if (i < (4 - len / 2)) {
      HT_ASSERT((input->shape(i)) == (output->shape(i)));
      // endpoint[i * 2] = input->shape(i) - 1;
      // endpoint[i * 2 + 1] = endpoint[i * 2] + 1;
      endpoint[i * 2] = 0;
      endpoint[i * 2 + 1] = input->shape(i);
    } else {
      HT_ASSERT((input->shape(i) + paddings[(i - (4 - len / 2)) * 2] +
                 paddings[(i - (4 - len / 2)) * 2 + 1]) == (output->shape(i)));
      endpoint[i * 2] = paddings[(i - (4 - len / 2)) * 2];
      endpoint[i * 2 + 1] = paddings[(i - (4 - len / 2)) * 2] + input->shape(i);
    }
  }
  size_t size = output->numel();
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  if (mode == "constant") {
    HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
      input->dtype(), spec_t, "PadCuda", [&]() {
        pad_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
          input->data_ptr<spec_t>(), output->data_ptr<spec_t>(), endpoint[0],
          endpoint[1], output->shape(0), endpoint[2], endpoint[3],
          output->shape(1), endpoint[4], endpoint[5], output->shape(2),
          endpoint[6], endpoint[7], output->shape(3), constant_values);
      });
  }
}

void PadGradientCuda(const NDArray& output_grad, NDArray& input_grad,
                     const HTShape& paddings, const Stream& stream,
                     std::string mode = "constant") {
  HT_ASSERT(output_grad->is_cuda()) << "Output_grad is not on a host device.";
  HT_ASSERT(input_grad->is_cuda()) << "Input_grad is not on a host device.";
  HT_ASSERT(input_grad->device() == output_grad->device())
    << "input and output grads are not on the same host device. "
    << "Devices: (input_grad) " << input_grad->device() << " vs. (output_grad) "
    << output_grad->device();
  size_t pad_len = paddings.size();
  size_t len = pad_len;
  size_t begin_p[4];
  size_t N = input_grad->shape(0);
  size_t C = input_grad->shape(1);
  size_t H = input_grad->shape(2);
  size_t W = input_grad->shape(3);

  size_t out_N = output_grad->shape(0);
  size_t out_C = output_grad->shape(1);
  size_t out_H = output_grad->shape(2);
  size_t out_W = output_grad->shape(3);

  for (int i = 0; i < 4; i++) {
    if (i < (4 - len / 2)) {
      begin_p[i] = 0;
    } else {
      begin_p[i] = paddings[(i - (4 - len / 2)) * 2];
    }
  }
  size_t size = input_grad->numel();
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  if (mode == "constant") {
    HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
      input_grad->dtype(), spec_t, "PadGradientCuda", [&]() {
        pad_gradient_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
          output_grad->data_ptr<spec_t>(), input_grad->data_ptr<spec_t>(), N, C,
          H, W, begin_p[0], begin_p[1], begin_p[2], begin_p[3], out_N, out_C,
          out_H, out_W);
      });
  }
}

} // namespace impl
} // namespace hetu
