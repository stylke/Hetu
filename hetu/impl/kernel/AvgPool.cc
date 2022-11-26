#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"

namespace hetu {
namespace impl {

template <typename spec_t>
void avgpool_cpu(const size_t threads, const spec_t* input_data,
                 spec_t* output_data, const size_t N, const size_t C,
                 const size_t H, const size_t W, const size_t kernel_H,
                 const size_t kernel_W, const size_t p_H, const size_t p_W,
                 const size_t padding, const size_t stride) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < threads; ++idx) {
    size_t idx_W = idx % p_W;
    idx /= p_W;
    size_t idx_H = idx % p_H;
    idx /= p_H;
    size_t idx_C = idx % C;
    size_t idx_N = idx / C;
    int hs = (int) idx_H * stride - padding;
    int ws = (int) idx_W * stride - padding;
    size_t hend = std::min(hs + kernel_H, H);
    size_t wend = std::min(ws + kernel_W, W);
    hs = std::max(hs, 0);
    ws = std::max(ws, 0);
    float temp = 0;
    for (size_t i = hs; i < hend; i++) {
      for (size_t j = ws; j < wend; j++) {
        temp += input_data[idx_N * C * H * W + idx_C * H * W + i * W + j];
      }
    }
    output_data[idx] = temp / (kernel_H * kernel_W);
  }
}

template <typename spec_t>
void avgpool_gradient_cpu(const size_t threads, const spec_t* input_data,
                          spec_t* output_data, const size_t N, const size_t C,
                          const size_t H, const size_t W, const size_t kernel_H,
                          const size_t kernel_W, const size_t p_H,
                          const size_t p_W, const size_t padding,
                          const size_t stride) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < threads; ++idx) {
    size_t idx_W = idx % p_W;
    idx /= p_W;
    size_t idx_H = idx % p_H;
    idx /= p_H;
    size_t idx_C = idx % C;
    size_t idx_N = idx / C;
    size_t hs = (idx_H < kernel_H) ? 0 : (idx_H - kernel_H) / stride + 1;
    size_t hend = std::min(idx_H / stride + 1, H);
    size_t ws = (idx_W < kernel_W) ? 0 : (idx_W - kernel_W) / stride + 1;
    size_t wend = std::min(idx_W / stride + 1, W);
    float temp = 0;
    const size_t pooling_size = kernel_H * kernel_W;
    for (size_t i = hs; i < hend; i++) {
      for (size_t j = ws; j < wend; j++) {
        temp += input_data[idx_N * C * H * W + idx_C * H * W + i * W + j];
      }
    }
    output_data[idx] = temp / pooling_size;
  }
}

void AvgPoolCpu(const NDArray& input, const size_t kernel_H,
                const size_t kernel_W, NDArray& output, const size_t padding,
                const size_t stride, const Stream& stream) {
  HT_ASSERT(input->is_cpu()) << "Input is not on a host device.";
  HT_ASSERT(output->is_cpu()) << "Output is not on a host device.";
  HT_ASSERT(input->device() == output->device())
    << "input and output are not on the same host device. "
    << "Devices: (input) " << input->device() << " vs. (output) "
    << output->device();
  size_t input_N = input->shape(0);
  size_t input_C = input->shape(1);
  size_t input_H = input->shape(2);
  size_t input_W = input->shape(3);
  size_t output_H = output->shape(2);
  size_t output_W = output->shape(3);
  size_t pooled_H = (input_H + 2 * padding - kernel_H) / stride + 1;
  size_t pooled_W = (input_W + 2 * padding - kernel_W) / stride + 1;
  size_t output_size = input_N * input_C * output_H * output_W;
  if (output_size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "AvgPoolCpu", [&]() {
      avgpool_cpu<spec_t>(output_size, input->data_ptr<spec_t>(),
                          output->data_ptr<spec_t>(), input_N, input_C, input_H,
                          input_W, kernel_H, kernel_W, pooled_H, pooled_W,
                          padding, stride);
    });
}

void AvgPoolGradientCpu(const NDArray& output_Y, const NDArray& gradient_Y,
                        const NDArray& input_X, const size_t kernel_H,
                        const size_t kernel_W, NDArray& gradient_X,
                        const size_t padding, const size_t stride,
                        const Stream& stream) {
  HT_ASSERT(output_Y->is_cpu()) << "Output is not on a host device.";
  HT_ASSERT(gradient_Y->is_cpu()) << "Output_grad is not on a host device.";
  HT_ASSERT(input_X->is_cpu()) << "Input is not on a host device.";
  HT_ASSERT(gradient_X->is_cpu()) << "Input_grad is not on a host device.";
  // HT_ASSERT(input_grad->device() == output_grad->device())
  //   << "input and output grads are not on the same host device. "
  //   << "Devices: (input_grad) " << input_grad->device()
  //   << " vs. (output_grad) " << output_grad->device();
  // HT_ASSERT(IsConcatable(output_grad, input_grad, axis))
  //   << "input and output are not Concatable.";
  size_t N = gradient_Y->shape(0);
  size_t C = gradient_Y->shape(1);
  size_t H = gradient_Y->shape(2);
  size_t W = gradient_Y->shape(3);

  size_t pooled_H = gradient_X->shape(2);
  size_t pooled_W = gradient_X->shape(3);

  size_t output_size = N * C * pooled_H * pooled_W;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input_X->dtype(), spec_t, "AvgPoolGradientCpu", [&]() {
      avgpool_gradient_cpu<spec_t>(output_size, gradient_Y->data_ptr<spec_t>(),
                                   gradient_X->data_ptr<spec_t>(), N, C, H, W,
                                   kernel_H, kernel_W, pooled_H, pooled_W,
                                   padding, stride);
    });
}

} // namespace impl
} // namespace hetu
