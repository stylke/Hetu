#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/cuda_math.h"
#include "hetu/impl/utils/offset_calculator.cuh"

namespace hetu {
namespace impl {

template <typename spec_t>
__global__ void mseloss_kernel(const spec_t* pred, const spec_t* label,
                               size_t n_rows, spec_t* loss,
                               const OffsetCalculator* pred_offset_calculator,
                               const OffsetCalculator* label_offset_calculator,
                               const OffsetCalculator* loss_offset_calculator) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_rows)
    return;
  auto pred_offset = pred_offset_calculator->get(idx);
  auto label_offset = label_offset_calculator->get(idx);
  auto loss_offset = loss_offset_calculator->get(idx);
  loss[loss_offset] = (pred[pred_offset] - label[label_offset])
                    * (pred[pred_offset] - label[label_offset]);
}

template <typename spec_t>
__global__ void
mseloss_gradient_kernel(const spec_t* pred, const spec_t* label,
                        const spec_t* grad_loss, size_t n_rows, spec_t* output,
                        const OffsetCalculator* pred_offset_calculator,
                        const OffsetCalculator* label_offset_calculator,
                        const OffsetCalculator* grad_loss_offset_calculator,
                        const OffsetCalculator* out_offset_calculator) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_rows)
    return;
  auto pred_offset = pred_offset_calculator->get(idx);
  auto label_offset = label_offset_calculator->get(idx);
  auto grad_loss_offset = grad_loss_offset_calculator->get(idx);
  auto out_offset = out_offset_calculator->get(idx);
  output[out_offset] = 2 * grad_loss[grad_loss_offset] * (pred[pred_offset] - label[label_offset]);
}

void MSELossCuda(const NDArray& pred, const NDArray& label,
                 NDArray& loss, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(pred);
  HT_ASSERT_SAME_DEVICE(pred, label);
  HT_ASSERT_SAME_DEVICE(pred, loss);
  HT_ASSERT_SAME_NDIM(pred, label);
  HT_ASSERT_SAME_NDIM(pred, loss);

  size_t n_rows = 1;
  for (size_t i = 0; i < pred->ndim(); i++)
    n_rows *= pred->shape(i);
  if (n_rows == 0)
    return;
  dim3 blocks, threads;
  threads.x = MIN(n_rows, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(n_rows, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  NDArray pred_offset_calculator_arr, label_offset_calculator_arr,
          loss_offset_calculator_arr;
  OffsetCalculator *pred_offset_calculator, *label_offset_calculator,
                   *loss_offset_calculator;
  std::tie(pred_offset_calculator_arr, pred_offset_calculator) =
    AllocOffsetCalculator(pred, stream);
  std::tie(label_offset_calculator_arr, label_offset_calculator) =
    AllocOffsetCalculator(label, stream);
  std::tie(loss_offset_calculator_arr, loss_offset_calculator) = 
    AllocOffsetCalculator(loss, stream);
  HT_DISPATCH_FLOATING_TYPES(
    pred->dtype(), spec_t, "MSELossCuda", [&]() {
      mseloss_kernel<<<blocks, threads, 0, cuda_stream>>>(
        pred->data_ptr<spec_t>(), label->data_ptr<spec_t>(), n_rows,
        loss->data_ptr<spec_t>(), pred_offset_calculator,
        label_offset_calculator, loss_offset_calculator);
    });
  NDArray::MarkUsedBy({pred, label, loss, pred_offset_calculator_arr,
                      label_offset_calculator_arr, loss_offset_calculator_arr}, stream);
}

void MSELossGradientCuda(const NDArray& pred, const NDArray& label,
                         const NDArray& grad_loss, NDArray& output,
                         const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(pred);
  HT_ASSERT_SAME_DEVICE(pred, label);
  HT_ASSERT_SAME_DEVICE(pred, grad_loss);
  HT_ASSERT_SAME_DEVICE(pred, output);
  HT_ASSERT_SAME_NDIM(pred, label);
  HT_ASSERT_SAME_NDIM(pred, grad_loss);
  HT_ASSERT_SAME_NDIM(pred, output);

  size_t n_rows = 1;
  for (size_t i = 0; i < pred->ndim(); i++)
    n_rows *= pred->shape(i);
  if (n_rows == 0)
    return;
  dim3 blocks, threads;
  threads.x = MIN(n_rows, 1024);
  blocks.x = DIVUP(n_rows, 1024);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  NDArray pred_offset_calculator_arr, label_offset_calculator_arr,
          grad_loss_offset_calculator_arr, out_offset_calculator_arr;
  OffsetCalculator *pred_offset_calculator, *label_offset_calculator,
                   *grad_loss_offset_calculator, *out_offset_calculator;
  std::tie(pred_offset_calculator_arr, pred_offset_calculator) =
    AllocOffsetCalculator(pred, stream);
  std::tie(label_offset_calculator_arr, label_offset_calculator) =
    AllocOffsetCalculator(label, stream);
  std::tie(grad_loss_offset_calculator_arr, grad_loss_offset_calculator) =
    AllocOffsetCalculator(grad_loss, stream);
  std::tie(out_offset_calculator_arr, out_offset_calculator) = 
    AllocOffsetCalculator(output, stream);
  HT_DISPATCH_FLOATING_TYPES(
    pred->dtype(), spec_t, "MSELossGradientCuda", [&]() {
      mseloss_gradient_kernel<<<blocks, threads, 0, cuda_stream>>>(
        pred->data_ptr<spec_t>(), label->data_ptr<spec_t>(),
        grad_loss->data_ptr<spec_t>(), n_rows, output->data_ptr<spec_t>(),
        pred_offset_calculator, label_offset_calculator,
        grad_loss_offset_calculator, out_offset_calculator);
    });
  NDArray::MarkUsedBy({pred, label, grad_loss, output, pred_offset_calculator_arr,
                      label_offset_calculator_arr, grad_loss_offset_calculator_arr,
                      out_offset_calculator_arr}, stream);
}

} // namespace impl
} // namespace hetu
