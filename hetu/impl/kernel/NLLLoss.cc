#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"

namespace hetu {
namespace impl {

template <typename spec_t>
void nllloss_cpu(const spec_t* pred, const int64_t* label, 
                    size_t n_rows, size_t n_cols, spec_t* loss) {
  for (int idx = 0; idx < n_rows; ++idx) {
    int64_t id = label[idx];
    if (id < 0 || id >= n_cols) {
      loss[idx] = 0;
    } else {
      loss[idx] = -pred[n_cols * idx + id];
    }
  }
}

template <typename spec_t>
void nllloss_gradient_cpu(const spec_t* pred, const int64_t* label,
                          const spec_t* grad_loss, size_t n_rows, size_t n_cols,
                          spec_t* output) {
  for (int idx = 0; idx < n_rows; ++idx) {
    int64_t id = label[idx];
    if (id < 0 || id >= n_cols) {
      output[n_cols * idx + id] = 0;
    } else {
      output[n_cols * idx + id] = - grad_loss[idx];
    }
  }
}


void NLLLossCpu(const NDArray& pred, const NDArray& label,
                NDArray& loss, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(pred);
  HT_ASSERT_SAME_DEVICE(pred, label);
  HT_ASSERT_SAME_DEVICE(pred, loss);
  HT_ASSERT_SAME_SHAPE(label, loss);
  HT_ASSERT(pred->ndim() == label->ndim() + 1);
  for (size_t i = 0; i < label->ndim(); i++)
    HT_ASSERT(pred->shape(i) == label->shape(i));

  size_t n_rows = 1, n_cols;
  for (size_t i = 0; i < label->ndim(); i++)
    n_rows *= label->shape(i);
  n_cols = pred->shape(pred->ndim() - 1);
  if (n_rows == 0)
    return;

  HT_DISPATCH_FLOATING_TYPES(
    pred->dtype(), spec_t, "NLLLossCpu", [&]() {
      nllloss_cpu(
        pred->data_ptr<spec_t>(), label->data_ptr<int64_t>(), n_rows, n_cols,
        loss->data_ptr<spec_t>());
    });
}

void NLLLossGradientCpu(const NDArray& pred, const NDArray& label,
                        const NDArray& grad_loss, NDArray& output,
                        const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(pred);
  HT_ASSERT_SAME_DEVICE(pred, label);
  HT_ASSERT_SAME_DEVICE(pred, grad_loss);
  HT_ASSERT_SAME_DEVICE(pred, output);

  size_t n_rows = 1, n_cols;
  for (size_t i = 0; i < pred->ndim(); i++)
    n_rows *= label->shape(i);
  n_cols = pred->shape(pred->ndim() - 1);
  if (n_rows == 0)
    return;
  HT_DISPATCH_FLOATING_TYPES(
    pred->dtype(), spec_t, "NLLLossGradientCpu", [&]() {
      nllloss_gradient_cpu(
        pred->data_ptr<spec_t>(), label->data_ptr<int64_t>(),
        grad_loss->data_ptr<spec_t>(), n_rows, n_cols, output->data_ptr<spec_t>());
    });
}

} // namespace impl
} // namespace hetu
