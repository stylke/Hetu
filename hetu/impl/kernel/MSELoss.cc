#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include "hetu/impl/stream/CPUStream.h"
#include <cmath>

namespace hetu {
namespace impl {

template <typename spec_t>
void mseloss_cpu(const spec_t* pred, const spec_t* label,
                              size_t n_rows, spec_t* loss) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < n_rows; idx++) {
    loss[idx] = (pred[idx] - label[idx]) * (pred[idx] - label[idx]);
  }
}

template <typename spec_t>
void mseloss_gradient_cpu(const spec_t* pred, const spec_t* label,
                                       const spec_t* grad_loss, size_t n_rows,
                                       spec_t* output) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < n_rows; idx++) {
    output[idx] = 2 * grad_loss[idx] * (pred[idx] - label[idx]);
  }
}

void MSELossCpu(const NDArray& pred, const NDArray& label,
                           NDArray& loss, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(pred);
  HT_ASSERT_SAME_DEVICE(pred, label);
  HT_ASSERT_SAME_DEVICE(pred, loss);
  HT_ASSERT_SAME_NDIM(pred, label);
  HT_ASSERT_SAME_NDIM(pred, loss);

  CPUStream cpu_stream(stream);

  size_t n_rows = 1;
  for (size_t i = 0; i < pred->ndim() - 1; i++)
    n_rows *= pred->shape(i);
  if (n_rows == 0)
    return;
  HT_DISPATCH_FLOATING_TYPES(
    pred->dtype(), spec_t, "MSELossCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [pred, label, loss, n_rows]() {
      mseloss_cpu(pred->data_ptr<spec_t>(),
                  label->data_ptr<spec_t>(), n_rows,
                  loss->data_ptr<spec_t>());
      },"MSELoss");
      //cpu_stream.Sync();
    });
}

void MSELossGradientCpu(const NDArray& pred, const NDArray& label,
                                   const NDArray& grad_loss, NDArray& output,
                                   const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(pred);
  HT_ASSERT_SAME_DEVICE(pred, label);
  HT_ASSERT_SAME_DEVICE(pred, grad_loss);
  HT_ASSERT_SAME_DEVICE(pred, output);
  HT_ASSERT_SAME_NDIM(pred, label);
  HT_ASSERT_SAME_NDIM(pred, grad_loss);
  HT_ASSERT_SAME_NDIM(pred, output);

  CPUStream cpu_stream(stream);
  dnnl::engine eng(dnnl::engine::kind::cpu, 0);

  size_t n_rows = 1;
  for (size_t i = 0; i < pred->ndim() - 1; i++)
    n_rows *= pred->shape(i);
  if (n_rows == 0)
    return;
  HT_DISPATCH_FLOATING_TYPES(
    pred->dtype(), spec_t, "MSELossGradientCuda", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [pred, label, grad_loss, output, n_rows]() {
      mseloss_gradient_cpu(
        pred->data_ptr<spec_t>(), label->data_ptr<spec_t>(),
        grad_loss->data_ptr<spec_t>(), n_rows, output->data_ptr<spec_t>());
      },"MSELossGradient");
      //cpu_stream.Sync();
    });
}

} // namespace impl
} // namespace hetu
