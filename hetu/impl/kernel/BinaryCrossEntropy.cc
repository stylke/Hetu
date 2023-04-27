#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/stream/CPUStream.h"
#include "hetu/impl/utils/omp_utils.h"
#include <cmath>

namespace hetu {
namespace impl {

template <typename spec_t>
void binary_cross_entropy_cpu(const spec_t* pred, const spec_t* label,
                              size_t n_rows, spec_t* loss) {
  for (size_t idx = 0; idx < n_rows; idx++) {
    spec_t v1 = std::log(pred[idx]);
    spec_t v2 = std::log(1 - pred[idx]);
    // clip to -100 following PyTorch
    constexpr spec_t min_value = -100;
    loss[idx] =
      -label[idx] * MAX(v1, min_value) - (1 - label[idx]) * MAX(v2, min_value);
  }
}

template <typename spec_t>
void binary_cross_entropy_gradient_cpu(const spec_t* pred, const spec_t* label,
                                       const spec_t* grad_loss, size_t n_rows,
                                       spec_t* output) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < n_rows; idx++) {
    spec_t denominator = pred[idx] * (1 - pred[idx]);
    output[idx] = grad_loss[idx] * (pred[idx] - label[idx]) / MAX(denominator, 1e-12);
  }
}

void BinaryCrossEntropyCpu(const NDArray& pred, const NDArray& label,
                           NDArray& loss, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(pred);
  HT_ASSERT_SAME_DEVICE(pred, label);
  HT_ASSERT_SAME_DEVICE(pred, loss);
  HT_ASSERT_SAME_NDIM(pred, label);
  HT_ASSERT_SAME_NDIM(pred, loss);

  CPUStream cpu_stream(stream);
  dnnl::engine eng(dnnl::engine::kind::cpu, cpu_stream.stream_id());

  size_t n_rows = 1;
  for (size_t i = 0; i < pred->ndim() - 1; i++)
    n_rows *= pred->shape(i);
  if (n_rows == 0)
    return;
  HT_DISPATCH_FLOATING_TYPES(
    pred->dtype(), spec_t, "BinaryCrossEntropyCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [pred, label, loss, n_rows]() {
      binary_cross_entropy_cpu(pred->data_ptr<spec_t>(),
                               label->data_ptr<spec_t>(), n_rows,
                               loss->data_ptr<spec_t>());
      },
      "BinaryCrossEntropy");
      //cpu_stream.Sync();
    });
}

void BinaryCrossEntropyGradientCpu(const NDArray& pred, const NDArray& label,
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

  size_t n_rows = 1;
  for (size_t i = 0; i < pred->ndim() - 1; i++)
    n_rows *= pred->shape(i);
  if (n_rows == 0)
    return;
  HT_DISPATCH_FLOATING_TYPES(
    pred->dtype(), spec_t, "BinaryCrossEntropyGradientCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [pred, label, grad_loss, output, n_rows]() {
      binary_cross_entropy_gradient_cpu(
        pred->data_ptr<spec_t>(), label->data_ptr<spec_t>(),
        grad_loss->data_ptr<spec_t>(), n_rows, output->data_ptr<spec_t>());
      },
      "BinaryCrossEntropyGradient");
      //cpu_stream.Sync();
    });
}

} // namespace impl
} // namespace hetu
