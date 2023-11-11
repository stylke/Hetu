#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/ndarray_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include "hetu/impl/stream/CPUStream.h"

namespace hetu {
namespace impl {

template <typename spec_t>
void add_const_cpu(const spec_t* input, spec_t value, size_t size,
                   spec_t* output) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; idx++)
    output[idx] = input[idx] + value;
}

template <typename spec_t>
void sgd_update_cpu(const spec_t* grad, spec_t* param, float lr, size_t size) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; idx++)
    param[idx] -= lr * grad[idx];
}

template <typename spec_t>
void momentum_update_cpu(const spec_t* grad, spec_t* param, spec_t* velocity,
                         float lr, float momentum, size_t size) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; idx++) {
    velocity[idx] = momentum * velocity[idx] - lr * grad[idx];
    param[idx] = param[idx] + velocity[idx];
  }
}

template <typename spec_t>
void nesterov_momentum_update_cpu(const spec_t* grad, spec_t* param,
                                  spec_t* velocity, float lr, float momentum,
                                  size_t size) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; idx++) {
    float temp = lr * grad[idx];
    velocity[idx] = momentum * (velocity[idx] - temp);
    param[idx] = param[idx] + velocity[idx] - temp;
  }
}

void SGDUpdateCpu(const NDArray& grad, NDArray& param, NDArray& velocity,
                  float lr, float momentum, bool nesterov,
                  const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(grad);
  HT_ASSERT_CPU_DEVICE(param);
  HT_ASSERT_EXCHANGABLE(grad, param);
  CPUStream cpu_stream(stream);

  if (momentum != 0) {
    HT_ASSERT_CPU_DEVICE(velocity);
    HT_ASSERT_EXCHANGABLE(velocity, param);
  }
  size_t size = grad->numel();
  if (size == 0)
    return;
  HT_DISPATCH_FLOATING_TYPES(grad->dtype(), spec_t, "SGDUpdateCpu", [&]() {
    auto _future = cpu_stream.EnqueueTask(
    [momentum, grad, param, velocity, lr, nesterov, size]() {
      if (momentum == 0) {
        sgd_update_cpu<spec_t>(grad->data_ptr<spec_t>(),
                              param->data_ptr<spec_t>(), lr, size);
      } else if (!nesterov) {
        momentum_update_cpu<spec_t>(
          grad->data_ptr<spec_t>(), param->data_ptr<spec_t>(),
          velocity->data_ptr<spec_t>(), lr, momentum, size);
      } else {
        nesterov_momentum_update_cpu<spec_t>(
          grad->data_ptr<spec_t>(), param->data_ptr<spec_t>(),
          velocity->data_ptr<spec_t>(), lr, momentum, size);
      }
    },"SGDUpdate");
  });
 NDArray::MarkUsedBy({grad, param, velocity}, stream);
}

} // namespace impl
} // namespace hetu
