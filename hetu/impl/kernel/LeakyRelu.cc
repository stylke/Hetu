#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"

namespace hetu {
namespace impl {

template <typename spec_t>
void leaky_relu_cpu(const spec_t* input, spec_t alpha, size_t size,
                    spec_t* output) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    output[idx] = input[idx];
    if (input[idx] < 0)
      output[idx] *= alpha;
  }
}

template <typename spec_t>
void leaky_relu_gradient_cpu(const spec_t* input, const spec_t* output_grad,
                             spec_t alpha_reciprocal, size_t size,
                             spec_t* output) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    output[idx] = output_grad[idx];
    if (input[idx] < 0)
      output[idx] *= alpha_reciprocal;
  }
}

void LeakyReluCpu(const NDArray& input, double alpha, NDArray& output,
                  const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_EXCHANGABLE(input, output);

  size_t size = output->numel();
  if (size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "LeakyReluCpu", [&]() {
      leaky_relu_cpu<spec_t>(input->data_ptr<spec_t>(), alpha, size,
                             output->data_ptr<spec_t>());
    });
}

void LeakyReluGradientCpu(const NDArray& input, const NDArray& output_grad,
                          double alpha, NDArray& input_grad,
                          const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output_grad);
  HT_ASSERT_SAME_DEVICE(input, input_grad);
  HT_ASSERT_SAME_DEVICE(input, output_grad);
  HT_ASSERT_SAME_DEVICE(input, input_grad);

  size_t size = input_grad->numel();
  if (size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "LeakyReluGradientCpu", [&]() {
      leaky_relu_gradient_cpu<spec_t>(
        input->data_ptr<spec_t>(), output_grad->data_ptr<spec_t>(),
        static_cast<spec_t>(1.0 / alpha), size, input_grad->data_ptr<spec_t>());
    });
}

} // namespace impl
} // namespace hetu
