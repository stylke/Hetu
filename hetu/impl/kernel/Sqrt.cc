#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include "cmath"

namespace hetu {
namespace impl {

template <typename spec_t>
void sqrt_cpu(const spec_t* input, size_t size, spec_t* output) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    output[idx] = std::sqrt(input[idx]);
  }
}

template <typename spec_t>
void reciprocal_sqrt_cpu(const spec_t* output_grad, size_t size,
                         spec_t* output) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    output[idx] = static_cast<spec_t>(1) / std::sqrt(output_grad[idx]);
  }
}

void SqrtCpu(const NDArray& input, NDArray& output, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_EXCHANGABLE(input, output);

  size_t size = output->numel();
  if (size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "SqrtCpu", [&]() {
      sqrt_cpu<spec_t>(input->data_ptr<spec_t>(), size,
                       output->data_ptr<spec_t>());
    });
}

void ReciprocalSqrtCpu(const NDArray& output_grad, NDArray& input_grad,
                       const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(output_grad);
  HT_ASSERT_SAME_DEVICE(output_grad, input_grad);
  HT_ASSERT_EXCHANGABLE(output_grad, input_grad);

  size_t size = input_grad->numel();
  if (size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input_grad->dtype(), spec_t, "ReciprocalSqrtCpu", [&]() {
      reciprocal_sqrt_cpu<spec_t>(output_grad->data_ptr<spec_t>(), size,
                                  input_grad->data_ptr<spec_t>());
    });
}

} // namespace impl
} // namespace hetu
