#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include "cmath"

namespace hetu {
namespace impl {

template <typename spec_t>
void pow_cpu(const spec_t* input, double exponent, size_t size,
             spec_t* output) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    output[idx] = std::pow(input[idx], exponent);
  }
}

void PowCpu(const NDArray& input, double exponent, NDArray& output,
            const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_EXCHANGABLE(input, output);

  size_t size = output->numel();
  if (size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "PowCpu", [&]() {
      pow_cpu<spec_t>(input->data_ptr<spec_t>(), exponent, size,
                      output->data_ptr<spec_t>());
    });
}

} // namespace impl
} // namespace hetu
