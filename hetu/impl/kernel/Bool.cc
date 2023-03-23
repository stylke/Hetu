#include "hetu/core/ndarray.h"
#include "hetu/impl/utils/common_utils.h"


namespace hetu {
namespace impl {

template <typename spec_t>
void bool_cpu(const spec_t* input, size_t size, bool* output) {
  for (size_t idx = 0; idx < size; ++idx) {
    if (input[idx] > 0)
      output[idx] = 1;
    else
      output[idx] = 0;
  }
}

void BoolCpu(const NDArray& input, NDArray& output, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT(output->dtype() == DataType::BOOL);

  size_t size = input->numel();
  if (size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "BoolCpu", [&]() {
      bool_cpu<spec_t>(
        input->data_ptr<spec_t>(), size, output->data_ptr<bool>());
    });
}

} // namespace impl
} // namespace hetu
