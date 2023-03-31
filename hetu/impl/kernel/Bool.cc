#include "hetu/core/ndarray.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/stream/CPUStream.h"
#include "hetu/impl/utils/omp_utils.h"


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

  CPUStream cpu_stream(stream);
  dnnl::engine eng(dnnl::engine::kind::cpu, cpu_stream.stream_id());

  size_t size = input->numel();
  if (size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "BoolCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [input, output, size]() {
      bool_cpu<spec_t>(
        input->data_ptr<spec_t>(), size, output->data_ptr<bool>());
      },
      "Bool");
      //cpu_stream.Sync();
    });
}

} // namespace impl
} // namespace hetu
