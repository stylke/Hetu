#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/stream/CPUStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"

namespace hetu {
namespace impl {

template <typename spec_t, typename Operator>
void binary_const_cpu(const spec_t* input, spec_t value, size_t size,
                      Operator op, spec_t* output) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; idx++)
    output[idx] = op(value, input[idx]);
}

template<typename Operator>
void BinaryConstToolCpu(const NDArray& input, double value,
                        NDArray& output, Operator op, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);

  size_t size = input->numel();
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "BinaryConstCpu", [&]() {
      binary_const_cpu<spec_t>(
        input->data_ptr<spec_t>(), static_cast<spec_t>(value), size, op,
        output->data_ptr<spec_t>());
    });
}

void AddConstCpu(const NDArray& input, double value,
                 NDArray& output, const Stream& stream) {

  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
      input->dtype(), spec_t, "AddConstCpu", [&]() {
        auto op = std::plus<spec_t>();
        BinaryConstToolCpu(input, value, output, op, stream);
      }); 
}

void SubConstCpu(const NDArray& input, double value,
                 NDArray& output, const Stream& stream) {

  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
      input->dtype(), spec_t, "SubConstCpu", [&]() {
        auto op = std::minus<spec_t>();
        BinaryConstToolCpu(input, value, output, op, stream);
      }); 
}

void MulConstCpu(const NDArray& input, double value,
                 NDArray& output, const Stream& stream) {

  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
      input->dtype(), spec_t, "MulConstCpu", [&]() {
        auto op = std::multiplies<spec_t>();
        BinaryConstToolCpu(input, value, output, op, stream);
      }); 
}

void DivConstCpu(const NDArray& input, double value,
                 NDArray& output, const Stream& stream) {

  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
      input->dtype(), spec_t, "DivConstCpu", [&]() {
        auto op = std::divides<spec_t>();
        BinaryConstToolCpu(input, value, output, op, stream);
      }); 
}

} // namespace impl
} // namespace hetu
