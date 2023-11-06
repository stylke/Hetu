#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include "hetu/impl/stream/CPUStream.h"

namespace hetu {
namespace impl {

template <typename spec_t>
void transpose_cpu(const spec_t* input, spec_t* output, const uint* buf,
                   const uint ndims, size_t size) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    const uint* in_strides = buf;
    const uint* out_strides = buf + ndims;
    const uint* perm = buf + ndims * 2;
    uint i_idx = 0;
    uint t = idx;
    for (size_t i = 0; i < ndims; ++i) {
      const uint ratio = t / out_strides[i];
      t -= ratio * out_strides[i];
      i_idx += ratio * in_strides[perm[i]];
    }
    output[idx] = input[i_idx];
  }
}

void TransposeCpu(const NDArray& input, NDArray& output, int64_t* perm,
                  const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);

  CPUStream cpu_stream(stream);

  uint ndim = uint(input->ndim());
  uint ndim_ = uint(output->ndim());
  HT_ASSERT(ndim == ndim_);
  const int64_t* in_dims = input->shape().data();
  const int64_t* out_dims = output->shape().data();
  uint* buf = (uint*) malloc(3 * ndim * sizeof(uint));
  uint in_stride = 1;
  uint out_stride = 1;
  for (int i = ndim - 1; i >= 0; --i) {
    buf[i] = uint(in_stride);
    buf[ndim + i] = uint(out_stride);
    buf[ndim * 2 + i] = uint(perm[i]);
    in_stride *= uint(in_dims[i]);
    out_stride *= uint(out_dims[i]);
  }
  HT_ASSERT(in_stride == out_stride);
  size_t size = in_stride;
  if (size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "TransposeCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
        [input, output, buf, ndim ,size]() {
        transpose_cpu<spec_t>(input->data_ptr<spec_t>(),
                              output->data_ptr<spec_t>(), buf, ndim, size);
        free(buf);
        },"Transpose");
      
    });
}

} // namespace impl
} // namespace hetu
