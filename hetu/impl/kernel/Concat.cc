#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"

namespace hetu {
namespace impl {

template <typename spec_t>
void Concat_cpu(const spec_t* inputA, const spec_t* inputB, size_t size,
                size_t concat_size, size_t offset1, size_t offset2,
                spec_t* output) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    int all_offset = offset1 + offset2;
    int post_ind = idx % concat_size;
    int temp = idx / concat_size;
    int mid_ind = temp % all_offset;
    int pre_ind = temp / all_offset;
    float val;
    if (mid_ind < (int) offset1) {
      int x_ind = (pre_ind * offset1 + mid_ind) * concat_size + post_ind;
      val = inputA[x_ind];
    } else {
      int y_ind =
        (pre_ind * offset2 + mid_ind - offset1) * concat_size + post_ind;
      val = inputB[y_ind];
    }
    output[idx] = val;
  }
}

template <typename spec_t>
void Concat_gradient_cpu(const spec_t* output_grad, size_t size,
                         int concat_size, int concat_offset, int small_offset,
                         int big_offset, spec_t* input_grad) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    int post_ind = idx % concat_size;
    int temp = idx / concat_size;
    int mid_ind = temp % small_offset + concat_offset;
    int pre_ind = temp / small_offset;
    int o_idx = (pre_ind * big_offset + mid_ind) * concat_size + post_ind;
    input_grad[idx] = output_grad[o_idx];
  }
}

void ConcatCpu(const NDArray& inputA, const NDArray& inputB, NDArray& output,
               size_t axis, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(inputA);
  HT_ASSERT_SAME_DEVICE(inputA, inputB);
  HT_ASSERT_SAME_DEVICE(inputA, output);

  size_t size = output->numel();
  size_t offset1 = inputA->shape(axis);
  size_t offset2 = inputB->shape(axis);
  int concat_size = 1;
  int now_ndim = inputA->ndim();
  for (int i = axis + 1; i < now_ndim; i++) {
    int cur_dim = inputA->shape(i);
    concat_size *= cur_dim;
  }
  if (size == 0 || offset1 == 0 || offset2 == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    inputA->dtype(), spec_t, "ConcatCpu", [&]() {
      Concat_cpu<spec_t>(inputA->data_ptr<spec_t>(), inputB->data_ptr<spec_t>(),
                         size, concat_size, offset1, offset2,
                         output->data_ptr<spec_t>());
    });
}

void ConcatGradientCpu(const NDArray& output_grad, NDArray& input_grad,
                       size_t axis, size_t id, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(output_grad);
  HT_ASSERT_SAME_DEVICE(output_grad, input_grad);

  size_t size = input_grad->numel();
  size_t big_offset = output_grad->shape(axis);
  size_t small_offset = input_grad->shape(axis);
  size_t concat_offset = (id == 1) ? (big_offset - small_offset) : 0;
  size_t concat_size = 1;
  size_t now_ndim = output_grad->ndim();
  for (size_t i = axis + 1; i < now_ndim; ++i) {
    int cur_dim = output_grad->shape(i);
    concat_size *= cur_dim;
  }
  if (size == 0 || small_offset == 0 || big_offset == 0)
    return;

  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input_grad->dtype(), spec_t, "ConcatGradientCpu", [&]() {
      Concat_gradient_cpu<spec_t>(output_grad->data_ptr<spec_t>(), size,
                                  concat_size, concat_offset, small_offset,
                                  big_offset, input_grad->data_ptr<spec_t>());
    });
}

} // namespace impl
} // namespace hetu
