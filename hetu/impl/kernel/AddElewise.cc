#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/stream/CPUStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"

namespace hetu {
namespace impl {

template <typename spec_t>
void add_elewise_cpu(const spec_t* inputA, const spec_t* inputB, size_t size,
                     spec_t* output) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; idx++)
    output[idx] = inputA[idx] + inputB[idx];
}

template <typename spec_t>
void add_elewise_broadcast_cpu(const spec_t* inputA, const spec_t* inputB,
                               size_t size, spec_t* output, uint* A_dims,
                               uint* B_dims, size_t A_ndims, size_t B_ndims,
                               uint* out_strides, size_t out_dims) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; idx++) {
    size_t A_ind = 0;
    size_t temp = idx;
    for (int i = 0; i < out_dims; ++i) {
      A_ind *= A_dims[i];
      A_ind += (A_dims[i] > 1) * temp / out_strides[i];
      temp %= out_strides[i];
    }
    size_t B_ind = 0;
    temp = idx;
    for (int i = 0; i < out_dims; ++i) {
      B_ind *= B_dims[i];
      B_ind += (B_dims[i] > 1) * temp / out_strides[i];
      temp %= out_strides[i];
    }
    output[idx] = inputA[A_ind] + inputB[B_ind];
  }
}

void AddElewiseCpu(const NDArray& inputA, const NDArray& inputB,
                   NDArray& output, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(inputA);
  HT_ASSERT_SAME_DEVICE(inputA, output);
  HT_ASSERT_SAME_DEVICE(inputB, output);

  size_t size;
  size_t sizeA = inputA->numel();
  size_t sizeB = inputB->numel();
  if (sizeA == sizeB) {
    size = sizeA;
    HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
      inputA->dtype(), spec_t, "AddElewiseCuda", [&]() {
        add_elewise_cpu<spec_t>(inputA->data_ptr<spec_t>(),
                                inputB->data_ptr<spec_t>(), size,
                                output->data_ptr<spec_t>());
      });
  } else {
    size_t allocated = output->ndim() * sizeof(uint);
    uint* A_dims = (uint*) malloc(allocated);
    uint* B_dims = (uint*) malloc(allocated);
    uint* out_strides = (uint*) malloc(allocated);
    size_t output_dim = output->ndim();
    size_t output_size = 1;
    size_t diff = output_dim - inputA->ndim();
    for (int i = output_dim - 1; i >= 0; --i) {
      out_strides[i] = output_size;
      output_size *= output->shape(i);
      if (i < diff) {
        A_dims[i] = 1;
      } else {
        A_dims[i] = inputA->shape(i - diff);
      }
    }
    diff = output_dim - inputB->ndim();
    for (int i = output_dim - 1; i >= 0; --i) {
      if (i < diff) {
        B_dims[i] = 1;
      } else {
        B_dims[i] = inputB->shape(i - diff);
      }
    }
    size = output->numel();
    HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
      inputA->dtype(), spec_t, "AddElewiseCuda", [&]() {
        add_elewise_broadcast_cpu<spec_t>(
          inputA->data_ptr<spec_t>(), inputB->data_ptr<spec_t>(), size,
          output->data_ptr<spec_t>(), A_dims, B_dims, (size_t) inputA->ndim(),
          (size_t) inputB->ndim(), out_strides, output_dim);
      });
    free(A_dims);
    free(B_dims);
    free(out_strides);
  }
}

} // namespace impl
} // namespace hetu
