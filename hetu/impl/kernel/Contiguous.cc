#include "hetu/core/ndarray.h"
#include "hetu/core/memory_pool.h"
#include "hetu/impl/stream/CPUStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include <chrono>

namespace hetu {
namespace impl {

template <typename spec_t>
void contiguous_cpu(const spec_t* input, spec_t* output,
                    const int64_t* stride, const int64_t* new_stride,
                    const uint ndims, size_t size) {
  for (int64_t idx = 0; idx < size; ++idx) {
    int64_t i_idx = 0;
    int64_t t = idx;
    #pragma unroll
    for (int i = 0; i < ndims; ++i) {
      const int64_t ratio = t / new_stride[i];
      t -= ratio * new_stride[i];
      i_idx += ratio * stride[i];
    }
    output[idx] = input[i_idx];
  }
}

template <typename spec_t>
void contiguous_gradient_cpu(const spec_t* input, spec_t* output,
                             const int64_t* stride, const int64_t* new_stride,
                             const uint ndims, size_t size) {
  for (int64_t idx = 0; idx < size; ++idx) {
    int64_t i_idx = 0;
    int64_t t = idx;
    #pragma unroll
    for (int i = 0; i < ndims; ++i) {
      const int64_t ratio = t / stride[i];
      t -= ratio * stride[i];
      i_idx += ratio * new_stride[i];
    }
    output[i_idx] = input[idx];
  }
}

void ContiguousCpu(const NDArray& input, NDArray& output,
                    const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT(input->numel() == output->numel());

  int ndim = input->ndim();
  int64_t* stride = (int64_t*) malloc(ndim * sizeof(int64_t));
  int64_t* new_stride = (int64_t*) malloc(ndim * sizeof(int64_t));
  for (int i = 0; i < ndim; ++i) {
    stride[i] = input->stride(i);
    new_stride[i] = output->stride(i);
  }
  size_t size = output->numel();
  CPUStream cpu_stream(stream);
  

    
  if (size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "Contiguous", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [input, output, stride, new_stride, ndim, size]() {
      contiguous_cpu<spec_t>(input->data_ptr<spec_t>(), output->data_ptr<spec_t>(), 
                             stride, new_stride, ndim, size);
      free(stride);
      free(new_stride);
      },
      "Contiguous");
    });
}

void ContiguousGradientCpu(const NDArray& input, NDArray& output,
                           const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT(input->numel() == output->numel());

  int ndim = input->ndim();
  int64_t* stride = (int64_t*) malloc(ndim * sizeof(int64_t));
  int64_t* new_stride = (int64_t*) malloc(ndim * sizeof(int64_t));
  for (int i = 0; i < ndim; ++i) {
    stride[i] = input->stride(i);
    new_stride[i] = output->stride(i);
  }
  size_t size = output->numel();
  CPUStream cpu_stream(stream);

    
  if (size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "ContiguousGradient", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [input, output, stride, new_stride, ndim, size]() {
      contiguous_gradient_cpu<spec_t>(input->data_ptr<spec_t>(), output->data_ptr<spec_t>(), 
                                      stride, new_stride, ndim, size);
      free(stride);
      free(new_stride);
      },
      "ContiguousGradient");
    });
}

} // namespace impl
} // namespace hetu
