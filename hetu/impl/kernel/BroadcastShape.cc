#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include "hetu/impl/stream/CPUStream.h"

namespace hetu {
namespace impl {

template <typename spec_t>
void broadcast_shape_cpu(const spec_t* input, spec_t* output, uint* out_strides,
                         uint* in_dims, size_t ndims, size_t size) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    size_t i_ind = 0;
    size_t temp = idx;
    for (size_t i = 0; i < ndims; ++i) {
      i_ind *= in_dims[i];
      i_ind += (in_dims[i] > 1) * temp / out_strides[i];
      temp %= out_strides[i];
    }
    output[idx] = input[i_ind];
  }
}

void BroadcastShapeCpu(const NDArray& input, NDArray& output,
                       const HTShape& add_axes, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);

  CPUStream cpu_stream(stream);
  dnnl::engine eng(dnnl::engine::kind::cpu, 0);

  size_t size = output->numel();
  size_t input_size = input->numel();

  int input_dim = input->ndim();
  int output_dim = output->ndim();
  size_t allocated = output_dim * sizeof(uint);
  uint* out_strides = (uint*) malloc(allocated);
  uint* in_dims = (uint*) malloc(allocated);

  int64_t output_size = 1;
  size_t diff = output_dim - input_dim;

  if (add_axes.empty()) {
    for (int i = output_dim - 1; i >= 0; --i) {
      out_strides[i] = output_size;
      output_size *= output->shape(i);
      if ((size_t) i < diff) {
        in_dims[i] = 1;
      } else {
        in_dims[i] = input->shape(i - diff);
      }
    }
  } else {
    for (int i = output_dim - 1; i >= 0; --i) {
      out_strides[i] = output_size;
      output_size *= output->shape(i);
      in_dims[i] = 0;
    }
    for (size_t i = 0; i < diff; ++i) {
      in_dims[add_axes[i]] = 1;
    }
    int o_ind = 0;
    for (size_t i = 0; i < input->ndim(); ++i) {
      while (in_dims[o_ind++] == 1)
        ;
      in_dims[o_ind - 1] = input->shape(i);
    }
  }

  if (size == 0 || input_size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "BroadcastShapeCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [input, output, out_strides, in_dims, output_dim, size]() {
      broadcast_shape_cpu<spec_t>(input->data_ptr<spec_t>(),
                                  output->data_ptr<spec_t>(), out_strides,
                                  in_dims, output_dim, size);
      free(out_strides);
      free(in_dims);
      },
      "BroadcastShape");
      //cpu_stream.Sync();
    });
}

template <typename spec_t>
void broadcast_shape_mul_cpu(const spec_t* input, spec_t const_value,
                             spec_t* output, uint* out_strides, uint* in_dims,
                             size_t ndims, size_t size) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    size_t i_ind = 0;
    size_t temp = idx;
    for (size_t i = 0; i < ndims; ++i) {
      i_ind *= in_dims[i];
      i_ind += (in_dims[i] > 1) * temp / out_strides[i];
      temp %= out_strides[i];
    }
    output[idx] = input[i_ind] * const_value;
  }
}

void BroadcastShapeMulCpu(const NDArray& input, double const_value,
                          NDArray& output, const HTShape& add_axes,
                          const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);

  CPUStream cpu_stream(stream);
  dnnl::engine eng(dnnl::engine::kind::cpu, 0);

  size_t size = output->numel();
  size_t input_size = input->numel();

  int input_dim = input->ndim();
  int output_dim = output->ndim();
  size_t allocated = output_dim * sizeof(uint);
  uint* out_strides = (uint*) malloc(allocated);
  uint* in_dims = (uint*) malloc(allocated);

  size_t output_size = 1;
  int diff = output_dim - input_dim;

  if (add_axes.empty()) {
    for (int i = output_dim - 1; i >= 0; --i) {
      out_strides[i] = output_size;
      output_size *= output->shape(i);
      if (i < diff) {
        in_dims[i] = 1;
      } else {
        in_dims[i] = input->shape(i - diff);
      }
    }
  } else {
    for (int i = output_dim - 1; i >= 0; --i) {
      out_strides[i] = output_size;
      output_size *= output->shape(i);
      in_dims[i] = 0;
    }
    for (int i = 0; i < diff; ++i) {
      in_dims[add_axes[i]] = 1;
    }
    int o_ind = 0;
    for (int i = 0; i < (int) input->ndim(); ++i) {
      while (in_dims[o_ind++] == 1)
        ;
      in_dims[o_ind - 1] = input->shape(i);
    }
  }

  if (size == 0 || input_size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "BroadcastShapeMulCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [input, output, out_strides, const_value, in_dims, output_dim, size]() {
      broadcast_shape_mul_cpu<spec_t>(
        input->data_ptr<spec_t>(), static_cast<spec_t>(const_value),
        output->data_ptr<spec_t>(), out_strides, in_dims, output_dim, size);
      free(out_strides);
      free(in_dims);
      },
      "BroadcastShapeMul");
      //cpu_stream.Sync();
    });
}

} // namespace impl
} // namespace hetu
