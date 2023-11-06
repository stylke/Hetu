#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include "hetu/impl/stream/CPUStream.h"

namespace hetu {
namespace impl {

template <typename spec_t>
void slice_cpu(const spec_t* input, spec_t* output, const int64_t* output_shape,
               const int64_t* input_shape, const int64_t* begin_pos,
               size_t ndim, size_t size) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    size_t tmp_index = idx;
    size_t i_index = 0;
    int64_t i_mat = 1;
    for (int i = ndim - 1; i >= 0; --i) {
      int64_t offset = begin_pos[i] + tmp_index % output_shape[i];
      tmp_index /= output_shape[i];
      i_index += offset * i_mat;
      i_mat *= input_shape[i];
    }
    output[idx] = input[i_index];
  }
}

template <typename spec_t>
void slice_gradient_cpu(const spec_t* input, spec_t* output,
                        const int64_t* output_shape, const int64_t* input_shape,
                        const int64_t* begin_pos, size_t ndim, size_t size) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    output[idx] = 0;
    size_t tmp_index = idx;
    size_t i_index = 0;
    int64_t i_mat = 1;
    for (int i = ndim - 1; i >= 0; --i) {
      int64_t offset = tmp_index % output_shape[i];
      if (offset < begin_pos[i] || offset >= begin_pos[i] + input_shape[i]) {}
      tmp_index /= output_shape[i];
      i_index += (offset - begin_pos[i]) * i_mat;
      i_mat *= input_shape[i];
    }
    output[idx] = input[i_index];
  }
}

void SliceCpu(const NDArray& input, NDArray& output, int64_t* begin_pos,
              const Stream& stream) {
  HT_ASSERT(input->is_cpu()) << "Input is not on a host device.";
  HT_ASSERT(output->is_cpu()) << "Output is not on a host device.";
  HT_ASSERT(input->device() == output->device())
    << "input and output are not on the same host device. "
    << "Devices: (input) " << input->device() << " vs. (output) "
    << output->device();
  HT_ASSERT(input->ndim() == output->ndim())
    << "input and output has different dims. ";

  CPUStream cpu_stream(stream);

  size_t ndim = input->ndim();
  size_t o_size = 1;
  for (size_t i = 0; i < ndim; ++i) {
    HT_ASSERT(begin_pos[i] >= 0);
    HT_ASSERT(begin_pos[i] + output->shape(i) <= input->shape(i));
    o_size *= output->shape(i);
  }
  size_t alloc_size = ndim * sizeof(int64_t);
  void* pos = malloc(alloc_size);
  void* i_shape = malloc(alloc_size);
  void* o_shape = malloc(alloc_size);
  size_t size = o_size;
  if (size == 0)
    return;
  memcpy(pos, (void*) begin_pos, alloc_size);
  memcpy(i_shape, (void*) input->shape().data(), alloc_size);
  memcpy(o_shape, (void*) output->shape().data(), alloc_size);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "SliceCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [input, output, o_shape, i_shape, pos, ndim, size]() {
      slice_cpu<spec_t>(input->data_ptr<spec_t>(), output->data_ptr<spec_t>(),
                        (const int64_t*) o_shape, (const int64_t*) i_shape,
                        (const int64_t*) pos, ndim, size);
      free(o_shape);
      free(i_shape);
      free(pos);
      }, "Slice"); 
    });
}

void SliceGradientCpu(const NDArray& output_grad, NDArray& input_grad,
                      int64_t* begin_pos, const Stream& stream) {
  HT_ASSERT(output_grad->is_cpu()) << "Output_grad is not on a host device.";
  HT_ASSERT(input_grad->is_cpu()) << "Input_grad is not on a host device.";
  HT_ASSERT(input_grad->device() == output_grad->device())
    << "input_grad and output_grad are not on the same host device. "
    << "Devices: (input_grad) " << input_grad->device() << " vs. (output_grad) "
    << output_grad->device();
  HT_ASSERT(input_grad->ndim() == output_grad->ndim())
    << "input and output grad has different dims. ";

  CPUStream cpu_stream(stream);
  
  size_t ndim = output_grad->ndim();
  size_t o_size = 1;
  for (size_t i = 0; i < ndim; ++i) {
    HT_ASSERT(begin_pos[i] >= 0);
    HT_ASSERT(begin_pos[i] + output_grad->shape(i) <= input_grad->shape(i));
    o_size *= input_grad->shape(i);
  }
  size_t alloc_size = ndim * sizeof(int64_t);
  void* pos = malloc(alloc_size);
  void* i_shape = malloc(alloc_size);
  void* o_shape = malloc(alloc_size);
  size_t size = input_grad->numel();
  if (size == 0)
    return;
  memcpy(pos, (void*) begin_pos, alloc_size);
  memcpy(i_shape, (void*) output_grad->shape().data(), alloc_size);
  memcpy(o_shape, (void*) input_grad->shape().data(), alloc_size);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    output_grad->dtype(), spec_t, "SliceGradientCuda", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [input_grad, output_grad, o_shape, i_shape, pos, ndim, size]() {
      slice_gradient_cpu<spec_t>(
        output_grad->data_ptr<spec_t>(), input_grad->data_ptr<spec_t>(),
        (const int64_t*) o_shape, (const int64_t*) i_shape,
        (const int64_t*) pos, ndim, size);
      free(o_shape);
      free(i_shape);
      free(pos);
      }, "SliceGradient");
    });
}

} // namespace impl
} // namespace hetu
