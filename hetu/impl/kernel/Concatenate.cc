#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"

namespace hetu {
namespace impl {

template <typename spec_t>
void concatenate_cpu(const spec_t* input, spec_t* output, int input_width,
                     int output_width, int offset, int concat_size,
                     size_t size) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    int post_ind = idx % concat_size;
    int prev_ind = idx / concat_size;
    int mid_ind = prev_ind % input_width + offset;
    prev_ind = prev_ind / input_width;
    int out_ind = (prev_ind * output_width + mid_ind) * concat_size + post_ind;
    output[out_ind] = input[idx];
  }
}

template <typename spec_t>
void concatenate_gradient_cpu(const spec_t* output_grad, spec_t* input_grad,
                              int input_width, int output_width, int offset,
                              int concat_size, size_t size) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    int post_ind = idx % concat_size;
    int prev_ind = idx / concat_size;
    int mid_ind = prev_ind % input_width + offset;
    prev_ind = prev_ind / input_width;
    int out_ind = (prev_ind * output_width + mid_ind) * concat_size + post_ind;
    input_grad[idx] = output_grad[out_ind];
  }
}

void ConcatenateCpu(const NDArrayList& inputs, NDArray& output, size_t axis,
                    const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(output);
  for (int i = 0; i < inputs.size(); ++i)
    HT_ASSERT_SAME_DEVICE(inputs[i], output);

  // size_t size = output->numel();
  // size_t now_ndim = output->ndim();
  // HT_ASSERT(input->ndim() == now_ndim);
  // int num_concats = 1;
  // for (size_t i = 0; i < axis; ++i) {
  //   int cur_dim = output->shape(i);
  //   HT_ASSERT(input->shape(i) == cur_dim);
  //   num_concats *= cur_dim;
  // }
  // int concat_size = 1;
  // for (size_t i = axis + 1; i < now_ndim; ++i) {
  //   int cur_dim = output->shape(i);
  //   HT_ASSERT(input->shape(i) == cur_dim);
  //   concat_size *= cur_dim;
  // }
  // int input_width = input->shape(axis);
  // int output_width = output->shape(axis);
  // if (size == 0 || input_width == 0 || output_width == 0)
  //   return;
  dnnl::engine eng(dnnl::engine::kind::cpu, 0);
  dnnl::stream engine_stream(eng);  
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    inputs[0]->dtype(), spec_t, "ConcatenateCpu", [&]() {
      std::vector<dnnl::memory::desc> src_mds;
      std::vector<dnnl::memory> src_mems;

      for (int i = 0; i < inputs.size(); ++i) {
          auto md = dnnl::memory::desc(inputs[i]->shape(), dnnl::memory::data_type::f32, inputs[i]->stride());
          auto mem = dnnl::memory(md, eng);

          // Write data to memory object's handle.
          hetu::omp::write_to_dnnl_memory(inputs[i]->data_ptr<spec_t>(), mem);

          src_mds.push_back(md);
          src_mems.push_back(mem);
      }

      // Create primitive descriptor.
      auto concat_pd = dnnl::concat::primitive_desc(eng, axis, src_mds);

      // Create destination (dst) memory object using the memory descriptor
      // created by the primitive.
      auto dst_mem = dnnl::memory(concat_pd.dst_desc(), eng);

      // Create the primitive.
      auto concat_prim = dnnl::concat(concat_pd);

      // Primitive arguments.
      std::unordered_map<int, dnnl::memory> concat_args;
      for (int i = 0; i < inputs.size(); ++i)
          concat_args.insert({DNNL_ARG_MULTIPLE_SRC + i, src_mems[i]});
      concat_args.insert({DNNL_ARG_DST, dst_mem});

      // Primitive execution: concatenation.
      concat_prim.execute(engine_stream, concat_args);

      // Wait for the computation to finalize.
      engine_stream.wait();
      hetu::omp::read_from_dnnl_memory(output->data_ptr<spec_t>(), dst_mem);
    });
}

void ConcatenateGradientCpu(const NDArray& output_grad, NDArray& input_grad,
                            size_t axis, size_t offset, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(output_grad);
  HT_ASSERT_SAME_DEVICE(output_grad, input_grad);

  size_t size = input_grad->numel();
  size_t now_ndim = output_grad->ndim();
  HT_ASSERT(now_ndim == input_grad->ndim());
  int num_concats = 1;
  for (size_t i = 0; i < axis; ++i) {
    int cur_dim = output_grad->shape(i);
    HT_ASSERT(cur_dim == input_grad->shape(i));
    num_concats *= cur_dim;
  }
  int concat_size = 1;
  for (size_t i = axis + 1; i < now_ndim; ++i) {
    int cur_dim = output_grad->shape(i);
    HT_ASSERT(cur_dim == input_grad->shape(i));
    concat_size *= cur_dim;
  }
  int output_width = output_grad->shape(axis);
  int input_width = input_grad->shape(axis);
  if (size == 0 || input_width == 0 || output_width == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input_grad->dtype(), spec_t, "ConcatenateGradientCpu", [&]() {
      concatenate_gradient_cpu<spec_t>(
        output_grad->data_ptr<spec_t>(), input_grad->data_ptr<spec_t>(),
        input_width, output_width, offset, concat_size, size);
    });
}

} // namespace impl
} // namespace hetu
