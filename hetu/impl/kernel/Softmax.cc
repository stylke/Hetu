#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"

namespace hetu {
namespace impl {

void SoftmaxCpu(const NDArray& input, NDArray& output, int64_t dim, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);

  dnnl::engine eng(dnnl::engine::kind::cpu, 0);
  dnnl::stream engine_stream(eng);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "SoftmaxCuda", [&]() {
      // Create src dnnl::memory descriptor and dnnl::memory object.
      auto src_md = dnnl::memory::desc(input->shape(), dnnl::memory::data_type::f32, input->stride());
      auto dst_md = dnnl::memory::desc(input->shape(), dnnl::memory::data_type::f32, input->stride());
      auto src_mem = dnnl::memory(src_md, eng);
      auto dst_mem = dnnl::memory(dst_md, eng);

      // Write data to dnnl::memory object's handle.
      hetu::omp::write_to_dnnl_memory(input->data_ptr<spec_t>(), src_mem);

      // Softmax axis.
      const int axis = dim;

      // Create primitive descriptor.
      auto softmax_pd = dnnl::softmax_forward::primitive_desc(eng,
                        dnnl::prop_kind::forward_training, 
                        dnnl::algorithm::softmax_accurate, 
                        src_md, dst_md, axis);

      // Create the primitive.
      auto softmax_prim = dnnl::softmax_forward(softmax_pd);

      // Primitive arguments. Set up in-place execution by assigning src as DST.
      std::unordered_map<int, dnnl::memory> softmax_args;
      softmax_args.insert({DNNL_ARG_SRC, src_mem});
      softmax_args.insert({DNNL_ARG_DST, dst_mem});

      // Primitive execution.
      softmax_prim.execute(engine_stream, softmax_args);

      // Wait for the computation to finalize.
      engine_stream.wait();

      // Read data from dnnl::memory object's handle.
      hetu::omp::read_from_dnnl_memory(output->data_ptr<spec_t>(), dst_mem);
    });
}

void SoftmaxGradientCpu(const NDArray& input_Y, const NDArray& output_grad,
                        NDArray& input_grad, int64_t dim, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input_Y);
  HT_ASSERT_SAME_DEVICE(input_Y, output_grad);
  HT_ASSERT_SAME_DEVICE(input_Y, input_grad);

  dnnl::engine eng(dnnl::engine::kind::cpu, 0);
  dnnl::stream engine_stream(eng);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input_Y->dtype(), spec_t, "SoftmaxGradientCuda", [&]() {
      // Create src dnnl::memory descriptor and dnnl::memory object.
      auto src_md = dnnl::memory::desc(input_Y->shape(), dnnl::memory::data_type::f32, input_Y->stride());
      auto dst_md = dnnl::memory::desc(input_Y->shape(), dnnl::memory::data_type::f32, input_Y->stride());
      auto dst_mem = dnnl::memory(dst_md, eng);
      auto gdst_mem = dnnl::memory(dst_md, eng);
      auto gsrc_mem = dnnl::memory(src_md, eng);

      // Write data to dnnl::memory object's handle.
      hetu::omp::write_to_dnnl_memory(input_Y->data_ptr<spec_t>(), dst_mem);
      hetu::omp::write_to_dnnl_memory(output_grad->data_ptr<spec_t>(), gdst_mem);

      // Softmax axis.
      const int axis = dim;

      // Create primitive descriptor.
      auto softmax_pd = dnnl::softmax_forward::primitive_desc(eng,
                        dnnl::prop_kind::forward_training, 
                        dnnl::algorithm::softmax_accurate, 
                        src_md, dst_md, axis);
      
      auto softmax_bwd_pd = dnnl::softmax_backward::primitive_desc(eng, dnnl::algorithm::softmax_accurate, 
                                                                   src_md, dst_md, dst_md, axis, softmax_pd);

      // Create the primitive.
      auto softmax_prim = dnnl::softmax_backward(softmax_bwd_pd);

      // Primitive arguments. Set up in-place execution by assigning src as DST.
      std::unordered_map<int, dnnl::memory> softmax_args;
      softmax_args.insert({DNNL_ARG_DIFF_SRC, gsrc_mem});
      softmax_args.insert({DNNL_ARG_DIFF_DST, gdst_mem});
      softmax_args.insert({DNNL_ARG_DST, dst_mem});

      // Primitive execution.
      softmax_prim.execute(engine_stream, softmax_args);

      // Wait for the computation to finalize.
      engine_stream.wait();

      // Read data from dnnl::memory object's handle.
      hetu::omp::read_from_dnnl_memory(input_grad->data_ptr<spec_t>(), gsrc_mem);
    });
}

} // namespace impl
} // namespace hetu
