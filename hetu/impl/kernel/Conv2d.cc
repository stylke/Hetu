#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"

namespace hetu {
namespace impl {
void Conv2dCpu(const NDArray& input_x, const NDArray& input_f, NDArray& output,
               const int padding_h, const int padding_w, const int stride_h,
               const int stride_w, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input_x);
  HT_ASSERT_SAME_DEVICE(input_x, input_f);
  HT_ASSERT_SAME_DEVICE(input_x, output);

  dnnl::engine eng(dnnl::engine::kind::cpu, 0);
  dnnl::stream engine_stream(eng);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input_x->dtype(), spec_t, "Conv2dCpu", [&]() {
      // Create dnnl::memory descriptors with format_tag::any for the primitive. This
      // enables the convolution primitive to choose dnnl::memory layouts for an
      // optimized primitive implementation, and these layouts may differ from the
      // ones provided by the user.
      auto conv_src_md = dnnl::memory::desc(input_x->shape(), dnnl::memory::data_type::f32, 
                                            dnnl::memory::format_tag::nchw);
      auto conv_weights_md = dnnl::memory::desc(input_f->shape(), dnnl::memory::data_type::f32, 
                                                dnnl::memory::format_tag::oihw);
      auto conv_dst_md = dnnl::memory::desc(output->shape(), dnnl::memory::data_type::f32,
                                            dnnl::memory::format_tag::nchw);
      // auto user_bias_md = dnnl::memory::desc(bias_dims, dt::f32, tag::a);
      // auto user_bias_mem = dnnl::memory(user_bias_md, engine);

      auto conv_src_mem = dnnl::memory(conv_src_md, eng);
      auto conv_weights_mem = dnnl::memory(conv_weights_md, eng);
      auto conv_dst_mem = dnnl::memory(conv_dst_md, eng);

      // Create dnnl::memory descriptor and dnnl::memory object for input bias.

      // Write data to dnnl::memory object's handle.
      hetu::omp::write_to_dnnl_memory(input_x->data_ptr<spec_t>(), conv_src_mem);
      hetu::omp::write_to_dnnl_memory(input_f->data_ptr<spec_t>(), conv_weights_mem);
      // hetu::omp::write_to_dnnl_memory(bias_data.data(), user_bias_mem);

      // Create primitive post-ops (ReLU).
      const float alpha = 0.f;
      const float beta = 0.f;

      dnnl::memory::dims strides_dims = {int(stride_h), int(stride_w)};
      dnnl::memory::dims padding_dims_l = {int(padding_h), int(padding_w)};
      dnnl::memory::dims padding_dims_r = {int(padding_h), int(padding_w)};

      // Create primitive descriptor.
      auto conv_pd = dnnl::convolution_forward::primitive_desc(eng,
              dnnl::prop_kind::forward_training, dnnl::algorithm::convolution_direct,
              conv_src_md, conv_weights_md, conv_dst_md,
              strides_dims, padding_dims_l, padding_dims_r);

      // Create the primitive.
      auto conv_prim = dnnl::convolution_forward(conv_pd);

      // Primitive arguments.
      std::unordered_map<int, dnnl::memory> conv_args;
      conv_args.insert({DNNL_ARG_SRC, conv_src_mem});
      conv_args.insert({DNNL_ARG_WEIGHTS, conv_weights_mem});
      // conv_args.insert({DNNL_ARG_BIAS, user_bias_mem});
      conv_args.insert({DNNL_ARG_DST, conv_dst_mem});

      // Primitive execution: convolution with ReLU.
      conv_prim.execute(engine_stream, conv_args);

      // Wait for the computation to finalize.
      engine_stream.wait();

      // Read data from dnnl::memory object's handle.
      hetu::omp::read_from_dnnl_memory(output->data_ptr<spec_t>(), conv_dst_mem);      
    });
  return;
}

void Conv2dGradientofFilterCpu(const NDArray& input_x,
                               const NDArray& gradient_y, NDArray& gradient_f,
                               const int padding_h, const int padding_w,
                               const int stride_h, const int stride_w,
                               const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input_x);
  HT_ASSERT_SAME_DEVICE(input_x, gradient_y);
  HT_ASSERT_SAME_DEVICE(input_x, gradient_f);

  dnnl::engine eng(dnnl::engine::kind::cpu, 0);
  dnnl::stream engine_stream(eng);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input_x->dtype(), spec_t, "Conv2dGradientofFilterCpu", [&]() {
      auto conv_src_md = dnnl::memory::desc(input_x->shape(), dnnl::memory::data_type::f32, 
                                            dnnl::memory::format_tag::nchw);
      auto conv_weights_md = dnnl::memory::desc(gradient_f->shape(), dnnl::memory::data_type::f32, 
                                                dnnl::memory::format_tag::oihw);
      auto conv_dst_md = dnnl::memory::desc(gradient_y->shape(), dnnl::memory::data_type::f32,
                                            dnnl::memory::format_tag::nchw);

      auto conv_src_mem = dnnl::memory(conv_src_md, eng);
      auto conv_weights_mem = dnnl::memory(conv_weights_md, eng);
      auto conv_dst_mem = dnnl::memory(conv_dst_md, eng);

      // Create dnnl::memory descriptor and dnnl::memory object for input bias.

      // Write data to dnnl::memory object's handle.
      hetu::omp::write_to_dnnl_memory(input_x->data_ptr<spec_t>(), conv_src_mem);
      hetu::omp::write_to_dnnl_memory(gradient_y->data_ptr<spec_t>(), conv_dst_mem);
      // hetu::omp::write_to_dnnl_memory(bias_data.data(), user_bias_mem);

      // Create primitive post-ops (ReLU).
      const float alpha = 0.f;
      const float beta = 0.f;

      dnnl::memory::dims strides_dims = {int(stride_h), int(stride_w)};
      dnnl::memory::dims padding_dims_l = {int(padding_h), int(padding_w)};
      dnnl::memory::dims padding_dims_r = {int(padding_h), int(padding_w)};

      // Create primitive descriptor.
      auto conv_pd = dnnl::convolution_forward::primitive_desc(eng,
              dnnl::prop_kind::forward_training, dnnl::algorithm::convolution_direct,
              conv_src_md, conv_weights_md, conv_dst_md,
              strides_dims, padding_dims_l, padding_dims_r);
      
      auto conv_bwd_pd = dnnl::convolution_backward_weights::primitive_desc(eng,
              dnnl::algorithm::convolution_direct,
              conv_src_md, conv_dst_md, conv_weights_md,
              strides_dims, padding_dims_l, padding_dims_r, conv_pd);

      // Create the primitive.
      auto conv_prim = dnnl::convolution_backward_weights(conv_bwd_pd);

      // Primitive arguments.
      std::unordered_map<int, dnnl::memory> conv_args;
      conv_args.insert({DNNL_ARG_SRC, conv_src_mem});
      conv_args.insert({DNNL_ARG_DIFF_WEIGHTS, conv_weights_mem});
      conv_args.insert({DNNL_ARG_DIFF_DST, conv_dst_mem});

      // Primitive execution: convolution with ReLU.
      conv_prim.execute(engine_stream, conv_args);

      // Wait for the computation to finalize.
      engine_stream.wait();

      // Read data from dnnl::memory object's handle.
      hetu::omp::read_from_dnnl_memory(gradient_f->data_ptr<spec_t>(), conv_dst_mem);           
    });
  return;
}

void Conv2dGradientofDataCpu(const NDArray& input_f, const NDArray& gradient_y,
                             NDArray& gradient_x, const int padding_h,
                             const int padding_w, const int stride_h,
                             const int stride_w, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input_f);
  HT_ASSERT_SAME_DEVICE(input_f, gradient_y);
  HT_ASSERT_SAME_DEVICE(input_f, gradient_x);

  dnnl::engine eng(dnnl::engine::kind::cpu, 0);
  dnnl::stream engine_stream(eng);  
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input_f->dtype(), spec_t, "Conv2dGradientofDataCpu", [&]() {
      auto conv_src_md = dnnl::memory::desc(gradient_x->shape(), dnnl::memory::data_type::f32, 
                                            dnnl::memory::format_tag::nchw);
      auto conv_weights_md = dnnl::memory::desc(input_f->shape(), dnnl::memory::data_type::f32, 
                                                dnnl::memory::format_tag::oihw);
      auto conv_dst_md = dnnl::memory::desc(gradient_y->shape(), dnnl::memory::data_type::f32,
                                            dnnl::memory::format_tag::nchw);

      auto conv_src_mem = dnnl::memory(conv_src_md, eng);
      auto conv_weights_mem = dnnl::memory(conv_weights_md, eng);
      auto conv_dst_mem = dnnl::memory(conv_dst_md, eng);

      // Create dnnl::memory descriptor and dnnl::memory object for input bias.

      // Write data to dnnl::memory object's handle.
      hetu::omp::write_to_dnnl_memory(input_f->data_ptr<spec_t>(), conv_weights_mem);
      hetu::omp::write_to_dnnl_memory(gradient_y->data_ptr<spec_t>(), conv_dst_mem);
      // hetu::omp::write_to_dnnl_memory(bias_data.data(), user_bias_mem);

      // Create primitive post-ops (ReLU).
      const float alpha = 0.f;
      const float beta = 0.f;

      dnnl::memory::dims strides_dims = {int(stride_h), int(stride_w)};
      dnnl::memory::dims padding_dims_l = {int(padding_h), int(padding_w)};
      dnnl::memory::dims padding_dims_r = {int(padding_h), int(padding_w)};

      // Create primitive descriptor.
      auto conv_pd = dnnl::convolution_forward::primitive_desc(eng,
              dnnl::prop_kind::forward_training, dnnl::algorithm::convolution_direct,
              conv_src_md, conv_weights_md, conv_dst_md,
              strides_dims, padding_dims_l, padding_dims_r);
      
      auto conv_bwd_pd = dnnl::convolution_backward_data::primitive_desc(eng,
              dnnl::algorithm::convolution_direct,
              conv_src_md, conv_weights_md, conv_dst_md,
              strides_dims, padding_dims_l, padding_dims_r, conv_pd);

      // Create the primitive.
      auto conv_prim = dnnl::convolution_backward_data(conv_bwd_pd);

      // Primitive arguments.
      std::unordered_map<int, dnnl::memory> conv_args;
      conv_args.insert({DNNL_ARG_DIFF_SRC, conv_src_mem});
      conv_args.insert({DNNL_ARG_WEIGHTS, conv_weights_mem});
      conv_args.insert({DNNL_ARG_DIFF_DST, conv_dst_mem});

      // Primitive execution: convolution with ReLU.
      conv_prim.execute(engine_stream, conv_args);

      // Wait for the computation to finalize.
      engine_stream.wait();

      // Read data from dnnl::memory object's handle.
      hetu::omp::read_from_dnnl_memory(gradient_x->data_ptr<spec_t>(), conv_src_mem); 
    });
  return;
}

void Conv2dAddBiasCpu(const NDArray& input_x, const NDArray& input_f,
                      const NDArray& bias, NDArray& output,
                      const int padding_h, const int padding_w,
                      const int stride_h, const int stride_w,
                      const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input_x);
  HT_ASSERT_SAME_DEVICE(input_x, input_f);
  HT_ASSERT_SAME_DEVICE(input_x, bias);
  HT_ASSERT_SAME_DEVICE(input_x, output);

  dnnl::engine eng(dnnl::engine::kind::cpu, 0);
  dnnl::stream engine_stream(eng);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input_x->dtype(), spec_t, "Conv2dAddBiasCpu", [&]() {
      auto conv_src_md = dnnl::memory::desc(input_x->shape(), dnnl::memory::data_type::f32, 
                                            dnnl::memory::format_tag::nchw);
      auto conv_weights_md = dnnl::memory::desc(input_f->shape(), dnnl::memory::data_type::f32, 
                                                dnnl::memory::format_tag::oihw);
      auto conv_dst_md = dnnl::memory::desc(output->shape(), dnnl::memory::data_type::f32,
                                            dnnl::memory::format_tag::nchw);
      auto conv_bias_md = dnnl::memory::desc(bias->shape(), dnnl::memory::data_type::f32, 
                                             dnnl::memory::format_tag::a);

      auto conv_src_mem = dnnl::memory(conv_src_md, eng);
      auto conv_weights_mem = dnnl::memory(conv_weights_md, eng);
      auto conv_dst_mem = dnnl::memory(conv_dst_md, eng);
      auto conv_bias_mem = dnnl::memory(conv_bias_md, eng);

      // Create dnnl::memory descriptor and dnnl::memory object for input bias.

      // Write data to dnnl::memory object's handle.
      hetu::omp::write_to_dnnl_memory(input_x->data_ptr<spec_t>(), conv_src_mem);
      hetu::omp::write_to_dnnl_memory(input_f->data_ptr<spec_t>(), conv_weights_mem);
      hetu::omp::write_to_dnnl_memory(bias->data_ptr<spec_t>(), conv_bias_mem);

      // Create primitive post-ops (ReLU).
      const float alpha = 0.f;
      const float beta = 0.f;

      dnnl::memory::dims strides_dims = {int(stride_h), int(stride_w)};
      dnnl::memory::dims padding_dims_l = {int(padding_h), int(padding_w)};
      dnnl::memory::dims padding_dims_r = {int(padding_h), int(padding_w)};

      // Create primitive descriptor.
      auto conv_pd = dnnl::convolution_forward::primitive_desc(eng,
              dnnl::prop_kind::forward_training, dnnl::algorithm::convolution_direct,
              conv_src_md, conv_weights_md, conv_bias_md, conv_dst_md,
              strides_dims, padding_dims_l, padding_dims_r);

      // Create the primitive.
      auto conv_prim = dnnl::convolution_forward(conv_pd);

      // Primitive arguments.
      std::unordered_map<int, dnnl::memory> conv_args;
      conv_args.insert({DNNL_ARG_SRC, conv_src_mem});
      conv_args.insert({DNNL_ARG_WEIGHTS, conv_weights_mem});
      conv_args.insert({DNNL_ARG_BIAS, conv_bias_mem});
      conv_args.insert({DNNL_ARG_DST, conv_dst_mem});

      // Primitive execution: convolution with ReLU.
      conv_prim.execute(engine_stream, conv_args);

      // Wait for the computation to finalize.
      engine_stream.wait();

      // Read data from dnnl::memory object's handle.
      hetu::omp::read_from_dnnl_memory(output->data_ptr<spec_t>(), conv_dst_mem);  
    });
  return;
}
} // namespace impl
} // namespace hetu
