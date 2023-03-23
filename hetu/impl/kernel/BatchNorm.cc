#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"

namespace hetu {
namespace impl {
void BatchNormCpu(const NDArray& input_X, const NDArray& bn_scale,
                  const NDArray& bn_bias, NDArray& output_Y, double momentum,
                  double eps, NDArray& running_mean, NDArray& running_var,
                  NDArray& save_mean, NDArray& save_var,
                  const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input_X);
  HT_ASSERT_SAME_DEVICE(input_X, bn_scale);
  HT_ASSERT_SAME_DEVICE(input_X, bn_bias);
  HT_ASSERT_SAME_DEVICE(input_X, output_Y);
  HT_ASSERT_SAME_DEVICE(input_X, running_mean);
  HT_ASSERT_SAME_DEVICE(input_X, running_var);
  HT_ASSERT_SAME_DEVICE(input_X, save_mean);
  HT_ASSERT_SAME_DEVICE(input_X, save_var);

  dnnl::engine eng(dnnl::engine::kind::cpu, 0);
  dnnl::stream engine_stream(eng);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input_X->dtype(), spec_t, "BatchNormCuda", [&]() {
      auto src_md = dnnl::memory::desc(input_X->shape(), dnnl::memory::data_type::f32, input_X->stride());
      auto dst_md = dnnl::memory::desc(output_Y->shape(), dnnl::memory::data_type::f32, output_Y->stride());
      auto scaleshift_md = dnnl::memory::desc(bn_bias->shape(), dnnl::memory::data_type::f32, dnnl::memory::format_tag::x);

      auto src_mem = dnnl::memory(src_md, eng);
      auto dst_mem = dnnl::memory(dst_md, eng);
      auto scale_mem = dnnl::memory(scaleshift_md, eng);
      auto shift_mem = dnnl::memory(scaleshift_md, eng);

      // Write data to memory object's handle.
      hetu::omp::write_to_dnnl_memory(input_X->data_ptr<spec_t>(), src_mem);
      hetu::omp::write_to_dnnl_memory(bn_scale->data_ptr<spec_t>(), scale_mem);
      hetu::omp::write_to_dnnl_memory(bn_bias->data_ptr<spec_t>(), shift_mem);

      // Create primitive descriptor.
      auto bnorm_pd = dnnl::batch_normalization_forward::primitive_desc(eng,
              dnnl::prop_kind::forward_training, src_md, dst_md, float(eps),
              dnnl::normalization_flags::use_scale | dnnl::normalization_flags::use_shift);

      // Create memory objects using memory descriptors created by the primitive
      // descriptor: mean, variance, workspace.
      // NOTE: Here, the ReLU post-ops require a workspace for later usage in
      // backward propagation mode.
      auto mean_mem = dnnl::memory(bnorm_pd.mean_desc(), eng);
      auto variance_mem = dnnl::memory(bnorm_pd.variance_desc(), eng);
      auto workspace_mem = dnnl::memory(bnorm_pd.workspace_desc(), eng);

      // Create the primitive.
      auto bnorm_prim = dnnl::batch_normalization_forward(bnorm_pd);

      // Primitive arguments. Set up in-place execution by assigning src as DST.
      std::unordered_map<int, dnnl::memory> bnorm_args;
      bnorm_args.insert({DNNL_ARG_SRC, src_mem});
      bnorm_args.insert({DNNL_ARG_MEAN, mean_mem});
      bnorm_args.insert({DNNL_ARG_VARIANCE, variance_mem});
      bnorm_args.insert({DNNL_ARG_SCALE, scale_mem});
      bnorm_args.insert({DNNL_ARG_SHIFT, shift_mem});
      bnorm_args.insert({DNNL_ARG_WORKSPACE, workspace_mem});
      bnorm_args.insert({DNNL_ARG_DST, dst_mem});

      // Primitive execution: batch normalization with ReLU.
      bnorm_prim.execute(engine_stream, bnorm_args);

      // Wait for the computation to finalize.
      engine_stream.wait();

      // Read data from memory object's handle.
      hetu::omp::read_from_dnnl_memory(output_Y->data_ptr<spec_t>(), dst_mem);
      hetu::omp::read_from_dnnl_memory(save_mean->data_ptr<spec_t>(), mean_mem);
      hetu::omp::read_from_dnnl_memory(save_var->data_ptr<spec_t>(), variance_mem);
    });
  return;
}

void BatchNormGradientCpu(const NDArray& gradient_Y, const NDArray& input_X,
                          const NDArray& bn_scale, NDArray& gradient_X,
                          NDArray& gradient_bn_scale,
                          NDArray& gradient_bn_bias, double eps,
                          NDArray& save_mean, NDArray& save_var,
                          const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(gradient_Y);
  HT_ASSERT_SAME_DEVICE(gradient_Y, input_X);
  HT_ASSERT_SAME_DEVICE(gradient_Y, bn_scale);
  HT_ASSERT_SAME_DEVICE(gradient_Y, gradient_X);
  HT_ASSERT_SAME_DEVICE(gradient_Y, gradient_bn_scale);
  HT_ASSERT_SAME_DEVICE(gradient_Y, gradient_bn_bias);
  HT_ASSERT_SAME_DEVICE(gradient_Y, save_mean);
  HT_ASSERT_SAME_DEVICE(gradient_Y, save_var);

  dnnl::engine eng(dnnl::engine::kind::cpu, 0);
  dnnl::stream engine_stream(eng);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input_X->dtype(), spec_t, "BatchNormGradientCpu", [&]() {
      auto src_md = dnnl::memory::desc(input_X->shape(), dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw);
      auto gdst_md = dnnl::memory::desc(gradient_Y->shape(), dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw);
      auto scaleshift_md = dnnl::memory::desc(bn_scale->shape(), dnnl::memory::data_type::f32, dnnl::memory::format_tag::x);
      auto mean_md = dnnl::memory::desc(save_mean->shape(), dnnl::memory::data_type::f32, save_mean->stride());


      auto src_mem = dnnl::memory(src_md, eng);
      auto gsrc_mem = dnnl::memory(src_md, eng);
      auto gdst_mem = dnnl::memory(gdst_md, eng);
      auto mean_mem = dnnl::memory(mean_md, eng);
      auto variance_mem = dnnl::memory(mean_md, eng);
      auto scale_mem = dnnl::memory(scaleshift_md, eng);
      auto gscale_mem = dnnl::memory(scaleshift_md, eng);
      auto gbias_mem = dnnl::memory(scaleshift_md, eng);

      // Write data to memory object's handle.
      hetu::omp::write_to_dnnl_memory(input_X->data_ptr<spec_t>(), src_mem);
      hetu::omp::write_to_dnnl_memory(bn_scale->data_ptr<spec_t>(), scale_mem);
      hetu::omp::write_to_dnnl_memory(gradient_Y->data_ptr<spec_t>(), gdst_mem);
      hetu::omp::write_to_dnnl_memory(save_mean->data_ptr<spec_t>(), mean_mem);
      hetu::omp::write_to_dnnl_memory(save_var->data_ptr<spec_t>(), variance_mem);

      // Create primitive descriptor.
      auto bnorm_pd = dnnl::batch_normalization_forward::primitive_desc(eng,
              dnnl::prop_kind::forward_training, src_md, gdst_md, float(eps),
              dnnl::normalization_flags::use_scale | dnnl::normalization_flags::use_shift);

      auto bnorm_bwd_pd = dnnl::batch_normalization_backward::primitive_desc(eng,
              dnnl::prop_kind::backward, src_md, gdst_md, src_md, float(eps),
              dnnl::normalization_flags::use_scale | dnnl::normalization_flags::use_shift, bnorm_pd);
      

      // Create memory objects using memory descriptors created by the primitive
      // descriptor: mean, variance, workspace.
      // NOTE: Here, the ReLU post-ops require a workspace for later usage in
      // backward propagation mode.
      // auto mean_mem = dnnl::memory(bnorm_pd.mean_desc(), eng);
      // auto variance_mem = dnnl::memory(bnorm_pd.variance_desc(), eng);
      auto workspace_mem = dnnl::memory(bnorm_bwd_pd.workspace_desc(), eng);

      // Create the primitive.
      auto bnorm_prim = dnnl::batch_normalization_backward(bnorm_bwd_pd);

      // Primitive arguments. Set up in-place execution by assigning src as DST.
      std::unordered_map<int, dnnl::memory> bnorm_args;
      bnorm_args.insert({DNNL_ARG_SRC, src_mem});
      bnorm_args.insert({DNNL_ARG_MEAN, mean_mem});
      bnorm_args.insert({DNNL_ARG_VARIANCE, variance_mem});
      bnorm_args.insert({DNNL_ARG_SCALE, scale_mem});
      bnorm_args.insert({DNNL_ARG_DIFF_SCALE, gscale_mem});
      bnorm_args.insert({DNNL_ARG_DIFF_SHIFT, gbias_mem});
      bnorm_args.insert({DNNL_ARG_WORKSPACE, workspace_mem});
      bnorm_args.insert({DNNL_ARG_DIFF_DST, gdst_mem});
      bnorm_args.insert({DNNL_ARG_DIFF_SRC, gsrc_mem});

      // Primitive execution: batch normalization with ReLU.
      bnorm_prim.execute(engine_stream, bnorm_args);

      // Wait for the computation to finalize.
      engine_stream.wait();

      // Read data from memory object's handle.
      hetu::omp::read_from_dnnl_memory(gradient_X->data_ptr<spec_t>(), gsrc_mem);
      hetu::omp::read_from_dnnl_memory(gradient_bn_scale->data_ptr<spec_t>(), gscale_mem);
      hetu::omp::read_from_dnnl_memory(gradient_bn_bias->data_ptr<spec_t>(), gbias_mem);
    });
}    
} // namespace impl
} // namespace hetu
