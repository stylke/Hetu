#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/stream/CPUStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"

namespace hetu {
namespace impl {

void ReduceCpu(const NDArray& input, NDArray& output, const HTAxes& axes,
                ReductionType red_type, const Stream& stream) {
  dnnl::engine eng(dnnl::engine::kind::cpu, 0);
  dnnl::stream engine_stream(eng);
  HTAxes parsed_axes = NDArrayMeta::ParseAxes(axes, input->ndim());
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
  input->dtype(), spec_t, "ReduceCpu", [&]() {
    dnnl::memory::dims in_shape = input->shape();
    dnnl::memory::dims in_stride = input->stride();
    dnnl::memory::dims out_shape = input->shape();
    dnnl::memory::dims out_stride(input->ndim());
    for (int i = 0; i < parsed_axes.size(); ++i) {
      out_shape[parsed_axes[i]] = 1;
    }
    size_t stride_size = 1;
    for (int i = input->ndim() - 1; i >= 0; i--) {
      out_stride[i] = stride_size;
      stride_size *= out_shape[i];
    }
    auto src_md = dnnl::memory::desc(in_shape, dnnl::memory::data_type::f32, in_stride);
    auto dst_md = dnnl::memory::desc(out_shape, dnnl::memory::data_type::f32, out_stride);

    auto src_mem = dnnl::memory(src_md, eng);
    auto dst_mem = dnnl::memory(dst_md, eng);

    dnnl::algorithm algo;
    if (red_type == ReductionType::MEAN)
      algo = dnnl::algorithm::reduction_mean;
    else if (red_type == ReductionType::SUM)
      algo = dnnl::algorithm::reduction_sum;
    else
      HT_NOT_IMPLEMENTED << "Invalid reduction type.";

    // Write data to memory object's handle.
    hetu::omp::write_to_dnnl_memory(input->data_ptr<spec_t>(), src_mem);
    if (in_shape == out_shape)
      hetu::omp::read_from_dnnl_memory(output->data_ptr<spec_t>(), src_mem);
    else {

      // Create primitive descriptor.
      auto reduction_pd = dnnl::reduction::primitive_desc(
              eng, algo, src_md, dst_md, float(0.f), float(0.f));

      // Create the primitive.
      auto reduction_prim = dnnl::reduction(reduction_pd);

      // Primitive arguments.
      std::unordered_map<int, dnnl::memory> reduction_args;
      reduction_args.insert({DNNL_ARG_SRC, src_mem});
      reduction_args.insert({DNNL_ARG_DST, dst_mem});

      // Primitive execution: Reduction (Sum).
      reduction_prim.execute(engine_stream, reduction_args);

      // Wait for the computation to finalize.
      engine_stream.wait();

      // Read data from memory object's handle.
      hetu::omp::read_from_dnnl_memory(output->data_ptr<spec_t>(), dst_mem);
    }
  });
}

void ReduceMinCpu(const NDArray& input, NDArray& output, const int64_t* axes,
                  int64_t num_axes, const Stream& stream) {
  HTAxes m_axes(axes, axes + num_axes);
  ReduceCpu(input, output, m_axes, ReductionType::MIN, stream);
}

void ReduceMaxCpu(const NDArray& input, NDArray& output, const int64_t* axes,
                  int64_t num_axes, const Stream& stream) {
  HTAxes m_axes(axes, axes + num_axes);
  ReduceCpu(input, output, m_axes, ReductionType::MAX, stream);  
}

void ReduceMeanCpu(const NDArray& input, NDArray& output, const int64_t* axes,
                   int64_t num_axes, const Stream& stream) {
  HTAxes m_axes(axes, axes + num_axes);
  ReduceCpu(input, output, m_axes, ReductionType::MEAN, stream);
}

void ReduceSumCpu(const NDArray& input, NDArray& output, const int64_t* axes,
                  int64_t num_axes, const Stream& stream) {
  HTAxes m_axes(axes, axes + num_axes);
  ReduceCpu(input, output, m_axes, ReductionType::SUM, stream);
}

} // namespace impl
} // namespace hetu
