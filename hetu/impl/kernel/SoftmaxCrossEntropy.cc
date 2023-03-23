#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"

namespace hetu {
namespace impl {
template <typename spec_t>
void softmax_cross_entropy_cpu(const spec_t* logsoftmax,
                               const spec_t* label,
                               spec_t* output, size_t size) {
  for (size_t idx = 0; idx < size; ++idx) 
    output[idx] = -logsoftmax[idx] * label[idx];
}

void SoftmaxCrossEntropyCpu(const NDArray& input, const NDArray& label,
                            NDArray& output, const Stream& stream) {
  size_t indim = input->ndim();
  HT_ASSERT(indim == label->ndim() && indim == output->ndim() + 1)
    << "Indim is " << indim << ", Label dim is " << label->ndim()
    << ", Output dim is " << output->ndim();
  int n_ = 1;
  for (int i = 0; i < indim - 1; ++i) {
    n_ *= input->shape(i);
  }
  int c_ = input->shape(indim - 1);
  size_t size = n_ * c_;

  if (size == 0)
    return;

  dnnl::engine eng(dnnl::engine::kind::cpu, 0);
  dnnl::stream engine_stream(eng);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "SoftmaxCrossEntropyCuda", [&]() {
      DataPtr temp_data_ptr = AllocFromMemoryPool(input->device(), size * sizeof(spec_t));
      void* temp_data = temp_data_ptr.ptr;

      auto src_md = dnnl::memory::desc(input->shape(), dnnl::memory::data_type::f32, input->stride());
      auto dst_md = dnnl::memory::desc(input->shape(), dnnl::memory::data_type::f32, input->stride());
      auto src_mem = dnnl::memory(src_md, eng);
      auto dst_mem = dnnl::memory(dst_md, eng);

      hetu::omp::write_to_dnnl_memory(input->data_ptr<spec_t>(), src_mem);

      // Softmax axis.
      const int axis = 1;
      auto softmax_pd = dnnl::softmax_forward::primitive_desc(eng,
                        dnnl::prop_kind::forward_training, 
                        dnnl::algorithm::softmax_log, 
                        src_md, dst_md, axis);

      auto softmax_prim = dnnl::softmax_forward(softmax_pd);

      std::unordered_map<int, dnnl::memory> softmax_args;
      softmax_args.insert({DNNL_ARG_SRC, src_mem});
      softmax_args.insert({DNNL_ARG_DST, dst_mem});

      // Primitive execution.
      softmax_prim.execute(engine_stream, softmax_args);

      // Wait for the computation to finalize.
      engine_stream.wait();

      // Read data from dnnl::memory object's handle.
      hetu::omp::read_from_dnnl_memory(temp_data, dst_mem);


      softmax_cross_entropy_cpu<spec_t>(
        (const spec_t*) temp_data, label->data_ptr<spec_t>(),
        (spec_t*) temp_data, size);

      HTShape outshape = output->shape(); outshape.emplace_back(1);
      HTShape outstride = output->stride(); outstride.emplace_back(1);
      auto rsrc_md = dnnl::memory::desc(input->shape(), dnnl::memory::data_type::f32, input->stride());
      auto rdst_md = dnnl::memory::desc(outshape, dnnl::memory::data_type::f32, outstride);

      auto rsrc_mem = dnnl::memory(rsrc_md, eng);
      auto rdst_mem = dnnl::memory(rdst_md, eng);

      // Write data to memory object's handle.
      hetu::omp::write_to_dnnl_memory(temp_data, rsrc_mem);
      if (input->shape() == outshape)
        hetu::omp::read_from_dnnl_memory(output->data_ptr<spec_t>(), rsrc_mem);
      else {
        // Create primitive descriptor.
        auto reduction_pd = dnnl::reduction::primitive_desc(
                eng, dnnl::algorithm::reduction_sum, rsrc_md, rdst_md, float(0.f), float(0.f));

        // Create the primitive.
        auto reduction_prim = dnnl::reduction(reduction_pd);

        // Primitive arguments.
        std::unordered_map<int, dnnl::memory> reduction_args;
        reduction_args.insert({DNNL_ARG_SRC, rsrc_mem});
        reduction_args.insert({DNNL_ARG_DST, rdst_mem});

        // Primitive execution: Reduction (Sum).
        reduction_prim.execute(engine_stream, reduction_args);

        // Wait for the computation to finalize.
        engine_stream.wait();

        // Read data from memory object's handle.
        hetu::omp::read_from_dnnl_memory(output->data_ptr<spec_t>(), rdst_mem);

      }
      FreeToMemoryPool(temp_data_ptr);
  });
} 

template <typename spec_t>
void softmax_cross_entropy_gradient_cpu(
  const spec_t* pred, const spec_t* y_, const spec_t* grad_data,
  spec_t* output_data, int last_dim, size_t size) {  
  for (size_t idx = 0; idx < size; ++idx)
    output_data[idx] = (pred[idx] - y_[idx]) * grad_data[idx / last_dim];
}

void SoftmaxCrossEntropyGradientCpu(const NDArray& input_y,
                                    const NDArray& label, const NDArray& grad,
                                    NDArray& output, const Stream& stream) {
  size_t indim = input_y->ndim();
  HT_ASSERT(indim == label->ndim() && indim == output->ndim() &&
            indim == grad->ndim() + 1)
    << "Indim is " << indim << ", Label dim is " << label->ndim()
    << ", Output dim is " << output->ndim();
  int n_ = 1;
  for (int i = 0; i < indim - 1; ++i) {
    n_ *= input_y->shape(i);
  }
  int c_ = input_y->shape(indim - 1);
  size_t size = n_ * c_;

  dnnl::engine eng(dnnl::engine::kind::cpu, 0);
  dnnl::stream engine_stream(eng);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input_y->dtype(), spec_t, "SoftmaxCrossEntropyCuda", [&]() {
      const spec_t* grad_data = (const spec_t*) grad->data_ptr<spec_t>();
      const spec_t* y_data = (const spec_t*) input_y->data_ptr<spec_t>();
      const spec_t* label_data = (const spec_t*) label->data_ptr<spec_t>();
      spec_t* output_data = (spec_t*) output->data_ptr<spec_t>();

      DataPtr temp_data_ptr =
        AllocFromMemoryPool(grad->device(), size * sizeof(spec_t));
      void* temp_data = temp_data_ptr.ptr;

      spec_t alpha = 1.0;
      spec_t beta = 0.0;
      auto src_md = dnnl::memory::desc(input_y->shape(), dnnl::memory::data_type::f32, input_y->stride());
      auto dst_md = dnnl::memory::desc(input_y->shape(), dnnl::memory::data_type::f32, input_y->stride());
      auto src_mem = dnnl::memory(src_md, eng);
      auto dst_mem = dnnl::memory(dst_md, eng);

      hetu::omp::write_to_dnnl_memory(input_y->data_ptr<spec_t>(), src_mem);

      // Softmax axis.
      const int axis = 1;
      auto softmax_pd = dnnl::softmax_forward::primitive_desc(eng,
                        dnnl::prop_kind::forward_training, 
                        dnnl::algorithm::softmax_accurate, 
                        src_md, dst_md, axis);

      auto softmax_prim = dnnl::softmax_forward(softmax_pd);

      std::unordered_map<int, dnnl::memory> softmax_args;
      softmax_args.insert({DNNL_ARG_SRC, src_mem});
      softmax_args.insert({DNNL_ARG_DST, dst_mem});

      // Primitive execution.
      softmax_prim.execute(engine_stream, softmax_args);

      // Wait for the computation to finalize.
      engine_stream.wait();

      // Read data from dnnl::memory object's handle.
      hetu::omp::read_from_dnnl_memory(temp_data, dst_mem);

      softmax_cross_entropy_gradient_cpu<spec_t>(
          (const spec_t*) temp_data, label->data_ptr<spec_t>(),
          grad->data_ptr<spec_t>(), output->data_ptr<spec_t>(), c_, size);

      FreeToMemoryPool(temp_data_ptr);
    });
}
} // namespace impl
} // namespace hetu
