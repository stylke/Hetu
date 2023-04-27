#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/stream/CPUStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"

namespace hetu {
namespace impl {

template <typename spec_t, typename Operator>
void binary_const_cpu(const spec_t* input, spec_t value, size_t size,
                      Operator op, spec_t* output) {
  for (size_t idx = 0; idx < size; idx++)
    output[idx] = op(value, input[idx]);
}

template<typename Operator>
void BinaryConstToolCpu(const NDArray& input, double value,
                        NDArray& output, Operator op, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);

  size_t size = input->numel();
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "BinaryConstCpu", [&]() {
      binary_const_cpu<spec_t>(
        input->data_ptr<spec_t>(), static_cast<spec_t>(value), size, op,
        output->data_ptr<spec_t>());
    });
}

void AddConstCpu(const NDArray& input, double value,
                 NDArray& output, const Stream& stream) {
  CPUStream cpu_stream(stream);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
  input->dtype(), spec_t, "AddConstCpu", [&]() {
    auto _future = cpu_stream.EnqueueTask(
      [stream, input, output, value]() {
      dnnl::engine eng(dnnl::engine::kind::cpu, stream.stream_index());
      dnnl::memory::data_type mtype;
      if (input->dtype() == DataType::FLOAT32)
        mtype = dnnl::memory::data_type::f32;
      else 
        mtype = dnnl::memory::data_type::f64;
      auto mat_md = dnnl::memory::desc(input->shape(), mtype, input->stride());
      auto src_mem = dnnl::memory(mat_md, eng, input->data_ptr<spec_t>());
      auto dst_mem = dnnl::memory(mat_md, eng, output->data_ptr<spec_t>());

      auto AddConst_pd = dnnl::eltwise_forward::primitive_desc(eng, dnnl::prop_kind::forward_training,
                          dnnl::algorithm::eltwise_linear, mat_md, mat_md, float(1.f), float(value));
      auto AddConst = dnnl::eltwise_forward(AddConst_pd);
      std::unordered_map<int, dnnl::memory> binary_args;
      binary_args.insert({DNNL_ARG_SRC, src_mem});
      binary_args.insert({DNNL_ARG_DST, dst_mem});

      dnnl::stream engine_stream(eng);
      AddConst.execute(engine_stream, binary_args);
      engine_stream.wait();
    },
    "AddConst");
  });
  //cpu_stream.Sync();
}

void SubConstCpu(const NDArray& input, double value,
                 NDArray& output, const Stream& stream) {

  CPUStream cpu_stream(stream);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
  input->dtype(), spec_t, "SubConstCpu", [&]() {
    auto _future = cpu_stream.EnqueueTask(
    [stream, input, output, value]() {
      dnnl::engine eng(dnnl::engine::kind::cpu, stream.stream_index());
      auto mat_md = dnnl::memory::desc(input->shape(), dnnl::memory::data_type::f32, input->stride());
      auto src_mem = dnnl::memory(mat_md, eng, input->data_ptr<spec_t>());
      auto dst_mem = dnnl::memory(mat_md, eng, output->data_ptr<spec_t>());

      auto SubConst_pd = dnnl::eltwise_forward::primitive_desc(eng, dnnl::prop_kind::forward_training,
                          dnnl::algorithm::eltwise_linear, mat_md, mat_md, float(-1.f), float(value));
      auto SubConst = dnnl::eltwise_forward(SubConst_pd);

      std::unordered_map<int, dnnl::memory> binary_args;
      binary_args.insert({DNNL_ARG_SRC, src_mem});
      binary_args.insert({DNNL_ARG_DST, dst_mem});
      dnnl::stream engine_stream(eng);
      SubConst.execute(engine_stream, binary_args);
      engine_stream.wait();
    },
    "SubConst");
    //cpu_stream.Sync();
  });
}

void MulConstCpu(const NDArray& input, double value,
                 NDArray& output, const Stream& stream) {
  CPUStream cpu_stream(stream);
  dnnl::engine eng(dnnl::engine::kind::cpu, cpu_stream.stream_id());
  dnnl::stream engine_stream(eng);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
  input->dtype(), spec_t, "MulConstCpu", [&]() {
    auto _future = cpu_stream.EnqueueTask(
      [stream, input, output, value]() {
      dnnl::engine eng(dnnl::engine::kind::cpu, stream.stream_index());
      auto mat_md = dnnl::memory::desc(input->shape(), dnnl::memory::data_type::f32, input->stride());
      auto src_mem = dnnl::memory(mat_md, eng, input->data_ptr<spec_t>());
      auto dst_mem = dnnl::memory(mat_md, eng, output->data_ptr<spec_t>());

      auto MulConst_pd = dnnl::eltwise_forward::primitive_desc(eng, dnnl::prop_kind::forward_training,
                          dnnl::algorithm::eltwise_linear, mat_md, mat_md, float(value), float(0.f));
      auto MulConst = dnnl::eltwise_forward(MulConst_pd);

      std::unordered_map<int, dnnl::memory> binary_args;
      binary_args.insert({DNNL_ARG_SRC, src_mem});
      binary_args.insert({DNNL_ARG_DST, dst_mem});
      dnnl::stream engine_stream(eng);
      MulConst.execute(engine_stream, binary_args);
      engine_stream.wait();
    },
    "MulConst");
    //cpu_stream.Sync();
  });
}

void DivConstCpu(const NDArray& input, double value,
                 NDArray& output, const Stream& stream) {

  CPUStream cpu_stream(stream);
  dnnl::engine eng(dnnl::engine::kind::cpu, cpu_stream.stream_id());
  dnnl::stream engine_stream(eng);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
  input->dtype(), spec_t, "DivConstCpu", [&]() {
    auto _future = cpu_stream.EnqueueTask(
      [stream, input, output, value]() {
      dnnl::engine eng(dnnl::engine::kind::cpu, stream.stream_index());
      auto mat_md = dnnl::memory::desc(input->shape(), dnnl::memory::data_type::f32, input->stride());
      auto src_mem = dnnl::memory(mat_md, eng, input->data_ptr<spec_t>());
      auto dst_mem = dnnl::memory(mat_md, eng, output->data_ptr<spec_t>());

      auto DivConst_pd = dnnl::eltwise_forward::primitive_desc(eng, dnnl::prop_kind::forward_training,
                          dnnl::algorithm::eltwise_pow, mat_md, mat_md, float(value), float(-1.f));
      auto DivConst = dnnl::eltwise_forward(DivConst_pd);

      std::unordered_map<int, dnnl::memory> binary_args;
      binary_args.insert({DNNL_ARG_SRC, src_mem});
      binary_args.insert({DNNL_ARG_DST, dst_mem});
      dnnl::stream engine_stream(eng);
      DivConst.execute(engine_stream, binary_args);
      engine_stream.wait();
    },
    "DivConst");
    //cpu_stream.Sync();
  });
}

} // namespace impl
} // namespace hetu
