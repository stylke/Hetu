#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/stream/CPUStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"

namespace hetu {
namespace impl {

void BinaryElewiseToolCpu(const NDArray& inputA, const NDArray& inputB,
                          NDArray& output, dnnl::algorithm op, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(inputA);
  HT_ASSERT_SAME_DEVICE(inputA, output);
  HT_ASSERT_SAME_DEVICE(inputB, output);

  size_t size;
  size_t sizeA = inputA->numel();
  size_t sizeB = inputB->numel();
  dnnl::memory::data_type mtype;
  if (inputA->dtype() == DataType::FLOAT32)
    mtype = dnnl::memory::data_type::f32;
  else
    mtype = dnnl::memory::data_type::f64;
  dnnl::engine eng(dnnl::engine::kind::cpu, 0);
  dnnl::stream engine_stream(eng);
  dnnl::memory::dims A_dims(output->ndim());
  dnnl::memory::dims A_stride(output->ndim());
  dnnl::memory::dims B_dims(output->ndim());
  dnnl::memory::dims B_stride(output->ndim());
  dnnl::memory::dims out_strides(output->ndim());

  size_t output_dim = output->ndim();
  size_t output_size = 1;
  size_t A_size = 1;
  size_t B_size = 1;
  size_t diff = output_dim - inputA->ndim();

  for (int i = output_dim - 1; i >= 0; --i) {
    out_strides[i] = output_size;
    output_size *= output->shape(i);
    if (i < int(diff)) {
      A_dims[i] = 1;
    } else {
      A_dims[i] = inputA->shape(i - diff);
    }
    A_stride[i] = A_size;
    A_size *= A_dims[i];
  }
  diff = output_dim - inputB->ndim();
  for (int i = output_dim - 1; i >= 0; --i) {
    if (i < int(diff)) {
      B_dims[i] = 1;
    } else {
      B_dims[i] = inputB->shape(i - diff);
    }
    B_stride[i] = B_size;
    B_size *= B_dims[i];
  }

  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    inputA->dtype(), spec_t, "BinaryElewiseCpu", [&]() {
          auto src_A_md = dnnl::memory::desc(A_dims, mtype, A_stride);
          auto src_B_md = dnnl::memory::desc(B_dims, mtype, B_stride);
          auto dst_md = dnnl::memory::desc(output->shape(), mtype, out_strides);

          // Create src memory objects.
          auto src_A_mem = dnnl::memory(src_A_md, eng);
          auto src_B_mem = dnnl::memory(src_B_md, eng);
          auto dst_mem = dnnl::memory(dst_md, eng);

          // Write data to memory object's handle.
          hetu::omp::write_to_dnnl_memory(inputA->data_ptr<spec_t>(), src_A_mem);
          hetu::omp::write_to_dnnl_memory(inputB->data_ptr<spec_t>(), src_B_mem);

          auto binary_pd = dnnl::binary::primitive_desc(eng, op,
                  src_A_md, src_B_md, dst_md);

          // Create the primitive.
          auto binary_prim = dnnl::binary(binary_pd);

          // Primitive arguments. Set up in-place execution by assigning src_0 as DST.
          std::unordered_map<int, dnnl::memory> binary_args;
          binary_args.insert({DNNL_ARG_SRC_0, src_A_mem});
          binary_args.insert({DNNL_ARG_SRC_1, src_B_mem});
          binary_args.insert({DNNL_ARG_DST, dst_mem});

          // Primitive execution: binary with ReLU.
          binary_prim.execute(engine_stream, binary_args);

          // Wait for the computation to finalize.
          engine_stream.wait();

          // Read data from memory object's handle.
          hetu::omp::read_from_dnnl_memory(output->data_ptr<spec_t>(), dst_mem);
    });
}

void AddElewiseCpu(const NDArray& inputA, const NDArray& inputB,
                    NDArray& output, const Stream& stream) {

  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
      inputA->dtype(), spec_t, "AddElewiseCpu", [&]() {
        BinaryElewiseToolCpu(inputA, inputB, output, dnnl::algorithm::binary_add, stream);
      }); 
}

void SubElewiseCpu(const NDArray& inputA, const NDArray& inputB,
                    NDArray& output, const Stream& stream) {

  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
      inputA->dtype(), spec_t, "SubElewiseCpu", [&]() {
        BinaryElewiseToolCpu(inputA, inputB, output, dnnl::algorithm::binary_sub, stream);
      }); 
}

void MulElewiseCpu(const NDArray& inputA, const NDArray& inputB,
                    NDArray& output, const Stream& stream) {

  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
      inputA->dtype(), spec_t, "MulElewiseCpu", [&]() {
        BinaryElewiseToolCpu(inputA, inputB, output, dnnl::algorithm::binary_mul, stream);
      }); 
}

void DivElewiseCpu(const NDArray& inputA, const NDArray& inputB,
                    NDArray& output, const Stream& stream) {

  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
      inputA->dtype(), spec_t, "DivElewiseCpu", [&]() {
        BinaryElewiseToolCpu(inputA, inputB, output, dnnl::algorithm::binary_div, stream);
      }); 
}

} // namespace impl
} // namespace hetu
