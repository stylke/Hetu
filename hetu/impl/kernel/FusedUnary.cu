// #include "hetu/core/ndarray.h"
// #include "hetu/impl/stream/CUDAStream.h"
// #include "hetu/impl/utils/common_utils.h"
// #include "hetu/impl/utils/cuda_utils.h"
// #include "hetu/impl/utils/cuda_math.h"
// #include "hetu/impl/utils/offset_calculator.cuh"
// #include "hetu/impl/kernel/Vectorized.cuh"
// #include "hetu/impl/kernel/Unary.cuh"
// #include "hetu/core/symbol.h"

// namespace hetu {
// namespace impl {

// template <typename spec_t>
// __global__ void fused_unary_kernel(const spec_t* input, const FusedParam* params,
//                                    size_t params_size, size_t size, spec_t* output,
//                                    const OffsetCalculator* in_offset_calculator,
//                                    const OffsetCalculator* out_offset_calculator) {
//   auto idx = blockIdx.x * blockDim.x + threadIdx.x;
//   if (idx >= size)
//     return;
//   spec_t val;
  
//   auto in_offset = in_offset_calculator->get(idx);
//   for (int i = 0; i < params_size; ++i) {
//     FusedParam param = params[i];
//     switch(param.type) {
//       case (FusedType::ADDCONST):
//         val = u_plus<spec_t>(spec_t(param.value))(val);
//         break;
//       case (FusedType::SUBCONST):
//         val = u_plus<spec_t>(spec_t(-param.value))(val);
//         break;
//       case (FusedType::SUBFROMCONST):
//         val = u_minus<spec_t>(spec_t(param.value))(val);
//         break;
//       case (FusedType::MULCONST):
//         val = u_multiplies<spec_t>(spec_t(param.value))(val);
//         break;
//       case (FusedType::DIVCONST):
//         val = u_multiplies<spec_t>(spec_t(1.0 / param.value))(val);
//         break;
//       case (FusedType::DIVFROMCONST):
//         val = u_divides<spec_t>(spec_t(param.value))(val);
//         break;
//       case (FusedType::POW):
//         val = u_pow<spec_t>(spec_t(param.value))(val);
//         break;
//       case (FusedType::EXP):
//         val = u_exp<spec_t>()(val);
//         break;
//       case (FusedType::LOG):
//         val = u_log<spec_t>()(val);
//         break;
//       case (FusedType::SQRT):
//         val = u_sqrt<spec_t>()(val);
//         break;
//       case (FusedType::ABS):
//         val = u_abs<spec_t>()(val);
//         break;
//     }
//   }
//   auto out_offset = out_offset_calculator->get(idx);
//   output[out_offset] = val;
// }

// void FusedUnaryCuda(const NDArray& input, std::vector<FusedParam> params, 
//                     NDArray& output, const Stream& stream) {
//   HT_ASSERT_CUDA_DEVICE(input);
//   HT_ASSERT_SAME_DEVICE(input, output);
//   HT_ASSERT_SAME_SHAPE(input, output);

//   size_t size = input->numel();
//   if (size == 0)
//     return;

//   dim3 blocks, threads;
//   threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
//   blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
//   CUDAStream cuda_stream(stream);
//   hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
//   NDArray in_offset_calculator_arr;
//   OffsetCalculator *in_offset_calculator;
//   std::tie(in_offset_calculator_arr, in_offset_calculator) = 
//     AllocOffsetCalculator(input, stream);
//   NDArray out_offset_calculator_arr;
//   OffsetCalculator *out_offset_calculator;
//   std::tie(out_offset_calculator_arr, out_offset_calculator) = 
//     AllocOffsetCalculator(output, stream);
//   NDArray param_arr = hetu::cuda::to_byte_ndarray((const uint8_t*)params.data(), 
//                                                   params.size() * sizeof(FusedParam), 
//                                                   cuda_stream.device_id());
//   HT_DISPATCH_FLOATING_TYPES(
//     input->dtype(), spec_t, "FusedUnaryCuda", [&]() {
//       fused_unary_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
//         input->data_ptr<spec_t>(), 
//         param_arr->data_ptr<FusedParam>(), params.size(), 
//         size, output->data_ptr<spec_t>(),
//         in_offset_calculator,
//         out_offset_calculator);
//     });
//   NDArray::MarkUsedBy({input, output}, stream);
// }

// } // namespace impl
// } // namespace hetu
