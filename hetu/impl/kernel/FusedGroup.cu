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
// __global__ void fused_group_kernel(const spec_t** inputs, size_t num_inputs,
//                                    const int64_t* input_output_idx,
//                                    const int64_t* num_input_output,
//                                    const FusedParam* params,
//                                    size_t params_size, size_t size, 
//                                    int64_t num_tensors,
//                                    spec_t** outputs, size_t num_outputs) {
//   size_t num_splits = params_size * 2;
//   auto idx = blockIdx.x * blockDim.x + threadIdx.x;
//   if (idx >= size)
//     return;
//   // spec_t* vals = new spec_t[num_tensors];
//   spec_t vals[128];
//   for (int i = 0; i < num_inputs; ++i) {
//     const spec_t* input = inputs[i];
//     vals[i] = input[idx];
//   }
  
//   int64_t ptr = 0;
//   for (int i = 0; i < params_size; ++i) {
//     int64_t op_num_inputs = num_input_output[i * 2];
//     int64_t op_num_outputs = num_input_output[i * 2 + 1];
//     // int64_t* input_idxs =  new int64_t[op_num_inputs];
//     // int64_t* output_idxs =  new int64_t[op_num_outputs];
//     int64_t input_idxs[10];
//     int64_t output_idxs[10];
//     for (int j = 0; j < op_num_inputs; ++j) {
//       input_idxs[j] = input_output_idx[ptr + j];
//     }
//     ptr += op_num_inputs;
//     for (int j = 0; j < op_num_outputs; ++j) {
//       output_idxs[j] = input_output_idx[ptr + j];
//     }
//     ptr += op_num_outputs;
//     FusedParam param = params[i];
//     switch(param.type) {
//       case (FusedType::ADDCONST):
//         vals[output_idxs[0]] = u_plus<spec_t>(spec_t(param.value))(vals[input_idxs[0]]);
//         break;
//       case (FusedType::SUBCONST):
//         vals[output_idxs[0]] = u_plus<spec_t>(spec_t(-param.value))(vals[input_idxs[0]]);
//         break;
//       case (FusedType::SUBFROMCONST):
//         vals[output_idxs[0]] = u_minus<spec_t>(spec_t(param.value))(vals[input_idxs[0]]);
//         break;
//       case (FusedType::MULCONST):
//         vals[output_idxs[0]] = u_multiplies<spec_t>(spec_t(param.value))(vals[input_idxs[0]]);
//         break;
//       case (FusedType::DIVCONST):
//         vals[output_idxs[0]] = u_multiplies<spec_t>(spec_t(1.0 / param.value))(vals[input_idxs[0]]);
//         break;
//       case (FusedType::DIVFROMCONST):
//         vals[output_idxs[0]] = u_divides<spec_t>(spec_t(param.value))(vals[input_idxs[0]]);
//         break;
//       case (FusedType::POW):
//         vals[output_idxs[0]] = u_pow<spec_t>(spec_t(param.value))(vals[input_idxs[0]]);
//         break;
//       case (FusedType::EXP):
//         vals[output_idxs[0]] = u_exp<spec_t>()(vals[input_idxs[0]]);
//         break;
//       case (FusedType::LOG):
//         vals[output_idxs[0]] = u_log<spec_t>()(vals[input_idxs[0]]);
//         break;
//       case (FusedType::SQRT):
//         vals[output_idxs[0]] = u_sqrt<spec_t>()(vals[input_idxs[0]]);
//         break;
//       case (FusedType::ABS):
//         vals[output_idxs[0]] = u_abs<spec_t>()(vals[input_idxs[0]]);
//         break;
//       case (FusedType::GELU):
//         vals[output_idxs[0]] = u_gelu<spec_t>()(vals[input_idxs[0]]);
//       case (FusedType::OUTPUT):
//         for (int j = 0; j < op_num_inputs; ++j) {
//           spec_t* output = outputs[j];
//           output[idx] = vals[input_idxs[j]];
//         }
//         break;
//     }
//   }
// }

// void FusedGroupCuda(const NDArrayList& inputs, FusedGroupParam fusedgroup,  
//                     NDArrayList& outputs, const Stream& stream) {

//   size_t size = inputs.at(0)->numel();
//   if (size == 0)
//     return;

//   dim3 blocks, threads;
//   threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
//   blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
//   CUDAStream cuda_stream(stream);
//   hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
//   NDArray input_output_idx_arr = hetu::cuda::to_int64_ndarray(fusedgroup.input_output_idx,
//                                                               cuda_stream.device_id());
//   NDArray num_input_output_arr = hetu::cuda::to_int64_ndarray(fusedgroup.num_input_output,
//                                                               cuda_stream.device_id());
//   NDArray param_arr = hetu::cuda::to_byte_ndarray((const uint8_t*)fusedgroup.param_info.data(), 
//                                                   fusedgroup.param_info.size() * sizeof(FusedParam), 
//                                                   cuda_stream.device_id());
//   HT_DISPATCH_FLOATING_TYPES(
//     inputs[0]->dtype(), spec_t, "FusedUnaryCuda", [&]() {
//       std::vector<spec_t*> input_ptrs;
//       std::vector<spec_t*> output_ptrs;
//       for (int i = 0; i < inputs.size(); ++i) {
//         input_ptrs.push_back(inputs[i]->data_ptr<spec_t>());
//       }
//       for (int i = 0; i < outputs.size(); ++i) {
//         output_ptrs.push_back(outputs[i]->data_ptr<spec_t>());
//       }
//       NDArray inputs_arr = hetu::cuda::to_byte_ndarray((const uint8_t*)input_ptrs.data(), 
//                                                        input_ptrs.size() * sizeof(spec_t*), 
//                                                        cuda_stream.device_id());
//       NDArray outputs_arr = hetu::cuda::to_byte_ndarray((const uint8_t*)output_ptrs.data(), 
//                                                        output_ptrs.size() * sizeof(spec_t*), 
//                                                        cuda_stream.device_id());
//       // HT_LOG_INFO << input_ptrs << " " << inputs.size() << " "
//       // << inputs[0]->raw_data_ptr() << " " << inputs[0]->is_contiguous() << " "
//       // << output_ptrs << " " << outputs[0]->raw_data_ptr() << "\n"
//       // << fusedgroup.input_output_idx << " " << fusedgroup.num_input_output 
//       // << " " << fusedgroup.param_info.size() << " " << size << " " << fusedgroup.num_tensors;
//       fused_group_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
//         inputs_arr->data_ptr<const spec_t*>(), inputs.size(),
//         input_output_idx_arr->data_ptr<const int64_t>(),
//         num_input_output_arr->data_ptr<const int64_t>(),
//         param_arr->data_ptr<const FusedParam>(), fusedgroup.param_info.size(),
//         size, fusedgroup.num_tensors,
//         outputs_arr->data_ptr<spec_t*>(), outputs.size());
//     });
//   NDArray::MarkUsedBy(inputs, stream);
//   NDArray::MarkUsedBy(outputs, stream);
//   NDArray::MarkUsedBy({input_output_idx_arr, num_input_output_arr, param_arr}, stream);
// }

// } // namespace impl
// } // namespace hetu
