#include "hetu/core/ndarray.h"
#include "hetu/execution/dar_executor.h"
#include "hetu/execution/dbr_executor.h"
#include "hetu/impl/communication/comm_group.h"
#include "hetu/impl/communication/mpi_comm_group.h"
#include "hetu/impl/communication/nccl_comm_group.h"
#include "hetu/autograd/ops/Variable.h"
#include "hetu/autograd/ops/Sum.h"
#include "hetu/autograd/ops/Communicate.h"
#include "test_utils.h"
#include <cmath>
#include <vector>
#include <numeric>

using namespace hetu;
using namespace hetu::autograd;
using namespace hetu::execution;
using namespace hetu::impl;
using namespace hetu::impl::comm;

constexpr auto TEST_DATA_TYPES = {kFloat32, kFloat64};

void TestDARAllReduceOp(DeviceType device_type, DataType dtype = kFloat32, const HTShape& shape = {1024, 1024}, bool print_result = false) {
  auto local_device = GetLocalDevice();
  auto all_devices = GetGlobalDeviceGroup();
  auto local_rank = all_devices.get_index(local_device);
  auto world_size = all_devices.num_devices();
  if(local_rank == 0){
    HT_LOG_INFO << "Testing AllReduceOp for device " << device_type
                << " and type " << dtype << "...";
  }
  HT_ASSERT(world_size >= 2) << "device num must >= 2 !";
  
  auto placeholder_op = PlaceholderOp(dtype, shape, OpMeta().set_device_group(all_devices)); 
  auto input_tensor = placeholder_op->output(0);
  auto comm_op = AllReduceOp(input_tensor, OpMeta().set_device_group(all_devices));
  auto output_tensor = comm_op->output(0);

  DARExecutor exec(local_device, all_devices, {output_tensor});
  const double a = 1.23, b = 3.141;
  std::vector<double> v;
  for(unsigned int i = 0; i < world_size; ++i){
    v.push_back(a + b * i);
  }
  double value = v[local_rank];
  double ground_truth = std::accumulate(v.begin(), v.end(), 0.000);
  if(local_rank == 0 && print_result){
    HT_LOG_INFO << "ground truth:" << ground_truth;
  }
  NDArray data = NDArray::full(shape, value, local_device, dtype);

  SynchronizeAllStreams();
  if(print_result){
    HT_LOG_INFO << local_device << ": init data = " << data;
  }
  FeedDict feed_dict = {{input_tensor->id(), data}};
  auto result = exec.Run({output_tensor}, feed_dict);  
  SynchronizeAllStreams();
  auto output_array = result[0];
  if(print_result){
    HT_LOG_INFO << local_device << ": result = " << output_array;
  }
  assert_fuzzy_eq(output_array, ground_truth);
  if(local_rank == 0){
    HT_LOG_INFO << "Testing AllReduceOp for device " << device_type
                << " and type " << dtype << " done";
  }
}

void TestDARReduceCommOp(DeviceType device_type, DataType dtype = kFloat32, const HTShape& shape = {1024, 1024}, bool print_result = false) {
  auto local_device = GetLocalDevice();
  auto all_devices = GetGlobalDeviceGroup();
  auto local_rank = all_devices.get_index(local_device);
  auto world_size = all_devices.num_devices();
  if(local_rank == 0){
    HT_LOG_INFO << "Testing ReduceCommOp for device " << device_type
                << " and type " << dtype << "...";
  }
  HT_ASSERT(world_size >= 2) << "device num must >= 2 !";
  
  auto placeholder_op = PlaceholderOp(dtype, shape, OpMeta().set_device_group(all_devices)); 
  auto input_tensor = placeholder_op->output(0);
  int reducer = world_size / 2;
  auto comm_op = ReduceCommOp(input_tensor, reducer, OpMeta().set_device_group(all_devices));
  auto output_tensor = comm_op->output(0);
  auto comm_grad_op = ReduceCommGradientOp(output_tensor, reducer, OpMeta().set_device_group(all_devices));
  auto input_grad = comm_grad_op->output(0);

  DARExecutor exec(local_device, all_devices, {output_tensor, input_grad});

  const double a = 1.23, b = 3.141;
  std::vector<double> v;
  for(unsigned int i = 0; i < world_size; ++i){
    v.push_back(a + b * i);
  }
  double value = v[local_rank];
  double ground_truth = std::accumulate(v.begin(), v.end(), 0.000);
  if(local_rank == 0 && print_result){
    HT_LOG_INFO << "ground truth:" << ground_truth;
  }
  NDArray data = NDArray::full(shape, value, local_device, dtype);

  SynchronizeAllStreams();
  if(print_result){
    HT_LOG_INFO << local_device << ": init data = " << data;
  }
  FeedDict feed_dict = {{input_tensor->id(), data}};
  auto result = exec.Run({output_tensor, input_grad}, feed_dict); 
  SynchronizeAllStreams();
  auto output_array = result[0], input_grad_array = result[1];
  if(print_result){
    HT_LOG_INFO << local_device << ": result = " << output_array;
    HT_LOG_INFO << local_device << ": input_grad = " << input_grad_array;
  }
  if((int)local_rank == reducer){
    assert_fuzzy_eq(output_array, ground_truth);
  }
  assert_fuzzy_eq(input_grad_array, ground_truth);
  if(local_rank == 0){
  HT_LOG_INFO << "Testing ReduceCommOp for device " << device_type
          << " and type " << dtype << "done";
  }
}

void TestDARBroadcastCommOp(DeviceType device_type, DataType dtype = kFloat32, const HTShape& shape = {1024, 1024}, bool print_result = false) {
  auto local_device = GetLocalDevice();
  auto all_devices = GetGlobalDeviceGroup();
  auto local_rank = all_devices.get_index(local_device);
  auto world_size = all_devices.num_devices();
  if(local_rank == 0){
    HT_LOG_INFO << "Testing BroadcastCommOp for device " << device_type
                << " and type " << dtype << "...";
  }
  HT_ASSERT(world_size >= 2) << "device num must >= 2 !";
  
  auto placeholder_op = PlaceholderOp(dtype, shape, OpMeta().set_device_group(all_devices)); 
  auto input_tensor = placeholder_op->output(0);
  int broadcaster = world_size / 2;
  auto comm_op = BroadcastCommOp(input_tensor, broadcaster, OpMeta().set_device_group(all_devices));
  auto output_tensor = comm_op->output(0);
  auto comm_grad_op = BroadcastCommGradientOp(output_tensor, broadcaster, OpMeta().set_device_group(all_devices));
  auto input_grad = comm_grad_op->output(0);

  DARExecutor exec(local_device, all_devices, {output_tensor, input_grad});

  const double a = 1.23, b = 3.141;
  std::vector<double> v;
  for(unsigned int i = 0; i < world_size; ++i){
    v.push_back(a + b * i);
  }
  double value = v[local_rank];
  double ground_truth = v[broadcaster];
  if(local_rank == 0 && print_result){
    HT_LOG_INFO << "ground truth:" << ground_truth;
  }
  NDArray data = NDArray::full(shape, value, local_device, dtype);

  SynchronizeAllStreams();
  if(print_result){
    HT_LOG_INFO << local_device << ": init data = " << data;
  }
  FeedDict feed_dict = {{input_tensor->id(), data}};
  auto result = exec.Run({output_tensor, input_grad}, feed_dict); 
  SynchronizeAllStreams();
  auto output_array = result[0], input_grad_array = result[1];
  if(print_result){
    HT_LOG_INFO << local_device << ": result = " << output_array;
    HT_LOG_INFO << local_device << ": input_grad = " << input_grad_array;
  } 
  assert_fuzzy_eq(output_array, ground_truth);
  if((int)local_rank == broadcaster){
    assert_fuzzy_eq(input_grad_array, value * world_size);
    HT_LOG_INFO << "Testing BroadcastCommOp for device " << device_type
                << " and type " << dtype << "done";
  }
}

void TestDARAllGatherOp(DeviceType device_type, DataType dtype = kFloat32, const HTShape& shape = {1024, 1024}, bool print_result = false) {
  auto local_device = GetLocalDevice();
  auto all_devices = GetGlobalDeviceGroup();
  auto local_rank = all_devices.get_index(local_device);
  auto world_size = all_devices.num_devices();
  if(local_rank == 0){
    HT_LOG_INFO << "Testing AllGatherOp for device " << device_type
                << " and type " << dtype << "...";
  }
  HT_ASSERT(world_size >= 2) << "device num must >= 2 !";
  
  auto placeholder_op = PlaceholderOp(dtype, shape, OpMeta().set_device_group(all_devices)); 
  auto input_tensor = placeholder_op->output(0);
  auto comm_op = AllGatherOp(input_tensor, OpMeta().set_device_group(all_devices));
  auto output_tensor = comm_op->output(0);
  auto comm_grad_op = AllGatherGradientOp(output_tensor, OpMeta().set_device_group(all_devices));
  auto input_grad = comm_grad_op->output(0);

  DARExecutor exec(local_device, all_devices, {output_tensor, input_grad});

  const double a = 1.23, b = 3.141;
  std::vector<double> v;
  for(unsigned int i = 0; i < world_size; ++i){
    v.push_back(a + b * i);
  }
  double value = v[local_rank];
  NDArray data = NDArray::full(shape, value, local_device, dtype);

  SynchronizeAllStreams();
  if(print_result){
    HT_LOG_INFO << local_device << ": init data = " << data;
  }
  FeedDict feed_dict = {{input_tensor->id(), data}};
  auto result = exec.Run({output_tensor, input_grad}, feed_dict); 
  SynchronizeAllStreams();
  auto output_array = result[0], input_grad_array = result[1];
  auto splits = NDArray::split(output_array, world_size, 0);
  if(print_result){
    HT_LOG_INFO << local_device << ": result = " << output_array;
    HT_LOG_INFO << local_device << ": input_grad = " << input_grad_array;
  }
  for (unsigned int r = 0; r < world_size; r++) {
    assert_fuzzy_eq(splits[r], v[r]);
  }
  assert_fuzzy_eq(input_grad_array, value);
  if(local_rank == 0){
    HT_LOG_INFO << "Testing AllGatherOp for device " << device_type
                << " and type " << dtype << " done";
  }
}

void TestDARReduceScatterOp(DeviceType device_type, DataType dtype = kFloat32, const HTShape& shape = {1024, 1024}, bool print_result = false) {
  auto local_device = GetLocalDevice();
  auto all_devices = GetGlobalDeviceGroup();
  auto local_rank = all_devices.get_index(local_device);
  auto world_size = all_devices.num_devices();
  if(local_rank == 0){
    HT_LOG_INFO << "Testing ReduceScatterOp for device " << device_type
                << " and type " << dtype << "...";
  }
  HT_ASSERT(world_size >= 2) << "device num must >= 2 !";
  
  auto placeholder_op = PlaceholderOp(dtype, shape, OpMeta().set_device_group(all_devices)); 
  auto input_tensor = placeholder_op->output(0);
  auto comm_op = ReduceScatterOp(input_tensor, OpMeta().set_device_group(all_devices));
  auto output_tensor = comm_op->output(0);
  auto comm_grad_op = ReduceScatterGradientOp(output_tensor, OpMeta().set_device_group(all_devices));
  auto input_grad = comm_grad_op->output(0);

  DARExecutor exec(local_device, all_devices, {output_tensor, input_grad});

  const double a = 1.23, b = 3.141;
  std::vector<double> v;
  for(unsigned int i = 0; i < world_size; ++i){
    v.push_back(a + b * i);
  }
  double value = v[local_rank];
  double ground_truth = std::accumulate(v.begin(), v.end(), 0.000);
  if(local_rank == 0 && print_result){
    HT_LOG_INFO << "ground truth:" << ground_truth;
  }
  NDArray data = NDArray::full(shape, value, local_device, dtype);

  SynchronizeAllStreams();
  if(print_result){
    HT_LOG_INFO << local_device << ": init data = " << data;
  }
  FeedDict feed_dict = {{input_tensor->id(), data}};
  auto result = exec.Run({output_tensor, input_grad}, feed_dict); 
  SynchronizeAllStreams();
  auto output_array = result[0], input_grad_array = result[1];
  if(print_result){
    HT_LOG_INFO << local_device << ": result = " << output_array;
    HT_LOG_INFO << local_device << ": input_grad = " << input_grad_array;
  }
  assert_fuzzy_eq(output_array, ground_truth);
  assert_fuzzy_eq(input_grad_array, ground_truth);
  if(local_rank == 0){
    HT_LOG_INFO << "Testing ReduceScatterOp for device " << device_type
                << " and type " << dtype << " done";
  }
}

void TestDARGatherOp(DeviceType device_type, DataType dtype = kFloat32, const HTShape& shape = {1024, 1024}, bool print_result = false) {
  auto local_device = GetLocalDevice();
  auto all_devices = GetGlobalDeviceGroup();
  auto local_rank = all_devices.get_index(local_device);
  auto world_size = all_devices.num_devices();
  if(local_rank == 0){
    HT_LOG_INFO << "Testing GatherOp for device " << device_type
                << " and type " << dtype << "...";
  }
  HT_ASSERT(world_size >= 2) << "device num must >= 2 !";
  
  auto placeholder_op = PlaceholderOp(dtype, shape, OpMeta().set_device_group(all_devices)); 
  auto input_tensor = placeholder_op->output(0);
  int gatherer = world_size / 2;
  auto comm_op = GatherOp(input_tensor, gatherer, OpMeta().set_device_group(all_devices));
  auto output_tensor = comm_op->output(0);
  auto comm_grad_op = GatherGradientOp(output_tensor, gatherer, OpMeta().set_device_group(all_devices));
  auto input_grad = comm_grad_op->output(0);

  DARExecutor exec(local_device, all_devices, {output_tensor, input_grad});

  const double a = 1.23, b = 3.141;
  std::vector<double> v;
  for(unsigned int i = 0; i < world_size; ++i){
    v.push_back(a + b * i);
  }
  double value = v[local_rank];
  NDArray data = NDArray::full(shape, value, local_device, dtype);

  SynchronizeAllStreams();
  if(print_result){
    HT_LOG_INFO << local_device << ": init data = " << data;
  }
  FeedDict feed_dict = {{input_tensor->id(), data}};
  auto result = exec.Run({output_tensor, input_grad}, feed_dict); 
  SynchronizeAllStreams();
  auto output_array = result[0], input_grad_array = result[1];
  if(print_result){
    HT_LOG_INFO << local_device << ": result = " << output_array;
    HT_LOG_INFO << local_device << ": input_grad = " << input_grad_array;
  }
  if((int)local_rank == gatherer){
    auto splits = NDArray::split(output_array, world_size, 0);
    for (unsigned int r = 0; r < world_size; r++) {
      assert_fuzzy_eq(splits[r], v[r]);
    }
  }
  assert_fuzzy_eq(input_grad_array, value);
  if(local_rank == 0){
    HT_LOG_INFO << "Testing GatherOp for device " << device_type
                << " and type " << dtype << " done";
  }
}

void TestDARScatterOp(DeviceType device_type, DataType dtype = kFloat32, const HTShape& shape = {1024, 1024}, bool print_result = false) {
  auto local_device = GetLocalDevice();
  auto all_devices = GetGlobalDeviceGroup();
  auto local_rank = all_devices.get_index(local_device);
  auto world_size = all_devices.num_devices();
  if(local_rank == 0){
    HT_LOG_INFO << "Testing ScatterOp for device " << device_type
                << " and type " << dtype << "...";
  }
  HT_ASSERT(world_size >= 2) << "device num must >= 2 !";
  
  auto placeholder_op = PlaceholderOp(dtype, shape, OpMeta().set_device_group(all_devices)); 
  auto input_tensor = placeholder_op->output(0);
  int scatterer = world_size / 2;
  auto comm_op = ScatterOp(input_tensor, scatterer, OpMeta().set_device_group(all_devices));
  auto output_tensor = comm_op->output(0);
  auto comm_grad_op = ScatterGradientOp(output_tensor, scatterer, OpMeta().set_device_group(all_devices));
  auto input_grad = comm_grad_op->output(0);

  DARExecutor exec(local_device, all_devices, {output_tensor, input_grad});

  const double a = 1.23, b = 3.141;
  std::vector<double> v;
  for(unsigned int i = 0; i < world_size; ++i){
    v.push_back(a + b * i);
  }
  double value = v[local_rank];
  double ground_truth = v[scatterer];
  if(local_rank == 0 && print_result){
    HT_LOG_INFO << "ground truth:" << ground_truth;
  }
  NDArray data = NDArray::full(shape, value, local_device, dtype);

  SynchronizeAllStreams();
  if(print_result){
    HT_LOG_INFO << local_device << ": init data = " << data;
  }
  FeedDict feed_dict = {{input_tensor->id(), data}};
  auto result = exec.Run({output_tensor, input_grad}, feed_dict); 
  SynchronizeAllStreams();
  auto output_array = result[0], input_grad_array = result[1];
  if(print_result){
    HT_LOG_INFO << local_device << ": result = " << output_array;
    HT_LOG_INFO << local_device << ": input_grad = " << input_grad_array;
  }
  assert_fuzzy_eq(output_array, ground_truth);
  if((int)local_rank == scatterer){
    assert_fuzzy_eq(input_grad_array, ground_truth);
    HT_LOG_INFO << "Testing ScatterOp for device " << device_type
                << " and type " << dtype << " done";
  }
}

int main(int argc, char** argv) {
  // auto device_type = kCPU;
  auto device_type = kCUDA;
  const HTShape& shape = {8, 8};
  bool print_result = false;
  if(device_type == kCPU){
    SetUpDeviceMappingWithAssignedLocalDeviceOnce(Device("cpu", GetWorldRank()));
  }
  else if(device_type == kCUDA){
    SetUpDeviceMappingWithAssignedLocalDeviceOnce(Device(device_type, GetWorldRank()));
  }
  for (const auto& dtype : TEST_DATA_TYPES) {
    TestDARAllReduceOp(device_type, dtype, shape, print_result);
    TestDARReduceCommOp(device_type, dtype, shape, print_result);
    TestDARBroadcastCommOp(device_type, dtype, shape, print_result);
    TestDARAllGatherOp(device_type, dtype, shape, print_result);
    TestDARReduceScatterOp(device_type, dtype, shape, print_result);
    TestDARGatherOp(device_type, dtype, shape, print_result);
    TestDARScatterOp(device_type, dtype, shape, print_result);
  }
  return 0;
}