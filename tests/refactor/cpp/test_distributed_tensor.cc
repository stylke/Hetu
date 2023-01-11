#include "hetu/core/ndarray.h"
#include "hetu/execution/dar_executor.h"
#include "hetu/execution/dbr_executor.h"
#include "hetu/impl/communication/comm_group.h"
#include "hetu/autograd/optim/Optimizer.h"
#include "hetu/autograd/ops/Variable.h"
#include "hetu/autograd/ops/MatMul.h"
#include "hetu/autograd/ops/BinaryCrossEntropy.h"
#include "hetu/autograd/ops/Sigmoid.h"
#include "hetu/autograd/ops/Relu.h"
#include "hetu/autograd/ops/Sum.h"
#include "hetu/autograd/ops/Communicate.h"
#include "hetu/autograd/distributed_states.h"
#include <cmath>

using namespace hetu;
using namespace hetu::autograd;
using namespace hetu::execution;
using namespace hetu::impl;
using namespace hetu::impl::comm;

void TestDARDistributedTensor(DataType dtype = kFloat32) {
  auto local_device = GetLocalDevice();
  auto all_devices = GetGlobalDeviceGroup();
  HT_ASSERT(all_devices.num_devices() >= 4) << "device num must >= 4 !";
  DeviceGroup all_device_group({all_devices.get(0), all_devices.get(1), all_devices.get(2), all_devices.get(3)});
  
  int dim = 2;
  auto placeholder_op = PlaceholderOp(dtype, {-1, dim}, OpMeta().set_device_group(all_device_group)); 
  auto tensor1 = placeholder_op->output(0);
  DistributedStates ds1(4, {{-2, 1}, {-1, 1}, {0, 4}}, {0});
  DistributedStates ds2(4, {{-2, 1}, {-1, 2}, {0, 2}}, {-1, 0});
  tensor1->set_distributed_states(ds1);
  auto comm_op = CommOp(tensor1, ds2, OpMeta().set_device_group(all_device_group));  
  auto tensor2 = comm_op->output(0);
  TensorList ls({tensor2, tensor2});
  auto result = SumOp(ls)->output(0);

  DARExecutor exec(local_device, all_devices, {result});

  NDArray data = NDArray::randn({8/4, dim}, local_device, dtype, 0.0, 1.0, (666 + all_device_group.get_index(local_device))); // ds1->ds2
  // NDArray data = NDArray::randn({8/4, dim}, local_device, dtype, 0.0, 1.0, (666 + all_device_group.get_index(local_device) % 2)); // ds2->ds1
  HT_LOG_INFO << local_device << ": init data = " << data;
  SynchronizeAllStreams();
  
  FeedDict feed_dict = {{tensor1->id(), data}};

  auto r = exec.Run({result}, feed_dict);
  HT_LOG_INFO << local_device << ": result = " << r;
}

int main(int argc, char** argv) {
  SetUpDeviceMappingAndAssignLocalDeviceOnce();
  TestDARDistributedTensor();
  return 0;
}

