#include "hetu/core/ndarray.h"
#include "hetu/execution/dar_executor.h"
#include "hetu/execution/dbr_executor.h"
#include "hetu/impl/communication/comm_group.h"
#include "hetu/autograd/optim/Optimizer.h"
#include "hetu/autograd/distributed_states.h"
#include "hetu/autograd/ops/op_headers.h"
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
  
  int n = 2; // 每个device最开始各自分到的数据数量, data.shape=[n, dim], 如果有4个device, 则总data.shape=[n*4, dim]
  int dim = 4; 

  DistributedStates ds0(4, {{-2, 1}, {-1, 4}}, {-1});
  DistributedStates ds1(4, {{-2, 1}, {-1, 1}, {0, 4}}, {0});

  auto tensor1 = PlaceholderOp(dtype, {n, dim}, OpMeta().set_device_group(all_device_group).set_name("tensor1"))->output(0);
  tensor1->set_distributed_states(ds1); // tensor1: ds1: (2,4)*4

  auto w1_data = (NDArray::rand({dim, dim}, Device(kCPU), kFloat32, 0.0, 1.0, 2023) - 0.5) * (2.0 / std::sqrt(dim));
  auto w1 = VariableOp(w1_data, true, OpMeta().set_device_group(all_device_group).set_name("w1"))->output(0);
  w1->set_distributed_states(ds0); // duplicate: w=(4,4)

  auto tensor2 = MatMulOp(tensor1, w1, false, false, OpMeta().set_name("MM1"))->output(0);

  DistributedStates ds3(4, {{-2, 1}, {-1, 2}, {0, 2}}, {0, -1}); // mp=2(横切), dp=2
  auto tensor3 = CommOp(tensor2, ds3, OpMeta().set_name("comm_op1"))->output(0); 

  auto w2_data = (NDArray::rand({dim, dim/2}, Device(kCPU), kFloat32, 0.0, 1.0, 2023+1+local_device.index()%2) - 0.5) * (2.0 / std::sqrt(dim));
  auto w2 = VariableOp(w2_data, true, OpMeta().set_device_group(all_device_group).set_name("w2"))->output(0);
  DistributedStates ds2(4, {{-2, 1}, {-1, 2}, {1, 2}}, {-1, 1});
  w2->set_distributed_states(ds2); // dp=2, mp=2(竖切)  

  auto result = MatMulOp(tensor3, w2, false, false, OpMeta().set_name("MM2"))->output(0); // ds4

  DistributedStates ds4(4, {{-2, 1}, {-1, 1}, {0, 2}, {1, 2}}, {0, 1});
  auto y = PlaceholderOp(dtype, {n*2, dim/2}, OpMeta().set_device_group(all_device_group).set_name("y"))->output(0);  
  y->set_distributed_states(ds4);
  auto loss = BinaryCrossEntropyOp(result, y)->output(0);

  SGDOptimizer optimizer(0.1, 0.0);
  HT_LOG_INFO << local_device << ": optimizer.minimize begin!";
  auto train_op = optimizer.Minimize(loss);
  HT_LOG_INFO << local_device << ": optimizer.minimize end!";

  HT_LOG_INFO << local_device << ": create executor begin!";
  DARExecutor exec(local_device, all_devices, {result, train_op});
  HT_LOG_INFO << local_device << ": create executor end!";

  NDArray data = NDArray::randn({n, dim}, local_device, dtype, 0.0, 1.0, (666 + all_device_group.get_index(local_device))); // ds1->ds2
  NDArray labels = NDArray::zeros({n*2, dim/2}, local_device, dtype);

  SynchronizeAllStreams();
  
  FeedDict feed_dict = {{tensor1->id(), data}, {y->id(), labels}};

  //
  TensorList weights;
  OpList weights_op;
  HT_LOG_DEBUG << local_device << ": get topo order begin!";
  auto topo = TopoSort(train_op);
  HT_LOG_DEBUG << local_device << ": get topo order end!";
  HT_LOG_DEBUG << local_device << ": final topo: " << topo;
  
  for (auto op : topo) {
    if (is_variable_op(op)) {
      weights.push_back(op->output(0));
      weights_op.push_back(op);
    }
  }
  weights.push_back(train_op);
  for (size_t i = 0; i < weights.size()-1; i++) {
    HT_LOG_INFO << local_device << ": init weight: " << weights[i] << ", value: " << reinterpret_cast<VariableOp&>(weights_op[i])->data();
  }
  //  

  // auto r = exec.Run({result, train_op}, feed_dict);
  // HT_LOG_INFO << local_device << ": result = " << r[0];

  auto results = exec.Run(weights, feed_dict);
  for (size_t i = 0; i < weights.size()-1; i++) {
    if (results[i] != Tensor())
      HT_LOG_INFO << local_device << ": updated weight: " << weights[i] << ", value: " << results[i];
  }    
}

int main(int argc, char** argv) {
  SetUpDeviceMappingAndAssignLocalDeviceOnce();
  TestDARDistributedTensor();
  return 0;
}

