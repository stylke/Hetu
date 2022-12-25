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
#include <cmath>

using namespace hetu;
using namespace hetu::autograd;
using namespace hetu::execution;
using namespace hetu::impl;
using namespace hetu::impl::comm;

void TestDARPipelineParallelMLP(DataType dtype = kFloat32,
                                const HTShape& dims = {256, 128, 64, 16, 1}) {
  HT_LOG_INFO << "Testing pipeline-parallel MLP in define-and-run mode";
  HT_ASSERT(dims.back() == 1) << "Label size should be 1";
  auto local_device = GetLocalDevice();
  auto all_devices = GetGlobalDeviceGroup();
  HT_ASSERT(dims.size() > all_devices.num_devices())
    << "Cannot split " << dims.size() - 1 << " layers to "
    << all_devices.num_devices() << " devices";

  auto& first_stage_device = all_devices.get(0);
  auto& last_stage_device = all_devices.get(all_devices.num_devices() - 1);
  DeviceGroup first_stage_device_group({first_stage_device});
  DeviceGroup last_stage_device_group({last_stage_device});
  auto x = PlaceholderOp(dtype, {-1, dims[0]},
                         OpMeta().set_device_group(first_stage_device_group))
             ->output(0);
  auto y = PlaceholderOp(dtype, {-1, 1},
                         OpMeta().set_device_group(last_stage_device_group))
             ->output(0);
  auto act = x;
  int num_layers_per_stage = DIVUP(dims.size() - 1, all_devices.num_devices());
  for (size_t i = 1; i < dims.size(); i++) {
    int stage_id = (i - 1) / num_layers_per_stage;
    auto& stage_device = all_devices.get(stage_id);
    DeviceGroup stage_device_group({stage_device});
    auto var = VariableOp({dims[i - 1], dims[i]}, HeUniformInitializer(), dtype,
                          true, OpMeta().set_device_group(stage_device_group))
                 ->output(0);
    act = MatMulOp(act, var, false, false,
                   OpMeta().set_device_group(stage_device_group))
            ->output(0);
    if (i + 1 < dims.size())
      act = ReluOp(act)->output(0);
  }
  auto prob = SigmoidOp(act)->output(0);
  auto loss = BinaryCrossEntropyOp(prob, y, "mean")->output(0);
  SGDOptimizer optimizer(0.1, 0.0);
  auto train_op = optimizer.Minimize(loss);

  DARExecutor exec(local_device, all_devices, {loss});

  NDArray features =
    NDArray::randn({1024, dims[0]}, local_device, dtype, 0.0, 1.0, 2022);
  NDArray labels = NDArray::ones({1024, 1}, local_device, dtype);

  SynchronizeAllStreams();
  FeedDict feed_dict = {{x->id(), features}, {y->id(), labels}};

  int num_micro_batches = 8;

  // if (local_device == all_devices.get(0))
  //   HT_LOG_INFO << "features: " << features << ", labels: " << labels << ",
  //   num_micro_batches: " << num_micro_batches;

  // warmup
  for (int i = 0; i < 10; i++)
    exec.Run({prob, loss, train_op}, feed_dict, num_micro_batches);

  TIK(train);
  for (int i = 0; i < 1000; i++)
    exec.Run({prob, loss, train_op}, feed_dict, num_micro_batches);
  TOK(train);
  HT_LOG_INFO << "Train 1000 iter cost " << COST_MSEC(train) << " ms";

  TIK(eval);
  for (int i = 0; i < 1000; i++)
    exec.Run({prob}, feed_dict, num_micro_batches);
  TOK(eval);
  HT_LOG_INFO << "Infer 1000 iter cost " << COST_MSEC(eval) << " ms";
}

int main(int argc, char** argv) {
  SetUpDeviceMappingAndAssignLocalDeviceOnce();
  TestDARPipelineParallelMLP();
  return 0;
}