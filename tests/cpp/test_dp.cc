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

void TestDARDataParallelMLP(DataType dtype = kFloat32,
                            const HTShape& dims = {256, 64, 16, 1}) {
  HT_LOG_INFO << "Testing data-parallel MLP in define-and-run mode";
  HT_ASSERT(dims.back() == 1) << "Label size should be 1";
  auto local_device = GetLocalDevice();
  auto all_devices = GetGlobalDeviceGroup();

  auto x = PlaceholderOp(dtype, {-1, dims[0]})->output(0);
  auto y = PlaceholderOp(dtype, {-1, 1})->output(0);
  auto act = x;
  for (size_t i = 1; i < dims.size(); i++) {
    auto var =
      VariableOp({dims[i - 1], dims[i]}, HeUniformInitializer(), dtype, true)
        ->output(0);
    act = MatMulOp(act, var)->output(0);
    if (i + 1 < dims.size())
      act = ReluOp(act)->output(0);
  }
  auto prob = SigmoidOp(act)->output(0);
  auto loss = BinaryCrossEntropyOp(prob, y, "mean")->output(0);
  SGDOptimizer optimizer(0.1, 0.0);
  auto train_op = optimizer.Minimize(loss);

  DARExecutor exec(local_device, all_devices);

  NDArray features = NDArray::randn({1024, dims[0]}, local_device, dtype);
  NDArray labels = NDArray::zeros({1024, 1}, local_device, dtype);
  SynchronizeAllStreams();
  FeedDict feed_dict = {{x->id(), features}, {y->id(), labels}};

  // warmup
  for (int i = 0; i < 10; i++)
    exec.Run({train_op}, feed_dict);

  TIK(train);
  for (int i = 0; i < 1000; i++)
    exec.Run({prob, loss, train_op}, feed_dict);
  TOK(train);
  HT_LOG_INFO << "Train 1000 iter cost " << COST_MSEC(train) << " ms";

  TIK(eval);
  for (int i = 0; i < 1000; i++)
    exec.Run({prob}, feed_dict);
  TOK(eval);
  HT_LOG_INFO << "Infer 1000 iter cost " << COST_MSEC(eval) << " ms";
}

void TestDBRDataParallelMLP(DataType dtype = kFloat32,
                            const HTShape& dims = {256, 64, 16, 1}) {
  HT_LOG_INFO << "Testing data-parallel MLP in define-by-run mode";
  HT_ASSERT(dims.back() == 1) << "Label size should be 1";
  auto local_device = GetLocalDevice();
  auto all_devices = GetGlobalDeviceGroup();

  TensorList vars;
  for (size_t i = 1; i < dims.size(); i++) {
    auto var =
      VariableOp({dims[i - 1], dims[i]}, HeUniformInitializer(), dtype, true)
        ->output(0);
    vars.push_back(var);
  }
  SGDOptimizer optimizer(vars, 0.1, 0);

  NDArray features = NDArray::randn({1024, dims[0]}, local_device, dtype);
  NDArray labels = NDArray::zeros({1024, 1}, local_device, dtype);
  SynchronizeAllStreams();

  auto fn = [&](bool training) {
    auto x = VariableOp(features, false)->output(0);
    auto y = VariableOp(labels, false)->output(0);
    auto act = x;
    for (size_t i = 1; i < dims.size(); i++) {
      ;
      act = MatMulOp(act, vars[i - 1])->output(0);
      if (i + 1 < dims.size())
        act = ReluOp(act)->output(0);
    }
    auto prob = SigmoidOp(act)->output(0);
    if (training) {
      auto loss = BinaryCrossEntropyOp(prob, y, "mean")->output(0);
      optimizer.ZeroGrad();
      loss->Backward();
      optimizer.Step();
    } else {
      prob->GetOrCompute();
    }
  };

  // warmup
  for (int i = 0; i < 10; i++)
    fn(true);

  TIK(train);
  for (int i = 0; i < 1000; i++)
    fn(true);
  TOK(train);
  HT_LOG_INFO << "Train 1000 iter cost " << COST_MSEC(train) << " ms";

  // warmup
  for (int i = 0; i < 10; i++)
    fn(false);

  TIK(eval);
  for (int i = 0; i < 1000; i++)
    fn(false);
  TOK(eval);
  HT_LOG_INFO << "Infer 1000 iter cost " << COST_MSEC(eval) << " ms";
}

int main(int argc, char** argv) {
  SetUpDeviceMappingAndAssignLocalDeviceOnce();
  TestDARDataParallelMLP();
  TestDBRDataParallelMLP();
  return 0;
}
