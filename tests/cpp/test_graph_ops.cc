#include "hetu/graph/headers.h"
// #include "hetu/graph/ops/variable.h"
// #include "hetu/graph/ops/placeholder.h"
// #include "hetu/graph/ops/matmul.h"
// #include "hetu/graph/ops/binary_cross_entropy.h"
#include "hetu/graph/ops/op_headers.h"
#include "hetu/graph/optim/optimizer.h"
#include <unistd.h>

using namespace hetu;
using namespace hetu::graph;

// void imperative_run(Graph& graph) {
//   HT_LOG_INFO << "----------";
//   Graph::push_graph_ctx(graph.id());
//   auto w = MakeParameterOp(OnesInitializer(), {5,1}, kFloat32, true, OpMeta().set_name("w").set_eager_device(Device(kCPU, 0)));
//   SGDOptimizer optimizer(TensorList{w}, 0.1f);
//   for (int i = 0; i < 1; i++) {
//     auto x = MakeVariableOp(ConstantInitializer(i * 0.01), {10,5}, kFloat32, false, OpMeta().set_name("x").set_eager_device(Device(kCPU, 0)));
//     auto y = MakeVariableOp(ZerosInitializer(), {10, 1}, kFloat32, false, OpMeta().set_name("y").set_eager_device(Device(kCPU, 0)));
//     auto pred = MakeMatMulOp(x, w);
//     auto loss = MakeBCEOp(pred, y);
//     // TODO: zero_grad --> backward --> step
//     optimizer.Minimize(loss)->get_or_compute();
//     HT_LOG_INFO << "loss = " << loss->get_or_compute();
//     HT_LOG_INFO << "pred = " << pred->get_or_compute();
//     HT_LOG_INFO << "w = " << w->get_or_compute();
//   }
// }

// void static_run(Graph& graph) {
//   HT_LOG_INFO << "----------";
//   Graph::push_graph_ctx(graph.id());
//   auto w = MakeParameterOp(OnesInitializer(), {5,1}, kFloat32, true, OpMeta().set_name("w"));
//   auto x = MakePlaceholderOp(NDArrayMeta().set_shape({10,5}).set_dtype(kFloat32), OpMeta().set_name("x"));
//   auto y = MakePlaceholderOp(NDArrayMeta().set_shape({10,1}).set_dtype(kFloat32), OpMeta().set_name("y"));
//   auto pred = MakeMatMulOp(x, w);
//   auto loss = MakeBCEOp(pred, y);
//   SGDOptimizer optimizer(0.1f);
//   auto train_op = optimizer.Minimize(loss);
//   for (size_t i = 0; i < 1; i++) {
//     auto x_val = NDArray::full({10,5}, i * 0.01, Device(kCPU));
//     auto y_val = NDArray::zeros({10,5}, Device(kCPU));
//     auto ret = graph.Run({loss, pred, w, train_op},
//                          {{x->id(), x_val}, {y->id(), y_val}});
//     HT_LOG_INFO << "loss = " << ret[0];
//     HT_LOG_INFO << "pred = " << ret[1];
//     HT_LOG_INFO << "w = " << ret[2];
//   }
// }

struct PyOptimizer {
  SGDOptimizer optimizer;
};

void imperative_run(Graph& graph) {
  HT_LOG_INFO << "----------";
  Graph::push_graph_ctx(graph.id());
  auto w = MakeParameterOp(OnesInitializer(), {5,1}, kFloat32, true, DistributedStates(), OpMeta().set_name("w").set_eager_device(Device(kCPU, 0)));
  // SGDOptimizer optimizer(TensorList{w}, 0.1f);
  HT_LOG_INFO << graph.id();
  for (int i = 0; i < 1; i++) {
    auto x = MakeParameterOp(NormalInitializer(), {3, 4}, kFloat32, true, DistributedStates(), OpMeta().set_name("x").set_eager_device(Device(kCPU, 0)));
    auto y = MakeVariableOp(NormalInitializer(), {3, 2, 10, 5}, kFloat32, false, DistributedStates(), OpMeta().set_name("y").set_eager_device(Device(kCPU, 0)));
    // auto bias = MakeVariableOp(NormalInitializer(), {3, 1, 10, 10}, kFloat32, false, OpMeta().set_name("bias").set_eager_device(Device(kCPU, 0)));
    // auto label = MakeVariableOp(ZerosInitializer(), {64, 16}, kInt64, false, OpMeta().set_name("bias").set_eager_device(Device(kCPU, 0)));
    // // auto pred = MakeSubElewiseOp(x, y);
    // auto f = MakeVariableOp(NormalInitializer(), {4, 1, 2, 2}, kFloat32, 
    //                         false, OpMeta().set_name("f").set_eager_device(Device(kCPU, 0)));
    // auto scale = MakeVariableOp(NormalInitializer(), {2}, kFloat32, 
    //                         false, OpMeta().set_name("f").set_eager_device(Device(kCPU, 0)));
    // auto lbias = MakeVariableOp(NormalInitializer(), {2}, kFloat32, 
    //                         false, OpMeta().set_name("f").set_eager_device(Device(kCPU, 0)));
    // auto rmean = MakeVariableOp(NormalInitializer(), {2}, kFloat32, 
    //                         false, OpMeta().set_name("f").set_eager_device(Device(kCPU, 0)));
    // auto rvar = MakeVariableOp(NormalInitializer(), {2}, kFloat32, 
    //                         false, OpMeta().set_name("f").set_eager_device(Device(kCPU, 0)));
    auto pred = MakeTransposeOp(x, {1, 0}, OpMeta().set_name("pred").set_eager_device(Device(kCPU, 0)));
    // auto pred = MakeReluOp(x);
    auto sumed_pred = MakeReduceOp(pred, ReductionType::MEAN, {}, {false}, OpMeta().set_name("loss").set_eager_device(Device(kCPU, 0)));
    SGDOptimizer optimizer(TensorList{x}, 0.1f);
    PyOptimizer py;
    PyOptimizer* py0 = new PyOptimizer;
    py.optimizer = optimizer;
    py0->optimizer = optimizer;
    HT_LOG_INFO << "BEGIN GRAD";
    auto train_op = optimizer.Minimize(sumed_pred);
    auto train_op2 = py.optimizer.Minimize(sumed_pred);
    auto train_op3 = py0->optimizer.Minimize(sumed_pred);
    // // TODO: zero_grad --> backward --> step
    HT_LOG_INFO << "x = " << x->get_or_compute();
    // HT_LOG_INFO << "y = " << y->get_or_compute();
    HT_LOG_INFO << "pred = " << pred->get_or_compute();
    hetu::impl::SynchronizeAllCPUStreams();
    // sleep(2);
    delete py0;

  }
}

void static_run(Graph& graph) {
  HT_LOG_INFO << "----------";
  Graph::push_graph_ctx(graph.id());
  for (size_t i = 0; i < 2; i++) {
    auto w = MakeParameterOp(OnesInitializer(), {5,1}, kFloat32, true, DistributedStates(), OpMeta().set_name("w"));
    auto x = MakePlaceholderOp(NDArrayMeta().set_shape({10, 5}).set_dtype(kFloat32), DistributedStates(), OpMeta().set_name("x"));
    auto y = MakePlaceholderOp(NDArrayMeta().set_shape({10, 1}).set_dtype(kFloat32), DistributedStates(), OpMeta().set_name("y"));
    auto pred = MakeMatMulOp(x, w);
    // auto pred = MakeAvgPoolOp(x, 2, 2, 0, 2, OpMeta().set_name("pred"));
    // auto loss = MakeBCEOp(pred, y);
    auto loss = MakeMSELossOp(pred, y, ReductionType::MEAN);
    SGDOptimizer optimizer(0.1f);
    auto train_op = optimizer.Minimize(loss);
    auto x_val = NDArray::full({10, 5}, i * 0.01, Device(kCPU));
    auto y_val = NDArray::zeros({10,1}, Device(kCPU));
    auto ret = graph.Run({loss, pred, w},
                         {{x->id(), NDArrayList({x_val})}, {y->id(), NDArrayList({y_val})}});
    HT_LOG_INFO << "loss = " << ret[0];
    HT_LOG_INFO << "pred = " << ret[1];
    HT_LOG_INFO << "w = " << ret[2];
  }
}

int main()
{
  HT_LOG_INFO << Graph::get_default_eager_graph().num_ops();
  imperative_run(Graph::get_default_eager_graph());
  // HT_LOG_INFO << Graph::get_default_eager_graph().num_ops();
  // imperative_run(Graph::get_default_define_by_run_graph());
  static_run(Graph::get_default_define_and_run_graph());
  return 0;
}