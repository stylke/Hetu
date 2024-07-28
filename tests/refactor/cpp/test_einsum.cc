#include "hetu/graph/headers.h"
#include "hetu/graph/ops/op_headers.h"
#include "hetu/graph/optim/optimizer.h"

using namespace hetu;
using namespace hetu::graph;


void imperative_run(Graph& graph) {
  HT_LOG_INFO << "----------";
  Graph::push_graph_ctx(graph.id());
  std::unordered_map<std::string, HTShapeList>  test_args = {
    {"ij->ji",{{64, 32},}},
    // {"ij,ij->ij", {{64, 32}, {64, 32}}},
    // {"ii->i",{{64, 64},}},
    // {"...ij->...ji",{{64, 32, 4, 2, 4},}},
    // {"ij->",{{64, 32},}},
    // {"ij->j",{{64, 32},}},
    // {"ik,k",{{64, 32},{32,}}},
    // {"ik,kj",{{64, 32},{32, 16}}},
    // {"i,i",{{2,},{2,}}},
    // {"ij,ij",{{64, 32},{64, 32}}},
    // {"i,j",{{64, },{32, }}},
    // {"ijk,ikl->ijl",{{64, 32, 16}, {64, 16, 24}}},
    // {"pqrs,tuqvr->pstuv", {{4, 5, 6, 8}, {9, 7, 5, 13, 6}}},
    // {"ik,jkl,il->ij",{{64, 32}, {16, 32, 48}, {64, 48}}},
    // {"ijk",{{64, 32, 16},}},
    // {"b n h w, n d -> b d h w",{{64, 32, 8, 4}, {32, 16}}},
    // {"n d, n d -> n",{{64, 32}, {64, 32}}},
    // {"i d, j d -> i j",{{64, 32}, {48, 32}}},
    // {"b h i d, b h j d -> b h i j",{{64, 32, 4, 8}, {64, 32, 6, 8}}},
    // {"b h i j, b h j d -> b h i d",{{64, 32, 4, 8}, {64, 32, 8, 6}}},
    // {"b i d, b i j d -> b i j",{{64, 32, 4}, {64, 32, 8, 4}}},
    // {"b x i d, b j d -> b x i j",{{64, 32, 4, 8}, {64, 5, 8}}},
    // {"b x i j, b j d -> b x i d",{{64, 32, 4, 5}, {64, 5, 8}}},
    // {"hij, ijc->ihc",{{64, 32, 16}, {32, 16, 8}}},
    // {"rac,rab->rbc",{{64, 32, 4}, {64, 32, 7}}},
    // {"ra,rab->rb",{{64, 32}, {64, 32, 8}}},
    // {"qhc,khc->qkh",{{64, 32, 4}, {48, 32, 4}}},
    // {"nm, mrc->nrc",{{64, 32}, {32, 8, 6}}},
    // {"abc,adc->bdc",{{64, 32, 15}, {64, 13, 15}}},
    // {"dceb,cef->dbf",{{64, 32, 4, 8}, {32, 4, 13}}},
    // {"acb,ade->dceb",{{64, 32, 7}, {64, 15, 9}}},
    // {"qkc,ch->hqk",{{64, 32, 4}, {4, 13}}},
    // {"bhqk,bkhc->bqhc",{{64, 32, 4, 8}, {64, 8, 32, 7}}},
    // {"bqa,ahc->bqhc",{{64, 32, 8}, {8, 15, 9}}},
    // {"...lc, ...c -> ...l",{{64, 32, 7}, {64, 7}}},
    // {"...lc, ...lc -> ...l",{{64, 32, 7}, {64, 32, 7}}},
    // {"...id,...jd->...ij",{{64, 32, 4, 8}, {64, 32, 5, 8}}},
    // {"...klm,kmn->...kln",{{64, 32, 4, 8}, {32, 8, 11}}},
    // {"...ikl, ...jk -> ...ijl",{{64, 32, 4, 8}, {64, 15, 4}}},
    // {"...l,...l->...",{{64, 32, 17}, {64, 32, 17}}},
    // {"ijk,ijk...->ij...",{{64, 32, 4}, {64, 32, 4, 9}}},
    // {"bxi,oij,byj->boxy",{{64, 32, 5}, {17, 5, 13}, {64, 9, 13}}},
    // {"ijac,ijkp->ijakcp",{{64, 32, 4, 8}, {64, 32, 5, 7}}},
    // {"cdij,cbi->cdbj",{{64, 32, 4, 8}, {64, 19, 4}}},
    // {"bsid,bsjd->bijd",{{64, 32, 4, 8}, {64, 32, 17, 8}}},
    // {"bsid,bsje->bijde",{{64, 32, 4, 8}, {64, 32, 17, 9}}},
    // {"...bac,...dae->...bdce",{{64, 32, 4, 8}, {64, 19, 4, 5}}},
    // {"...abc,...adc->...bdc",{{64, 32, 4, 8}, {64, 32, 7, 8}}},
    // {"...qhd,...khd->...hqk",{{64, 32, 4, 8}, {64, 23, 4, 8}}},
    // {"...vhf,...qhv->...qhf",{{64, 32, 4, 8}, {64, 19, 4, 32}}},
    // {"...ij,jk->ik",{{64, 32, 4, 8}, {8, 13}}},
  };
  HT_LOG_INFO << graph.id();
  for (int i = 0; i < 1; i++) {
    for (auto it = test_args.begin(); it != test_args.end(); it++) {
      std::string msg = it->first;
      HTShapeList shapes = it->second;
      int length = shapes.size();
      TensorList inputs(length);
      for (int i = 0; i < length; ++i) {
        inputs[i] = MakeParameterOp(NormalInitializer(), shapes[i], kFloat32, true, DistributedStates(), OpMeta().set_name("x" + std::to_string(i))
                                                                                            .set_eager_device(Device(kCPU, 0)));
      }
      HT_LOG_INFO << msg;
      HT_LOG_INFO << inputs;
      auto pred = MakeEinsumOp(msg, inputs);
      HT_LOG_INFO << "Get Op";
      HT_LOG_INFO << inputs;
      SGDOptimizer optimizer(inputs, 0.1f);
      HT_LOG_INFO << "BEGIN GRAD";
      // auto pred = MakeReluOp(x);
      auto sumed_pred = MakeReduceOp(pred, ReductionType::MEAN);
      // auto train_op = optimizer.Minimize(sumed_pred)->get_or_compute();
      HT_LOG_INFO << "BEGIN GRAD";
      // TODO: zero_grad --> backward --> step
      hetu::impl::SynchronizeAllCPUStreams();
      HT_LOG_INFO << "pred = " << pred->get_or_compute();
      hetu::impl::SynchronizeAllCPUStreams();
    }
  }
}

void static_run(Graph& graph) {
  HT_LOG_INFO << "----------";
  Graph::push_graph_ctx(graph.id());
  auto w = MakeParameterOp(OnesInitializer(), {5,1}, kFloat32, true, DistributedStates(), OpMeta().set_name("w"));
  auto x = MakePlaceholderOp(NDArrayMeta().set_shape({10, 5}).set_dtype(kFloat32), DistributedStates(), OpMeta().set_name("x"));
  auto y = MakePlaceholderOp(NDArrayMeta().set_shape({10, 1}).set_dtype(kFloat32), DistributedStates(), OpMeta().set_name("y"));
  auto pred = MakeMatMulOp(x, w);
  // auto pred = MakeAvgPoolOp(x, 2, 2, 0, 2, OpMeta().set_name("pred"));
  // auto loss = MakeBCEOp(pred, y);
  auto loss = MakeMSELossOp(pred, y, ReductionType::MEAN);
  SGDOptimizer optimizer(0.1f);
  auto train_op = optimizer.Minimize(loss);
  for (size_t i = 0; i < 1; i++) {
    auto x_val = NDArray::full({10, 5}, i * 0.01, Device(kCUDA));
    auto y_val = NDArray::zeros({10,1}, Device(kCUDA));
    auto ret = graph.Run({loss, pred, w},
                         {{x->id(), NDArrayList({x_val})}, {y->id(), NDArrayList({y_val})}});
    HT_LOG_INFO << "loss = " << ret[0];
    HT_LOG_INFO << "pred = " << ret[1];
    HT_LOG_INFO << "w = " << ret[2];
  }
  hetu::impl::SynchronizeAllCPUStreams();
}

int main()
{
  imperative_run(Graph::get_default_eager_graph());
  // imperative_run(Graph::get_default_define_by_run_graph());
  static_run(Graph::get_default_define_and_run_graph());
  return 0;
}