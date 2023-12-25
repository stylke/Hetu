#pragma once

#include "hetu/common/macros.h"
#include "hetu/core/ndarray.h"
#include "hetu/graph/graph.h"
#include "hetu/graph/define_and_run_graph.h"
#include "hetu/graph/executable_graph.h"
#include "hetu/graph/tensor.h"
#include "hetu/graph/operator.h"
#include "hetu/graph/init/initializer.h"
#include "hetu/impl/communication/comm_group.h"

namespace hetu {
namespace graph {

using ExecGraphPair = std::pair<std::shared_ptr<ExecutableGraph>, std::shared_ptr<ExecutableGraph>>;
using Device2DTListPairMap = std::unordered_map<Device, std::pair<std::vector<Device>, std::vector<Tensor>>>;

std::ostream& operator<<(std::ostream& os, const SwitchExecGraph& switcher);

class ParamSlice {
  protected:
    friend class SwitchExecGraph;

  public:
    ParamSlice(const TensorName& block_name, const std::vector<int32_t>& slice_num): 
      _block_name(block_name),
      _slice_num(slice_num) {
    }

    const std::string name() const {
      std::string suffix = "_slice";
      for (const auto& x : _slice_num) {
        suffix += "_" + std::to_string(x);
      }
      return _block_name + suffix;
    }

    const Tensor& OwnedSliceInst(size_t idx) const {
      HT_ASSERT(idx < _owned_slice_instances.size())
        << "idx out of range";
      return _owned_slice_instances[0];
    }

    const Tensor& NeededSliceInst(size_t idx) const {
      HT_ASSERT(idx < _needed_slice_instances.size())
        << "idx out of range";
      return _needed_slice_instances[0];
    }

    void AddOwnedSliceInst(const Device& device, const Tensor& tensor) {
      if (!_owned_slice_instances.empty()) {
        const HTShape& shape = _owned_slice_instances[0]->shape();
        const auto shape_size = shape.size();
        HT_ASSERT(shape_size == tensor->shape().size())
          << "the new slice instance shape should be equal to the old slice instance shape, " 
          << "but the new is " << tensor->shape() << " and the old is " << shape;
        for(size_t i = 0; i < shape_size; ++i) {
          HT_ASSERT(shape[i] == tensor->shape(i))
            << "the new slice instance shape should be equal to the old slice instance shape, "  
            << "but the new is " << tensor->shape() << " and the old is " << shape;
        }
      }
      _owned_devices.push_back(device);
      _owned_slice_instances.push_back(tensor);
    }

    void AddNeededSliceInst(const Device& device, const Tensor& tensor) {
      HT_ASSERT(!_owned_slice_instances.empty())
        << "the slice isn't owned by any devices, "
        << "please ensure you've added a slice instance before";
      const HTShape& shape = _owned_slice_instances[0]->shape();
      const auto shape_size = shape.size();
      HT_ASSERT(shape_size == tensor->shape().size())
        << "the needed slice shape should be equal to the owned slice shape, " 
        << "but the needed is " << tensor->shape() << " and the owned is " << shape;
      for(size_t i = 0; i < shape_size; ++i) {
        HT_ASSERT(shape[i] == tensor->shape(i))
          << "the needed slice shape should be equal to the owned slice shape, " 
          << "but the needed is " << tensor->shape() << " and the owned is " << shape;
      }
      _needed_devices.push_back(device);
      _needed_slice_instances.push_back(tensor);
    }

    void ParamSliceComm(Device2DTListPairMap& send_mapping, Device2DTListPairMap& recv_mapping);

  protected:
    TensorName _block_name;
    // 在一个block中的slice编号
    // 例如block有3*2*5个slice
    // 那么一个合法的_slice_num就是{2,1,3}
    std::vector<int32_t> _slice_num; 

    TensorList _owned_slice_instances;
    TensorList _needed_slice_instances;
    std::vector<Device> _owned_devices;
    std::vector<Device> _needed_devices;

    size_t _round_robin = 0;
};

class ParamBlock {
  protected:
    friend class SwitchExecGraph;

  public:
    ParamBlock(const TensorName& block_name, const std::vector<int32_t>& block_shape):
      _block_name(block_name), 
      _block_shape(block_shape) {
    }

    const std::string name() const {
      return _block_name;
    }

    const std::vector<int32_t>& BlockShape() const {
      return _block_shape;
    }

    std::vector<std::shared_ptr<ParamSlice>>& GetParamSlices() {
      return _param_slices;
    }

    std::shared_ptr<ParamSlice>& GetParamSlice(const std::vector<int32_t>& slice_num) {
      size_t size = slice_num.size();
      HT_ASSERT(size == _block_shape.size() && size > 0) 
        << "size should be equal to block shape size and non-zero";
      size_t cnt = 1, sum = 0;
      for (int32_t i = size - 1; i >= 0; --i) {
        HT_ASSERT(slice_num[i] < _block_shape[i]) 
          << "slice_num dim " << i << " is out of range"
          << ", slice_num = " << slice_num << " and block_shape = " << _block_shape;
        sum += slice_num[i] * cnt;
        cnt *= _block_shape[i];
      }
      HT_ASSERT(sum < _param_slices.size()) 
        << "slice is out of range";
      return _param_slices[sum];
    }

    void ParamBlockComm(Device2DTListPairMap& send_mapping, Device2DTListPairMap& recv_mapping);

  protected:
    std::vector<int32_t> _block_shape;
    TensorName _block_name;
    std::vector<std::shared_ptr<ParamSlice>> _param_slices; 
};

class SwitchExecGraph {
  protected:
    friend class DefineAndRunGraph;
    friend class ExecutableGraph;

  public:
    SwitchExecGraph(DefineAndRunGraph* define_graph, size_t plan_before, size_t plan_after) {
      _define_graph = define_graph;
      _define_graph_params = define_graph->params();
      _switch_plan_pair = std::make_pair(plan_before, plan_after);
      _switch_graph_pair = std::make_pair(define_graph->GetPlan(plan_before).exec_graph, 
                                          define_graph->GetPlan(plan_after).exec_graph);
      /*
      _switch_graph_params_pair = std::make_pair(plan_before.exec_graph->params()),
                                                 plan_after.exec_graph->params());
      */
    }

    const ExecGraphPair& SwitchGraphPair() const {
      return _switch_graph_pair;
    }
    
    void SwitchParams();

  protected:
    void CreateParamBlock(ParamBlock& block,
                          std::vector<int32_t>& slice_num, 
                          const TensorName& block_name,
                          int32_t dim);

    void MakeAllParamSlices(const Tensor& param, ParamBlock& block, 
                            const Device& device, const DeviceGroup& group,
                            std::vector<int32_t>& slice_num, std::vector<int32_t>& slice_relative_num,
                            const std::unordered_map<int32_t, int32_t>& state,
                            const std::vector<int32_t>& multiple, int32_t dim);

    Tensor MergeAllParamSlices(const Tensor& param, ParamBlock& block, 
                               const Device& device, const DeviceGroup& group,
                               std::vector<int32_t>& slice_num, std::vector<int32_t>& slice_relative_num,
                               const std::unordered_map<int32_t, int32_t>& state,
                               const std::vector<int32_t>& multiple, int32_t dim);

    void SwitchParam(const DistributedStates& src_ds, const DeviceGroup& src_group,
                const DistributedStates& dst_ds, const DeviceGroup& dst_group,
                const Tensor& comm_input, const Tensor& after_param);

    void MakeCommGraph();

  protected:
    DefineAndRunGraph* _define_graph; // 定义图
    TensorCRefList _define_graph_params; // 定义图的params tensor
    std::pair<size_t, size_t> _switch_plan_pair; // 需要切换的两个exec graph plan的编号
    ExecGraphPair _switch_graph_pair; // 需要切换的两个exec graph的指针

    std::shared_ptr<ExecutableGraph> _comm_graph; // 为了应对切换过程中的复杂通信情况而建立的执行图 
    std::unordered_set<Device> _comm_set; // 参与通信图的所有devices
    OpRefList _comm_topo; // 该图的local_topo
    Tensor2ShapeMap _comm_shape_plan; // 该图所有tensor的运行时的shape
    FeedDict _comm_feed_dict; // 该图的输入
    Tensor2TensorMap _comm_feed_dict_mapping; // 该图的输入到before graph的映射
    TensorList _comm_results; // 该图通信的结果，与_define_graph_params一一对应
    Tensor2TensorMap _comm_results_mapping; // 该图的输出到after graph的映射
    TensorList _dummy_links; // 只有send没有recv时BatchedISendIRecvOp的输出dummy tensor需要被记录并在之后fetch

    Device2DTListPairMap _send_mapping; // 记录了每个device要send的(device, tensor)的pair
    Device2DTListPairMap _recv_mapping; // 记录了每个device要recv的(device, placeholder的tensor（之后会替换）)的pair
    std::vector<std::shared_ptr<ParamBlock>> _param_blocks; // 记录了graph所包含的所有的抽象ParamBlock
};

} // namespace graph
} // namespace hetu