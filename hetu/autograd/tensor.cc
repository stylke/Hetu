#include "hetu/autograd/tensor.h"
#include "hetu/execution/dbr_executor.h"
#include "hetu/autograd/autograd.h"
#include "hetu/autograd/ops/Variable.h"
#include "queue"

namespace hetu {
namespace autograd {

Tensor::Tensor(const TensorName& name, int32_t output_id,
               const NDArrayMeta& meta)
: shared_ptr_wrapper<TensorDef>() {
  _ptr = make_ptr<TensorDef>(TensorDef::constrcutor_access_key(), name,
                             output_id, meta);
}

NDArray& TensorDef::GetOrCompute() {
  if (!is_computed()) {
    auto& exec = hetu::execution::GetOrCreateDBRExecutor();
    if (is_tensor())
      exec.Run({producer()->output(output_id())});
    else
      exec.Run({producer()->out_dep_linker()});
    HT_ASSERT(is_computed()) << "Tensor " << name() << " is not properly "
                             << "marked as computed.";
  }
  return _data;
}

bool TensorDef::is_variable() const {
  return is_variable_op(producer()) && is_tensor();
}

Tensor TensorDef::to_variable(bool trainable, const OpMeta& op_meta) {
  HT_ASSERT(is_tensor()) << "Cannot detach dependency linker as a variable";
  if (is_variable()) {
    set_trainable(trainable);
    return get_self();
  } else {
    return VariableOp(GetOrCompute(), trainable,
                      OpMeta::Merge(producer()->op_meta(), op_meta))
      ->output(0);
  }
}

bool TensorDef::is_trainable() const {
  return is_trainable_op(producer()) && is_tensor();
}

void TensorDef::set_trainable(bool trainable) {
  HT_ASSERT(is_variable()) << "Cannot set non-variable tensors as trainable";
  reinterpret_cast<VariableOp&>(producer())->set_trainable(trainable);
}

void TensorDef::reset_initializer(const Initializer& init) {
  HT_ASSERT(is_variable())
    << "Cannot reset initializers for non-variable tensors";
  reinterpret_cast<VariableOp&>(producer())->reset_initializer(init);
}

void TensorDef::reset_data(const NDArray& data) {
  HT_ASSERT(is_variable()) << "Cannot reset data for non-variable tensors";
  reinterpret_cast<VariableOp&>(producer())->reset_data(data);
}

Tensor& TensorDef::Gradient() {
  return _grad;
}

const Tensor& TensorDef::Gradient() const {
  return _grad;
}

void TensorDef::AccumulateGrad(const Tensor& grad) {
  if (!is_trainable()) {
    HT_LOG_WARN << "Trying to update a non-trainable variable. Will ignore it";
    return;
  }

  HT_ASSERT(grad->shape() == shape())
    << "Gradient shape " << grad->shape() << " does not match variable shape "
    << shape();
  if (_grad.is_defined()) {
    // // TODO: add in place
    // _grad = AddElewiseOp(_grad, grad)->output(0);
    HT_NOT_IMPLEMENTED << "Gradient accumulation not implemented";
  } else {
    _grad = grad;
  }
}

void TensorDef::ZeroGrad() {
  // Question: support set as zeros?
  if (_grad.is_defined())
    _grad = Tensor();
}

void TensorDef::Backward(const Tensor& grad) {
  // Question: should we forbid calling `Backward` twice? If yes, then how?
  HT_ASSERT(is_tensor()) << "Cannot call \"Backward\" for a non-Tensor output";
  const auto& producer_op = producer();
  const auto& self = producer()->output(output_id());
  auto topo_order = TopoSort(producer_op);
  TensorList vars;
  vars.reserve(topo_order.size());
  for (auto& op : topo_order)
    if (is_trainable_op(op))
      vars.push_back(op->output(0));
  TensorList grads = Gradients(self, vars, grad);
  HT_ASSERT_EQ(grads.size(), vars.size())
    << "Only " << grads.size() << " gradients are returned for " << vars.size()
    << " variables.";
  for (size_t i = 0; i < grads.size(); i++) {
    vars[i]->AccumulateGrad(grads[i]);
  }
}

Tensor& TensorDef::get_self() {
  if (is_tensor())
    return producer()->output(output_id());
  else
    return producer()->out_dep_linker();
}

const Tensor& TensorDef::get_self() const {
  if (is_tensor())
    return producer()->output(output_id());
  else
    return producer()->out_dep_linker();
}

void TensorDef::SetProducer(Operator& op) {
  HT_ASSERT(_producer == nullptr) << "Try to set the producer twice";
  _producer = std::make_unique<Operator>(op);
}

void TensorDef::AddConsumer(Operator& op) {
  _consumers.push_back(op);
}

const Operator& TensorDef::producer() const {
  HT_ASSERT(_producer != nullptr && (*_producer).is_defined())
    << "Producer is not set yet";
  return *_producer;
}

Operator& TensorDef::producer() {
  HT_ASSERT(_producer != nullptr && (*_producer).is_defined())
    << "Producer is not set yet";
  return *_producer;
}

size_t TensorDef::num_consumers() const {
  return _consumers.size();
}

const Operator& TensorDef::consumer(size_t i) const {
  return _consumers[i];
}

Operator& TensorDef::consumer(size_t i) {
  return _consumers[i];
}

std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
  if (tensor.is_defined())
    os << tensor->name();
  else
    os << "Tensor()";
  return os;
}

// for distributed states

std::unordered_map<int32_t, int32_t> DistributedStates::combine_states(std::pair<std::vector<int32_t>, int32_t>& src2dst) {
  auto states = std::unordered_map<int32_t, int32_t>(_states);
  auto src = src2dst.first;
  auto dst = src2dst.second;
  int32_t value = 1;
  // 1. erase src
  for (auto s : src) {
    // HT_ASSERT(states.find(s) != states.end()) << "src " << s << " must in the states.keys !";
    HT_ASSERT(s != dst) << "cannot combine src to the same dst dimension " << s;
    if (s == -2 || s == -1) { // partial or duplicate
      value *= states[s];
      states[s] = 1;
    } else { // normal dimension
      if (states.find(s) != states.end()) {
        value *= states[s];
        states.erase(s); // 普通的dimension, value=1时就从map里删除
      }
      // dimension after s: move forward 1 offset. why?
      std::vector<int32_t> keys; 
      keys.reserve(states.size());
      for (auto kv : states) {
        if (kv.first >= 0) {
          keys.push_back(kv.first);
        }
      }
      std::sort(keys.begin(), keys.end());
      for (auto key : keys) {
        if (key > s) {
          auto val = states[key];
          states.erase(key);
          states[key - 1] = val;
        }
      }
    }
  }
  // 2. merge to dst
  if (dst == -2 || dst == -1) {
    states[dst] *= value;
  } else {
    for (auto s : src) { // 和前面往前挪一位的操作相对应
      if (s >= 0 && dst > s) {
        dst -= 1;
      }
    }
    if (states.find(dst) == states.end()) {
      states[dst] = value;
    } else {
      states[dst] *= value;
    }
  }

  return states;
}

// 合并必须保证[src]+dst的dimension是连续的
std::vector<int32_t> DistributedStates::combine_order(std::pair<std::vector<int32_t>, int32_t>& src2dst) {
  auto order = std::vector<int32_t>(_order);
  auto src = src2dst.first;
  auto dst = src2dst.second;
  std::vector<int32_t> inds;
  auto collect_safe_index = [&](int32_t dim) {
    auto it = std::find(order.begin(), order.end(), dim);
    if (it != order.end()) {
      auto ind = std::distance(order.begin(), it);
      inds.push_back(ind);
    } else {
      HT_LOG_INFO << "dimension " << dim << " is not in order !";
    }
  };
  for (auto s : src) {
    collect_safe_index(s);
  }
  collect_safe_index(dst);
  std::sort(inds.begin(), inds.end());
  if (inds.size() > 0) {
    for (size_t i = 1; i < inds.size(); i++) {
      HT_ASSERT(inds[i] == inds[0] + i) << "Cannot combine dimensions not adjacent!"; // 要combine的inds必须连续?
    }
    for (auto it = inds.rbegin(); it != inds.rend(); ++it) {
      if (it + 1 != inds.rend()) {
        order.erase(order.begin() + *it);
      } else {
        order[*it] = dst;
      }
    }
    for (size_t i = 0; i < order.size(); i++) {
      if (order[i] > 0) {
        for (auto s : src) { // 通过src消除掉多少dimension, 在src后面的dimension就往前挪多少位
          if (s >= 0 && order[i] > s) {
            order[i] -= 1;
          }
        }
      }
    }
  }

  return order;
}
  
bool DistributedStates::equal_states_and_order(std::unordered_map<int32_t, int32_t>& states1, std::vector<int32_t>& order1,
                                               std::unordered_map<int32_t, int32_t>& states2, std::vector<int32_t>& order2) {
  return (states1 == states2) && (order1 == order2);                              
}

bool DistributedStates::check_combine(DistributedStates& dst_distributed_states,
                              std::pair<std::vector<int32_t>, int32_t>& src2dst) {
  auto states = combine_states(src2dst);
  auto order = combine_order(src2dst);

  auto dst_states = dst_distributed_states.get_states();
  auto dst_order = dst_distributed_states.get_order();
  return equal_states_and_order(states, order, dst_states, dst_order);
}

std::unordered_map<int32_t, int32_t> DistributedStates::reduce_states(int dim) {
  auto states = std::unordered_map<int32_t, int32_t>(_states);
  if (dim == -2 || dim == -1) {
    states[dim] = 1;
  } else if (states.find(dim) != states.end()) {
    states.erase(dim);
  }
  return states;
}

std::vector<int32_t> DistributedStates::reduce_order(int dim) {
  auto order = std::vector<int32_t>(_order);
  auto it = std::find(order.begin(), order.end(), dim); 
  if (it != order.end()) {
    order.erase(it);
  }
  return order;
}

bool DistributedStates::check_reduce_dim(DistributedStates& dst_distributed_states, int dim) {
  auto states = reduce_states(dim);
  auto order = reduce_order(dim);

  auto dst_states = dst_distributed_states.get_states();
  auto dst_order = dst_distributed_states.get_order();
  return equal_states_and_order(states, order, dst_states, dst_order);                                 
}

bool DistributedStates::check_allreduce(DistributedStates& dst_distributed_states) {
  std::pair<std::vector<int32_t>, int32_t> src2dst = {{-2}, -1};
  return _states[-2] > 1 && check_combine(dst_distributed_states, src2dst);
}

bool DistributedStates::check_allgather(DistributedStates& dst_distributed_states) {
  std::pair<std::vector<int32_t>, int32_t> src2dst = {{0}, -1};
  return _states[0] > 1 && check_combine(dst_distributed_states, src2dst);
}

bool DistributedStates::check_reducescatter(DistributedStates& dst_distributed_states) {
  std::pair<std::vector<int32_t>, int32_t> src2dst = {{-2}, 0};
  return _states[-2] > 1 && check_combine(dst_distributed_states, src2dst);
}

bool DistributedStates::check_boradcast(DistributedStates& dst_distributed_states) {
  return dst_distributed_states.get_states()[-1] > 1 && dst_distributed_states.check_reduce_dim(*this, -1);
}

bool DistributedStates::check_reduce(DistributedStates& dst_distributed_states) {
  return _states[-2] > 1 && check_reduce_dim(dst_distributed_states, -2);
}

int32_t DistributedStates::get_dim(int32_t index) {
  if (index == -2 || index == -1 || _states.find(index) != _states.end()) {
    return _states[index];
  } else {
    return 1;
  }
}

std::vector<int32_t> DistributedStates::get_loop_sizes() {
  std::vector<int32_t> loop_sizes = {1};
  for (auto it = _order.rbegin(); it != _order.rend(); it++) {
    auto tmp_size = loop_sizes[0] * get_dim(*it);
    loop_sizes.insert(loop_sizes.begin(), tmp_size);
  }
  loop_sizes.erase(loop_sizes.begin());
  return loop_sizes;
}

std::unordered_map<int32_t, int32_t> DistributedStates::map_device_to_state_index(int32_t device_index) {
  std::unordered_map<int32_t, int32_t> state_index;
  for (auto it = _order.rbegin(); it != _order.rend(); it++) {
    int32_t cur_order = *it;
    int32_t cur_state = _states[cur_order];
    state_index[cur_order] = device_index % cur_state;
    device_index /= cur_state;
  }
  return state_index;
}

StateNode* DistributedStates::build_state_tree() {
  // HT_ASSERT(max_dimension >= _max_dimension) << "max dimension " << max_dimension << " must >= " << _max_dimension;
  std::unordered_map<int32_t, std::vector<StateNode*>> dimension_to_layer_nodes;
  StateNode* root = new StateNode();  
  std::queue<StateNode*> q; // 不用递归, 直接迭代完成, 每一轮q里只有上一维dimension里的root节点, 需要创建的是这一维dimension的节点
  q.push(root);
  for (int32_t dimension = -2; dimension <= _max_dimension; dimension++) {
    if (!q.empty()) {
      for (size_t cur_root_idx = 0; cur_root_idx < q.size(); cur_root_idx++) {
        StateNode* cur_root = q.front();
        q.pop();
        std::vector<StateNode*> childs;
        if (dimension >= 0 && _states.find(dimension) == _states.end()) {
          StateNode* p = new StateNode(0, 1, dimension, cur_root);
          childs.push_back(p);
          q.push(p);
        } else {
          for (size_t i = 0; i < _states[dimension]; i++) {
            StateNode* p = new StateNode(i, _states[dimension], dimension, cur_root);
            childs.push_back(p);
            q.push(p);
          }
        }
        cur_root->_childs = childs;
        dimension_to_layer_nodes[dimension].insert(dimension_to_layer_nodes[dimension].end(), childs.begin(), childs.end());
      }
    }
  }
  _state_tree = root;
  _dimension_to_layer_nodes = dimension_to_layer_nodes;
  return root;
}

std::unordered_map<int32_t, StateNode*> DistributedStates::map_device_to_state_node() {
  // HT_ASSERT(max_dimension >= _max_dimension) << "max dimension " << max_dimension << " must >= " << _max_dimension;  
  std::unordered_map<int32_t, StateNode*> device_to_state_node;
  for (size_t device_index = 0; device_index < _device_num; device_index++) {
    auto state_index = map_device_to_state_index(device_index);
    StateNode* p = _state_tree;
    
    for (int32_t dimension = -2; dimension <= _max_dimension; dimension++) {
      // state_index只会保存states中partial/duplicate和切分份数>=2的dimension, 其余dimension不做切分, 默认就是完整的一块, 其index=0
      if (state_index.find(dimension) != state_index.end()) {
        p = p->_childs[state_index[dimension]];
      } else {
        p = p->_childs[0];
      }
    }
    device_to_state_node[device_index] = p;
  }
  _device_to_state_node = device_to_state_node;
  return device_to_state_node;
}

std::vector<std::unordered_map<int32_t, ConvertMethod>> DistributedStates::get_methods_for_convert_trees(DistributedStates& dst_distributed_states) {
  HT_ASSERT(_device_num == dst_distributed_states._device_num) << "dst device num " 
            << dst_distributed_states._device_num << " must be equal to " << _device_num;
  // int32_t max_dimension = std::max(_max_dimension, dst_distributed_states._max_dimension);
  int32_t min_dimension = std::min(_max_dimension, dst_distributed_states._max_dimension);

  StateNode* src_tree = build_state_tree();
  std::unordered_map<int32_t, std::vector<StateNode*>> src_dimension_to_layer_nodes = _dimension_to_layer_nodes;
  std::unordered_map<int32_t, StateNode*> src_device_to_state_node = map_device_to_state_node();
  
  StateNode* dst_tree = dst_distributed_states.build_state_tree();
  std::unordered_map<int32_t, std::vector<StateNode*>> dst_dimension_to_layer_nodes = dst_distributed_states._dimension_to_layer_nodes;
  std::unordered_map<int32_t, StateNode*> dst_device_to_state_node = dst_distributed_states.map_device_to_state_node();  

  std::vector<std::unordered_map<int32_t, ConvertMethod>> dimension_to_methods_for_convert_trees;
  
  // 遍历每一层dimension layer
  for (int32_t dimension = min_dimension; dimension >= -2; dimension--) {
    // 遍历该dimension layer中每个device的state, 对比src和dst, 判断要插入什么样的convert method
    for (size_t device_index = 0; device_index < _device_num; device_index++) {
      StateNode* src_cur_device_state_node = src_device_to_state_node[device_index];
      StateNode* dst_cur_device_state_node = dst_device_to_state_node[device_index];
      // 判断是否要split
      // 目前只支持单个维度的split/allgather的case
      int32_t src_range = src_cur_device_state_node->_range;
      int32_t src_index = src_cur_device_state_node->_index;
      int32_t dst_range = dst_cur_device_state_node->_range;
      int32_t dst_index = dst_cur_device_state_node->_index;

      // if (src_range == 1) 
    }
  }
}

} // namespace autograd
} // namespace hetu
