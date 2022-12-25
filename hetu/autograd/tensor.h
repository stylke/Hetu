#pragma once

#include "hetu/core/ndarray.h"
#include "hetu/autograd/common.h"
#include "hetu/autograd/op_meta.h"
#include "hetu/autograd/init/initializer.h"
#include "hetu/utils/shared_ptr_wrapper.h"
#include <functional>

namespace hetu {
namespace autograd {

/******************************************************
 * Tensor Definition
 ******************************************************/

class Tensor : public shared_ptr_wrapper<TensorDef> {
 public:
  Tensor() = default;
  Tensor(const TensorName& name, int32_t output_id,
         const NDArrayMeta& meta = {});
};

class TensorDef : public shared_ptr_target {
 protected:
  friend class OperatorDef;
  template <typename T>
  friend class OpWrapper;
  friend class Tensor;
  struct constrcutor_access_key {};

 public:
  TensorDef(const constrcutor_access_key&, const TensorName& name,
            int32_t output_id, const NDArrayMeta& meta = {}, DistributedStates distributed_states = {})
  : _id{_next_tensor_id()}, _name(name), _output_id(output_id), _meta(meta), _distributed_states(distributed_states) {}

  ~TensorDef() = default;

  // disable copy constructor and move constructor
  TensorDef(const TensorDef&) = delete;
  TensorDef& operator=(const TensorDef&) = delete;
  TensorDef(TensorDef&&) = delete;
  TensorDef& operator=(TensorDef&&) = delete;

  TensorId id() const {
    return _id;
  }

  TensorName name() const {
    return _name;
  }

  const Operator& producer() const;

  Operator& producer();

  int32_t output_id() const noexcept {
    return _output_id;
  }

  bool is_tensor() const noexcept {
    return _output_id >= 0;
  }

  size_t num_consumers() const;

  const Operator& consumer(size_t i) const;

  Operator& consumer(size_t i);

  const NDArrayMeta& meta() const noexcept {
    return _meta;
  }

  size_t ndim() const {
    return _meta.ndim();
  }

  size_t numel() const {
    return _meta.numel();
  }

  DataType dtype() const {
    return _meta.dtype;
  }

  const Device& device() const noexcept {
    return _meta.device;
  }

  bool is_cpu() const {
    return _meta.device.is_cpu();
  }

  bool is_cuda() const {
    return _meta.device.is_cuda();
  }

  const HTShape& shape() const {
    return _meta.shape;
  }

  int64_t shape(size_t axis) const {
    return _meta.shape[axis];
  }

  const HTStride& stride() const {
    return _meta.stride;
  }

  int64_t stride(size_t axis) const {
    return _meta.stride[axis];
  }

  bool has_shape() const {
    return _meta.shape.size() > 0;
  }

  const Device& placement() const noexcept {
    return device();
  }

  void set_placement(const Device& p) {
    _meta.set_device(p);
    _distributed_states.set_placement(p);
  }

  bool is_computed() const {
    return _computed;
  }

  NDArray& GetOrCompute();

  bool is_variable() const;

  Tensor to_variable(bool trainable = false, const OpMeta& op_meta = OpMeta());

  bool is_trainable() const;

  void set_trainable(bool trainable);

  void reset_initializer(const Initializer& init);

  void reset_data(const NDArray& data);

  Tensor& Gradient();

  const Tensor& Gradient() const;

  void Backward(const Tensor& grad = Tensor());

  void AccumulateGrad(const Tensor& grad);

  void ZeroGrad();

  DistributedStates get_distributed_states() {
    return _distributed_states;
  }

  // do when MapToParallelDevices
  void set_placement_group(const DeviceGroup& placement_group) {
    _distributed_states.set_placement_group(placement_group);
  }

 protected:
  void SetProducer(Operator& op);

  void AddConsumer(Operator& op);

  // Walkaround methods to get the corresponding wrapper
  Tensor& get_self();

  const Tensor& get_self() const;

  const TensorId _id;
  const TensorName _name;
  const int32_t _output_id;
  NDArrayMeta _meta;

  // The `_producer` member is wrapped into a unique pointer since
  // the forward declared `Operator` class has incomplete type now.
  std::unique_ptr<Operator> _producer;
  OpList _consumers;

  // for define-by-run mode
  bool _computed{false};
  NDArray _data;
  Tensor _grad;

  // for distributed attributes
  DistributedStates _distributed_states;

 private:
  static TensorId _next_tensor_id() {
    static std::atomic<TensorId> _global_tensor_id(0);
    return _global_tensor_id++;
  }
};

class DistributedStates {
 public:
  // 1. Tensor创建时该属性默认为空, 需要后续赋值 
  DistributedStates() : _device_num(-1), _states({}), _order({}) {};
  // 2. 在Tensor创建时就直接定义好切分状态, 此时直接传递placement group即可
  DistributedStates(DeviceGroup& placement_group, const std::unordered_map<int32_t, int32_t>& states, 
                    const std::vector<int32_t>& order = {}) { // states/order都是const的引用, 因此未来要update的话只能先copy创建一个新对象再对其修改
    _placement_group = placement_group;
    _device_num = placement_group.num_devices();
    set_states(states);
    set_order(order); 
  }
  // 3. 作为单独的分布式属性存在, 可用于指定要转换的目前切分状态, 此时暂时只需要赋值device_num, 而placement_group要与Tensor绑定后赋值
  DistributedStates(int32_t device_num, const std::unordered_map<int32_t, int32_t>& states, 
                    const std::vector<int32_t>& order = {}) {
    _placement_group = DeviceGroup(); // 空的device group, 在和tensor binding时需要填充
    _device_num = device_num;
    set_states(states);
    set_order(order); 
  }

  // 假设dp被包含在distributed attributes里的states[0], 则_device_num实际上就等于device placement_group的size
  // 理论上除了pp把不同layer的op切分到几组不同的device group之外, dp和tp中所有op都是共享同一组device group的
  // 只有在和Tensor绑定时才会调用该函数来赋值placement group
  void set_placement_group(const DeviceGroup& placement_group) {
    HT_ASSERT(_device_num == -1 || placement_group.num_devices() == _device_num) 
              << "devices num in placement_group " << placement_group.num_devices() 
              << " must be equal to distributed requirement " << _device_num << "!";
    _placement_group = placement_group;
    _device_num = placement_group.num_devices();
  }

  // 主要用来确定local device在device group中的index, 以便定位该device的部分tensor在global tensor中的位置
  void set_placement(const Device& placement) {
    HT_ASSERT(_placement_group.num_devices() > 0 && _placement_group.contains(placement))
              << "the placement device " << placement << " must in placement group " << _placement_group;    
    _placement = placement;
  }

  // 用来给placeholder_op/variable_op这类input/variable tensor赋值states/order用, 其中placement等信息已经在map to devices的时候赋值过了
  // 在外面更新states/order推荐使用set_distributed_states, 同时赋值states和order, 保证其一致性
  void set_distributed_states(const DistributedStates& dst_distributed_states) {
    HT_ASSERT(_device_num == dst_distributed_states._device_num)
              << "device num in dst_distributed_states: " << dst_distributed_states._device_num
              << " must equal to tensor requirement: " << _device_num << "!";
    set_states(dst_distributed_states._states); // set_states会检查是否和device_num相匹配
    set_order(dst_distributed_states._order); // set_order会检查是否和states相匹配
  }

  // states/order是否已经被成功赋值, 理论上通过构造函数/set_distributed_states进行赋值的, 正确性是已经保证了的, 这里只需要验证有没有
  bool is_valid() {
    return _device_num == 1 || (_device_num > 1 && _states.size() > 0 && _order.size() > 0); 
  }

  std::unordered_map<int32_t, int32_t> get_states() {
    return _states;
  }

  std::vector<int32_t> get_order() {
    return _order;
  }

  DeviceGroup get_placement_group() {
    return _placement_group;
  }

  Device get_placement() {
    return _placement;
  }

  int32_t get_placement_index() {
    return _placement_group.get_index(_placement);
  }

  int32_t get_device_num() {
    return _device_num;
  }

  std::unordered_map<int32_t, int32_t> combine_states(std::pair<std::vector<int32_t>, int32_t>& src2dst);
  std::vector<int32_t> combine_order(std::pair<std::vector<int32_t>, int32_t>& src2dst);
  bool equal_states_and_order(std::unordered_map<int32_t, int32_t>& states1, std::vector<int32_t>& order1,
                              std::unordered_map<int32_t, int32_t>& states2, std::vector<int32_t>& order2);
  bool check_combine(DistributedStates& dst_distributed_states, std::pair<std::vector<int32_t>, int32_t>& src2dst);

  std::unordered_map<int32_t, int32_t> reduce_states(int dim);
  std::vector<int32_t> reduce_order(int dim);
  bool check_reduce_dim(DistributedStates& dst_distributed_states, int dim);

  bool check_allreduce(DistributedStates& dst_distributed_states);
  bool check_allgather(DistributedStates& dst_distributed_states);
  bool check_reducescatter(DistributedStates& dst_distributed_states);
  bool check_boradcast(DistributedStates& dst_distributed_states);
  bool check_reduce(DistributedStates& dst_distributed_states);  

  int32_t get_dim(int32_t index);
  std::vector<int32_t> get_loop_sizes();
  std::unordered_map<int32_t, int32_t> map_device_to_state_index(int32_t device_index); // for single device

  // 下面这几个函数暂时没啥用...
  // dst_distributed_states不允许存在partial的情况, 实际上partial只存在于op的output tensor上, 为了简便起见, 处理的时候先把partial利用allreduce转成duplicate
  // global info
  // 为啥要传入max_dimension这个参数呢? 因为它要取src和dst的_max_dimension的最大值
  StateNode* build_state_tree();
  std::unordered_map<int32_t, StateNode*> map_device_to_state_node(); // for all devices

  // 获取每一维(state tree从下到上的每一层layer)中每个device需要做的convert method
  std::vector<std::unordered_map<int32_t, ConvertMethod>> get_methods_for_convert_trees(DistributedStates& dst_distributed_states);
  

 protected:
  // 一般只在构造函数里调用, 在外面调用需要保证states与order的一致性, 后面看看要不要把它移到protected里
  // 在外面调用推荐使用set_distributed_states, 同时赋值states和order, 保证其一致性
  void set_states(const std::unordered_map<int32_t, int32_t>& states) {
    // 必须先确定device_num再赋值states
    HT_ASSERT(_device_num != -1) << "must assign placement group for device num before set states!";
    // check states & set max dimension
    int32_t device_num = 1;
    int32_t max_dimension = INT32_MIN;
    for (auto& kv : states) {
      device_num *= kv.second;
      if (kv.first > max_dimension) {
        max_dimension = kv.first;
      }
    }
    HT_ASSERT(device_num == _device_num) << "the devices num " << device_num 
              <<" used in states is not equal to distributed requirement " << _device_num << "!";
    _states = states;
    _max_dimension = max_dimension;
  }

  // 一般只在构造函数里调用, 在外面调用需要保证states与order的一致性, 后面看看要不要把它移到protected里
  // 在外面调用推荐使用set_distributed_states, 同时赋值states和order, 保证其一致性
  void set_order(const std::vector<int32_t>& order) {
    HT_ASSERT(_states.size() > 0) << "must assign states before set order!";
    // check order, must match states
    if (order.size() == 0) {
      // get default order
      std::vector<int32_t> keys; 
      keys.reserve(_states.size());
      for (auto kv : _states) {
        if (kv.second > 1) { // partial/duplicate必须大于1才能分到order
          keys.push_back(kv.first);
        }
      }
      std::sort(keys.begin(), keys.end());
      _order = keys;      
    } else {
      for (auto kv : _states) {
        if (kv.second > 1) {
          HT_ASSERT(std::find(order.begin(), order.end(), kv.first) != order.end())
                    << "order is not matched with states!";
        }
      }
      _order = order;
    }
  }

  // partial和duplicate的key要求必须存在, 值可以是1表示没有; 其余dimension只有存在切分的时候key:value才会存在于states中
  std::unordered_map<int32_t, int32_t> _states; // {dimension: split_num}, {-2: partial, -1: duplicate, 0~n-1: dimension}
  std::vector<int32_t> _order; // for device mapping
  int32_t _max_dimension; // 记录_states里最大的那一维
  DeviceGroup _placement_group; // 在和Tensor binding的时候必须设置, 否则可以为空
  int32_t _device_num; // 如果_placement_group为空, 则_device_num必须设置
  Device _placement;

  // global info
  StateNode* _state_tree;
  std::unordered_map<int32_t, std::vector<StateNode*>> _dimension_to_layer_nodes; // 每一维都对应tree中的某一个layer的所有nodes
  std::unordered_map<int32_t, StateNode*> _device_to_state_node; // device index map到最后一维的state node上(layer node)
  // std::unordered_map<>
};

class StateNode {
 public:
  StateNode(): _index(-1), _range(-1), _dimension(-3), _parent(nullptr) {}
  StateNode(int32_t index, int32_t range, int32_t dimension, StateNode* parent)
          : _index(index), _range(range), _dimension(dimension), _parent(parent) {}
  int32_t _index; // 该node表示这一维中的第几块
  int32_t _range; // 这一维总共分为几块
  int32_t _dimension;
  StateNode* _parent;
  std::vector<StateNode*> _childs;
};

class ConvertMethod {
 public:
  int32_t _method; // 0: split, 1: allgather

  // for split: 本质上就是拆分src变成dst
  int32_t _range; // split几份
  int32_t _index; // 留下其中的第几份
  
  // for allgather: 本质上就是合并src变成dst
  std::vector<int32_t> _devices; // 要和哪些devices进行allgather合并 
};

/******************************************************
 * Logging & Streaming
 ******************************************************/

std::ostream& operator<<(std::ostream&, const Tensor&);

} // namespace autograd
} // namespace hetu

namespace std {
inline std::string to_string(const hetu::autograd::Tensor& tensor) {
  std::ostringstream os;
  os << tensor;
  return os.str();
}
} // namespace std
