#include "hetu/graph/headers.h"
#include "hetu/graph/eager_graph.h"
#include "hetu/graph/define_by_run_graph.h"

namespace hetu {
namespace graph {

TensorDef::TensorDef(const constrcutor_access_key&, TensorIdentifier ids,
                     TensorName name, NDArrayMeta meta)
: _ids(std::move(ids)), _name(std::move(name)), _meta(std::move(meta)) {
  auto& graph = Graph::GetGraph(_ids.graph_id);
  _inform_graph_on_destruction = (graph.type() == GraphType::EAGER ||
                                  graph.type() == GraphType::DEFINE_BY_RUN);
}

Tensor::Tensor(TensorIdentifier ids, TensorName name, NDArrayMeta meta)
: shared_ptr_wrapper<TensorDef>() {
  _ptr = make_ptr<TensorDef>(TensorDef::constrcutor_access_key(),
                             std::move(ids), std::move(name), std::move(meta));
}

Tensor::~Tensor() {
  if (!is_defined())
    return;

  if (_ptr->_inform_graph_on_destruction && _ptr->num_consumers() == 0 &&
      get_referrence_count() == 1) {
    // To avoid a second entrance
    _ptr->_inform_graph_on_destruction = false;
    // Inform graph to move or prune
    auto& graph = _ptr->graph();
    if (graph.type() == GraphType::EAGER) {
      reinterpret_cast<EagerGraph&>(graph).RemoveTensor(*this);
    } else if (graph.type() == GraphType::DEFINE_BY_RUN) {
      reinterpret_cast<DefineByRunGraph&>(graph).PruneTensor(*this);
    }
  }
  _ptr = nullptr;
}

void TensorDef::AddConsumer(Operator& op) {
  _consumers.push_back(std::ref(op));
}

void TensorDef::DelConsumer(const Operator& op) {
  _consumers.erase(
    std::remove_if(_consumers.begin(), _consumers.end(),
                   [&](const OpRef& x) { return x.get()->id() == op->id(); }));
}

const Graph& TensorDef::graph() const {
  return Graph::GetGraph(graph_id());
}

Graph& TensorDef::graph() {
  return Graph::GetGraph(graph_id());
}

const Operator& TensorDef::producer() const {
  return graph().GetOp(producer_id());
}

Operator& TensorDef::producer() {
  return graph().GetOp(producer_id());
}

const Operator& TensorDef::consumer(size_t i) const {
  return _consumers[i];
}

Operator& TensorDef::consumer(size_t i) {
  return _consumers[i];
}

Tensor& TensorDef::get_self() {
  return output_id() >= 0 ? producer()->output(output_id())
                          : producer()->out_dep_linker();
}

const Tensor& TensorDef::get_self() const {
  return output_id() >= 0 ? producer()->output(output_id())
                          : producer()->out_dep_linker();
}

bool TensorDef::is_variable() const {
  return is_variable_op(producer());
}

bool TensorDef::is_parameter() const {
  return is_parameter_op(producer());
}

NDArray TensorDef::get_or_compute() {
  return graph().GetOrCompute(get_self());
}

std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
  if (tensor.is_defined())
    os << tensor->name();
  else
    os << "Tensor()";
  return os;
}

} // namespace graph
} // namespace hetu
