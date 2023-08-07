#include "hetu/graph/autocast/autocast.h"
#include <thread>

namespace hetu {
namespace graph {

std::once_flag AutoCast::_init_flag;
std::vector<std::shared_ptr<AutoCast>> AutoCast::_autocasts;
std::shared_ptr<AutoCast> AutoCast::_default_autocast;
thread_local std::stack<AutoCastId> AutoCast::_cur_autocast_ctx;

void AutoCast::Init() {
    // exit handler
  std::atexit([]() {
  });

  auto concurrency = std::thread::hardware_concurrency();
  AutoCast::_autocasts.reserve(MIN(concurrency, 16) * 2);
  AutoCast::_default_autocast = AutoCast::MakeAutoCast(true);
}

DataType AutoCast::WidestType(const TensorList& inputs) {
  DataType widest_type = DataType::FLOAT16;
  for (const auto& input: inputs) {
    if (input->dtype() == DataType::FLOAT64)
      return DataType::FLOAT64;
    if (input->dtype() == DataType::FLOAT32)
      widest_type = DataType::FLOAT32;
  }
  return widest_type;
}

void AutoCast::Tensor_AutoCast(TensorList& inputs, DataType datatype) {
  auto autocast_id = AutoCast::cur_autocast_ctx();
  if (autocast_id == UINT64_MAX)
    return;
  auto autocast = AutoCast::GetAutoCast(autocast_id);
  if (!autocast.enabled())
    return;
  if (datatype == DataType::UNDETERMINED) {
    datatype = AutoCast::WidestType(inputs);
  }
  for (auto& input: inputs) {
    if (input->dtype() != datatype && 
        (input->dtype() == DataType::BFLOAT16 ||
         input->dtype() == DataType::FLOAT16 ||
         input->dtype() == DataType::FLOAT32 ||
         input->dtype() == DataType::FLOAT64)) {
      input = MakeDataTransferOp(datatype ,input);
    }
  }
}

} // namespace graph
} // namespace hetu