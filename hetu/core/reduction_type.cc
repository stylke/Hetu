#include "hetu/core/reduction_type.h"

namespace hetu {

std::string ReductionType2Str(const ReductionType& red_type) {
  switch (red_type) {
    case ReductionType::SUM: return "SUM";
    case ReductionType::AVG: return "AVG";
    case ReductionType::PROD: return "PROD";
    case ReductionType::MAX: return "MAX";
    case ReductionType::MIN: return "MIN";
    default:
      HT_VALUE_ERROR << "Unknown reduction type: "
                     << static_cast<int32_t>(red_type);
      __builtin_unreachable();
  }
}

std::ostream& operator<<(std::ostream& os, const ReductionType& red_type) {
  os << ReductionType2Str(red_type);
  return os;
}

} // namespace hetu
