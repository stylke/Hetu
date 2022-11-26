#pragma once

#include "hetu/common/macros.h"

namespace hetu {

enum class ReductionType : int8_t {
  SUM = 0,
  AVG,
  PROD,
  MAX,
  MIN,
  NUM_REDUCTION_TYPES
};

constexpr ReductionType kSUM = ReductionType::SUM;
constexpr ReductionType kAVG = ReductionType::AVG;
constexpr ReductionType kPROD = ReductionType::PROD;
constexpr ReductionType kMAX = ReductionType::MAX;
constexpr ReductionType kMIN = ReductionType::MIN;
constexpr int16_t NUM_REDUCTION_TYPES =
  static_cast<int16_t>(ReductionType::NUM_REDUCTION_TYPES);

std::string ReductionType2Str(const ReductionType&);
std::ostream& operator<<(std::ostream&, const ReductionType&);

} // namespace hetu

namespace std {
template <>
struct hash<hetu::ReductionType> {
  std::size_t operator()(hetu::ReductionType red_type) const noexcept {
    return std::hash<int>()(static_cast<int>(red_type));
  }
};
} // namespace std
