#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/random/CPURandomState.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include <random>

namespace hetu {
namespace impl {

template <typename spec_t>
void init_normal_cpu(spec_t* arr, size_t size, spec_t mean, spec_t stddev,
                     uint64_t seed) {
  std::mt19937 engine(seed);
  std::normal_distribution<spec_t> dist(mean, stddev);
  for (size_t i = 0; i < size; i++)
    arr[i] = dist(engine);
}

template <typename spec_t>
void init_uniform_cpu(spec_t* arr, size_t size, spec_t lb, spec_t ub,
                      uint64_t seed) {
  std::mt19937 engine(seed);
  std::uniform_real_distribution<spec_t> dist(lb, ub);
  for (size_t i = 0; i < size; i++)
    arr[i] = dist(engine);
}

template <typename spec_t>
void init_truncated_normal_cpu(spec_t* arr, size_t size, spec_t mean,
                               spec_t stddev, spec_t lb, spec_t ub,
                               uint64_t seed) {
  std::mt19937 engine(seed);
  std::normal_distribution<spec_t> dist(mean, stddev);
  for (size_t i = 0; i < size; i++) {
    do {
      arr[i] = dist(engine);
    } while (arr[i] < lb || arr[i] > ub);
  }
}

void NormalInitsCpu(NDArray& data, double mean, double stddev, uint64_t seed,
                    const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(data);
  size_t size = data->numel();
  if (size == 0)
    return;
  if (seed == 0)
    seed = GenNextRandomSeed();
  HT_DISPATCH_FLOATING_TYPES(data->dtype(), spec_t, "NormalInitsCpu", [&]() {
    init_normal_cpu<spec_t>(data->data_ptr<spec_t>(), size,
                            static_cast<spec_t>(mean),
                            static_cast<spec_t>(stddev), seed);
  });
}

void UniformInitsCpu(NDArray& data, double lb, double ub, uint64_t seed,
                     const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(data);
  HT_ASSERT(lb < ub) << "Invalid range for uniform random init: "
                     << "[" << lb << ", " << ub << ").";
  size_t size = data->numel();
  if (size == 0)
    return;
  if (seed == 0)
    seed = GenNextRandomSeed();
  HT_DISPATCH_FLOATING_TYPES(data->dtype(), spec_t, "UniformInitCpu", [&]() {
    init_uniform_cpu<spec_t>(data->data_ptr<spec_t>(), size,
                             static_cast<spec_t>(lb), static_cast<spec_t>(ub),
                             seed);
  });
}

void TruncatedNormalInitsCpu(NDArray& data, double mean, double stddev,
                             double lb, double ub, uint64_t seed,
                             const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(data);
  size_t size = data->numel();
  if (size == 0)
    return;
  if (seed == 0)
    seed = GenNextRandomSeed();
  HT_DISPATCH_FLOATING_TYPES(
    data->dtype(), spec_t, "TruncatedNormalInitsCpu", [&]() {
      init_truncated_normal_cpu<spec_t>(
        data->data_ptr<spec_t>(), size, static_cast<spec_t>(mean),
        static_cast<spec_t>(stddev), static_cast<spec_t>(lb),
        static_cast<spec_t>(ub), seed);
    });
}

} // namespace impl
} // namespace hetu
