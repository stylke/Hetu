#include "hetu/core/ndarray.h"
#include "hetu/test/test_utils.h"

using namespace hetu;

const auto TEST_DEVICES = {Device(kCPU), Device(kCUDA, 0)};
constexpr auto TEST_DATA_TYPES = {kFloat32, kFloat64};

void TestAddition(const Device& device, DataType dtype,
                  const HTShape& shape = {1024, 1024}) {
  HT_LOG_INFO << "Testing NDArray Addition for device " << device
              << " and type " << dtype << "...";
  const double a = 1.234, b = 3.14159;
  const double ground_truth = a + b;
  auto array_a = NDArray::full(shape, a, device, dtype);
  auto array_b = NDArray::full(shape, b, device, dtype);
  auto array_c = array_a + b;
  auto array_d = array_a + array_b;
  SynchronizeAllStreams();
  assert_fuzzy_eq(array_c, ground_truth);
  assert_fuzzy_eq(array_c, array_d);
  HT_LOG_INFO << "Testing NDArray Addition for device " << device
              << " and type " << dtype << " done";
}

void TestSubtraction(const Device& device, DataType dtype,
                     const HTShape& shape = {1024, 1024}) {
  HT_LOG_INFO << "Testing NDArray Subtraction for device " << device
              << " and type " << dtype << "...";
  const double a = 1.234, b = 3.14159;
  const double ground_truth = a - b;
  auto array_a = NDArray::full(shape, a, device, dtype);
  auto array_b = NDArray::full(shape, b, device, dtype);
  auto array_c = array_a - b;
  auto array_d = a - array_b;
  auto array_e = array_a - array_b;
  SynchronizeAllStreams();
  assert_fuzzy_eq(array_c, ground_truth);
  assert_fuzzy_eq(array_c, array_d);
  assert_fuzzy_eq(array_c, array_e);
  HT_LOG_INFO << "Testing NDArray Subtraction for device " << device
              << " and type " << dtype << " done";
}

void TestMultiplication(const Device& device, DataType dtype,
                        const HTShape& shape = {1024, 1024}) {
  HT_LOG_INFO << "Testing NDArray Multiplication for device " << device
              << " and type " << dtype << "...";
  const double a = 1.234, b = 3.14159;
  const double ground_truth = a * b;
  auto array_a = NDArray::full(shape, a, device, dtype);
  auto array_b = NDArray::full(shape, b, device, dtype);
  auto array_c = array_a * b;
  auto array_d = a * array_b;
  auto array_e = array_a * array_b;
  SynchronizeAllStreams();
  assert_fuzzy_eq(array_c, ground_truth);
  assert_fuzzy_eq(array_c, array_d);
  assert_fuzzy_eq(array_c, array_e);
  HT_LOG_INFO << "Testing NDArray Multiplication for device " << device
              << " and type " << dtype << " done";
}

void TestDivision(const Device& device, DataType dtype,
                  const HTShape& shape = {1024, 1024}) {
  HT_LOG_INFO << "Testing NDArray Division for device " << device
              << " and type " << dtype << "...";
  const double a = 1.234, b = 3.14159;
  const double ground_truth = a / b;
  auto array_a = NDArray::full(shape, a, device, dtype);
  auto array_b = NDArray::full(shape, b, device, dtype);
  auto array_c = array_a / b;
  auto array_d = a / array_b;
  auto array_e = array_a / array_b;
  SynchronizeAllStreams();
  assert_fuzzy_eq(array_c, ground_truth);
  assert_fuzzy_eq(array_c, array_d);
  assert_fuzzy_eq(array_c, array_e);
  HT_LOG_INFO << "Testing NDArray Division for device " << device
              << " and type " << dtype << " done";
}

void TestPow(const Device& device, DataType dtype,
             const HTShape& shape = {1024, 1024}) {
  HT_LOG_INFO << "Testing NDArray Pow for device " << device << " and type "
              << dtype << "...";
  auto array = NDArray::rand(shape, device, dtype);
  auto pow2 = NDArray::pow(array, 2);
  auto sqr = array * array;
  auto pow3 = NDArray::pow(array, 3);
  auto cube = sqr * array;
  auto pow4 = NDArray::pow(array, 4);
  auto quad = cube * array;
  SynchronizeAllStreams();
  assert_fuzzy_eq(pow2, sqr);
  assert_fuzzy_eq(pow3, cube);
  assert_fuzzy_eq(pow4, quad);
  HT_LOG_INFO << "Testing NDArray Pow for device " << device << " and type "
              << dtype << " done";
}

void TestSqrt(const Device& device, DataType dtype,
              const HTShape& shape = {1024, 1024}) {
  HT_LOG_INFO << "Testing NDArray Sqrt for device " << device << " and type "
              << dtype << "...";
  auto array = NDArray::rand(shape, device, dtype);
  auto sqrt = NDArray::sqrt(array);
  auto pow_half = NDArray::pow(array, 0.5);
  auto sqr_of_sqrt = sqrt * sqrt;
  SynchronizeAllStreams();
  assert_fuzzy_eq(sqrt, pow_half);
  assert_fuzzy_eq(sqr_of_sqrt, array);
  HT_LOG_INFO << "Testing NDArray Sqrt for device " << device << " and type "
              << dtype << " done";
}

int main(int argc, char** argv) {
  for (const auto& device : TEST_DEVICES) {
    for (const auto& dtype : TEST_DATA_TYPES) {
      TestAddition(device, dtype);
      TestSubtraction(device, dtype);
      TestMultiplication(device, dtype);
      TestDivision(device, dtype);
      TestPow(device, dtype);
      TestSqrt(device, dtype);
    }
  }
  return 0;
}
