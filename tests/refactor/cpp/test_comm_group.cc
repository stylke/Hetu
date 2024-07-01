#include "hetu/impl/communication/mpi_comm_group.h"
#include "hetu/impl/communication/nccl_comm_group.h"
// #include "hetu/autograd/ops/kernel_links.h"
#include "test_utils.h"
#include <thread>
#include <cmath>

using namespace hetu;
using namespace hetu::impl;
using namespace hetu::impl::comm;

const auto TEST_DEVICE_TYPES = {kCPU, kCUDA};
constexpr auto TEST_DATA_TYPES = {kFloat32, kFloat64};

void TestBroadcastAndReduce(DeviceType device_type, DataType dtype,
                            const std::vector<int>& ranks = {},
                            const HTShape& shape = {1024, 1024}) {
  HT_LOG_INFO << "Testing Broadcast and Reduce for device " << device_type
              << " and type " << dtype << "...";
  int world_rank = GetWorldRank();
  if (!ranks.empty() &&
      std::find(ranks.begin(), ranks.end(), world_rank) == ranks.end())
    return;
  Device device(device_type, world_rank);
  CommunicationGroup group;
  if (device.is_cpu())
    group = MPICommunicationGroup::GetOrCreate(ranks);
  else
    group = NCCLCommunicationGroup::GetOrCreate(ranks, device);
  HT_LOG_DEBUG << "Rank[" << group->rank() << "/" << group->size() << "] "
               << "World rank: " << world_rank;
  int root_rank = group->group_to_world_rank(0);
  NDArray array, reduced_array;
  const double scalar = 3.14159;
  const double reduced_scalar = scalar * group->size();
  if (world_rank == root_rank) {
    array = NDArray::full(shape, scalar, device, dtype);
    reduced_array = NDArray::empty(shape, device, dtype);
  } else {
    array = NDArray::empty(shape, device, dtype);
  }
  SynchronizeAllStreams();
  HT_LOG_DEBUG << "Rank[" << group->rank() << "/" << group->size() << "] "
               << "Intialized array: " << array;
  group->Broadcast(array, root_rank);
  group->Reduce(array, reduced_array, root_rank);
  group->Sync();
  if (world_rank == root_rank) {
    HT_LOG_DEBUG << "Rank[" << group->rank() << "/" << group->size() << "] "
                 << "Reduced array: " << reduced_array;
    assert_fuzzy_eq(reduced_array, reduced_scalar);
  } else {
    HT_LOG_DEBUG << "Rank[" << group->rank() << "/" << group->size() << "] "
                 << "Received broadcasted array: " << array;
    assert_fuzzy_eq(array, scalar);
  }
  group->Barrier(true);
  HT_LOG_INFO << "Testing Broadcast and Reduce for device " << device_type
              << " and type " << dtype << " done";
  group->Barrier(true);
}

void TestAllReduce(DeviceType device_type, DataType dtype,
                   const std::vector<int>& ranks = {},
                   const HTShape& shape = {1024, 1024}) {
  HT_LOG_INFO << "Testing AllReduce for device " << device_type << " and type "
              << dtype << "...";
  int world_rank = GetWorldRank();
  if (!ranks.empty() &&
      std::find(ranks.begin(), ranks.end(), world_rank) == ranks.end())
    return;
  Device device(device_type, world_rank);
  CommunicationGroup group;
  if (device.is_cpu())
    group = MPICommunicationGroup::GetOrCreate(ranks);
  else
    group = NCCLCommunicationGroup::GetOrCreate(ranks, device);
  HT_LOG_DEBUG << "Rank[" << group->rank() << "/" << group->size() << "] "
               << "World rank: " << world_rank;
  const double scalar = 1.234;
  const double scalar_of_rank = (group->rank() + 1) * scalar;
  const double reduced_scalar =
    ((group->size() + 1) * group->size() / 2) * scalar;
  NDArray array = NDArray::full(shape, scalar_of_rank, device, dtype);
  SynchronizeAllStreams();
  HT_LOG_DEBUG << "Rank[" << group->rank() << "/" << group->size() << "] "
               << "Intialized array: " << array;
  group->AllReduce(array, array);
  group->Sync();
  HT_LOG_DEBUG << "Rank[" << group->rank() << "/" << group->size() << "] "
               << "All reduced array: " << array;
  assert_fuzzy_eq(array, reduced_scalar);
  group->Barrier(true);
  HT_LOG_INFO << "Testing AllReduce for device " << device_type << " and type "
              << dtype << " done";
  group->Barrier(true);
}

void TestAllGather(DeviceType device_type, DataType dtype,
                   const std::vector<int>& ranks = {},
                   const HTShape& shape = {1024, 1024}) {
  HT_LOG_INFO << "Testing AllGather for device " << device_type << " and type "
              << dtype << "...";
  int world_rank = GetWorldRank();
  if (!ranks.empty() &&
      std::find(ranks.begin(), ranks.end(), world_rank) == ranks.end())
    return;
  Device device(device_type, world_rank);
  CommunicationGroup group;
  if (device.is_cpu())
    group = MPICommunicationGroup::GetOrCreate(ranks);
  else
    group = NCCLCommunicationGroup::GetOrCreate(ranks, device);
  HT_LOG_DEBUG << "Rank[" << group->rank() << "/" << group->size() << "] "
               << "World rank: " << world_rank;
  const double scalar = 1.234;
  const double scalar_of_rank = (group->rank() + 1) * scalar;
  NDArray array = NDArray::full(shape, scalar_of_rank, device, dtype);
  HTShape gather_shape = shape;
  gather_shape[0] *= group->size();
  NDArray gathered_array = NDArray::empty(gather_shape, device, dtype);
  SynchronizeAllStreams();
  HT_LOG_DEBUG << "Rank[" << group->rank() << "/" << group->size() << "] "
               << "Intialized array: " << array;
  group->AllGather(array, gathered_array);
  group->Sync();
  auto splits = NDArray::split(gathered_array, group->size(), 0);
  for (int r = 0; r < group->size(); r++) {
    HT_LOG_DEBUG << "Rank[" << group->rank() << "/" << group->size() << "] "
                 << "Gathered array from " << r << ": " << splits[r];
    assert_fuzzy_eq(splits[r], (r + 1) * scalar);
  }
  group->Barrier(true);
  HT_LOG_INFO << "Testing AllGather for device " << device_type << " and type "
              << dtype << " done";
  group->Barrier(true);
}

void TestReduceScatter(DeviceType device_type, DataType dtype,
                       const std::vector<int>& ranks = {},
                       const HTShape& shape = {1024, 1024}) {
  HT_LOG_INFO << "Testing ReduceScatter for device " << device_type
              << " and type " << dtype << "...";
  int world_rank = GetWorldRank();
  if (!ranks.empty() &&
      std::find(ranks.begin(), ranks.end(), world_rank) == ranks.end())
    return;
  Device device(device_type, world_rank);
  CommunicationGroup group;
  if (device.is_cpu())
    group = MPICommunicationGroup::GetOrCreate(ranks);
  else
    group = NCCLCommunicationGroup::GetOrCreate(ranks, device);
  HT_LOG_DEBUG << "Rank[" << group->rank() << "/" << group->size() << "] "
               << "World rank: " << world_rank;
  const double scalar = 1.234;
  const double scalar_of_rank = (group->rank() + 1) * scalar;
  const double reduced_scalar =
    ((group->size() + 1) * group->size() / 2) * scalar;
  HTShape input_shape = shape;
  input_shape[0] *= group->size();
  NDArray array = NDArray::full(input_shape, scalar_of_rank, device, dtype);
  NDArray scattered_array = NDArray::empty(shape, device, dtype);
  SynchronizeAllStreams();
  HT_LOG_DEBUG << "Rank[" << group->rank() << "/" << group->size() << "] "
               << "Intialized array: " << array;
  group->ReduceScatter(array, scattered_array);
  group->Sync();
  HT_LOG_DEBUG << "Rank[" << group->rank() << "/" << group->size() << "] "
               << "Reduced and scattered array: " << scattered_array;
  assert_fuzzy_eq(scattered_array, reduced_scalar);
  group->Barrier(true);
  HT_LOG_INFO << "Testing ReduceScatter for device " << device_type
              << " and type " << dtype << " done";
  group->Barrier(true);
}

void TestGatherAndScatter(DeviceType device_type, DataType dtype,
                          const std::vector<int>& ranks = {},
                          const HTShape& shape = {1024, 1024}) {
  HT_LOG_INFO << "Testing Gather and Scatter for device " << device_type
              << " and type " << dtype << "...";
  int world_rank = GetWorldRank();
  if (!ranks.empty() &&
      std::find(ranks.begin(), ranks.end(), world_rank) == ranks.end())
    return;
  Device device(device_type, world_rank);
  CommunicationGroup group;
  if (device.is_cpu())
    group = MPICommunicationGroup::GetOrCreate(ranks);
  else
    group = NCCLCommunicationGroup::GetOrCreate(ranks, device);
  HT_LOG_DEBUG << "Rank[" << group->rank() << "/" << group->size() << "] "
               << "World rank: " << world_rank;
  const double scalar1 = 1.234;
  const double scalar2 = 3.14159;
  const double scalar1_of_rank = (group->rank() + 1) * scalar1;
  const double scalar2_of_rank = (group->rank() + 1) * scalar2;
  NDArray array = NDArray::full(shape, scalar1_of_rank, device, dtype);
  NDArray gathered_array;
  int root_rank = group->group_to_world_rank(0);
  if (world_rank == root_rank) {
    HTShape gather_shape = shape;
    gather_shape[0] *= group->size();
    gathered_array = NDArray::empty(gather_shape, device, dtype);
  }
  SynchronizeAllStreams();
  HT_LOG_DEBUG << "Rank[" << group->rank() << "/" << group->size() << "] "
               << "Intialized array for gather: " << array;
  group->Gather(array, gathered_array, root_rank);
  if (world_rank == root_rank) {
    group->Sync();
    auto splits = NDArray::split(gathered_array, group->size(), 0);
    for (int r = 0; r < group->size(); r++) {
      HT_LOG_DEBUG << "Rank[" << group->rank() << "/" << group->size() << "] "
                   << "Gathered array from " << r << ": " << splits[r];
      assert_fuzzy_eq(splits[r], (r + 1) * scalar1);
      splits[r] = NDArray::full(shape, (r + 1) * scalar2, device, dtype);
    }
    // Note: we do not support in-place modification at this moment
    gathered_array = NDArray::cat(splits, 0);
    SynchronizeAllStreams();
    HT_LOG_DEBUG << "Rank[" << group->rank() << "/" << group->size() << "] "
                 << "Intialized array for scatter: " << gathered_array;
  }
  group->Scatter(gathered_array, array, root_rank);
  group->Sync();
  HT_LOG_DEBUG << "Rank[" << group->rank() << "/" << group->size() << "] "
               << "Scattered array: " << array;
  assert_fuzzy_eq(array, scalar2_of_rank);
  group->Barrier(true);
  HT_LOG_INFO << "Testing Gather and Scatter for device " << device_type
              << " and type " << dtype << " done";
  group->Barrier(true);
}

void TestSendAndRecv(DeviceType device_type, DataType dtype,
                     const std::vector<int>& ranks = {},
                     const HTShape& shape = {1024, 1024}) {
  HT_LOG_INFO << "Testing Send and Recv for device " << device_type
              << " and type " << dtype << "...";
  int world_rank = GetWorldRank();
  if (!ranks.empty() &&
      std::find(ranks.begin(), ranks.end(), world_rank) == ranks.end())
    return;
  Device device(device_type, world_rank);
  CommunicationGroup group;
  if (device.is_cpu())
    group = MPICommunicationGroup::GetOrCreate(ranks);
  else
    group = NCCLCommunicationGroup::GetOrCreate(ranks, device);
  HT_LOG_DEBUG << "Rank[" << group->rank() << "/" << group->size() << "] "
               << "World rank: " << world_rank;
  const double scalar = 3.14159;
  NDArray array;
  if (group->rank() == 0) {
    array = NDArray::full(shape, scalar, device, dtype);
    for (int i = 0; i < group->size(); i++) {
      if (i != group->rank()) {
        int dst = group->group_to_world_rank(i);
        group->Send(array, dst);
      }
    }
  } else {
    array = NDArray::empty(shape, device, dtype);
    group->Recv(array, group->group_to_world_rank(0));
  }
  group->Sync();
  HT_LOG_DEBUG << "Rank[" << group->rank() << "/" << group->size() << "] "
               << "Sent/Received array: " << array;
  assert_fuzzy_eq(array, scalar);
  group->Barrier(true);
  HT_LOG_INFO << "Testing Send and Recv for device " << device_type
              << " and type " << dtype << " done";
  group->Barrier(true);
}

void TestOverlap(DeviceType device_type, DataType dtype,
                 const std::vector<int>& collective_ranks = {},
                 const std::vector<int>& p2p_ranks = {0, 1},
                 const HTShape& shape = {1024, 1024}) {
  HT_LOG_INFO << "Testing Overlap for device " << device_type << " and type "
              << dtype << "...";
  HT_ASSERT(p2p_ranks.size() == 2) << "Invalid p2p ranks: " << p2p_ranks;
  int world_rank = GetWorldRank();
  bool in_collective_group = collective_ranks.empty() ||
    (std::find(collective_ranks.begin(), collective_ranks.end(), world_rank) !=
     collective_ranks.end());
  bool in_p2p_group = std::find(p2p_ranks.begin(), p2p_ranks.end(),
                                world_rank) != p2p_ranks.end();
  if (!in_p2p_group && !in_collective_group)
    return;
  Device device(device_type, world_rank);
  CommunicationGroup collective_group, p2p_group;
  const double scalar1 = 1.234;
  const double scalar2 = 3.14159;
  NDArrayList all_reduce_arrays, p2p_send_arrays, p2p_recv_arrays;
  int num_calls = 10;
  if (in_collective_group) {
    if (device.is_cpu())
      collective_group = MPICommunicationGroup::GetOrCreate(collective_ranks);
    else
      collective_group =
        NCCLCommunicationGroup::GetOrCreate(collective_ranks, device);
    const double scalar1_of_rank = (collective_group->rank() + 1) * scalar1;
    HT_LOG_DEBUG << "Collective Rank[" << collective_group->rank() << "/"
                 << collective_group->size() << "] World rank: " << world_rank;
    for (int i = 0; i < num_calls; i++)
      all_reduce_arrays.push_back(
        NDArray::full(shape, scalar1_of_rank * std::pow(10, i), device, dtype));
  }
  if (in_p2p_group) {
    if (device.is_cpu())
      p2p_group = MPICommunicationGroup::GetOrCreate(p2p_ranks);
    else
      p2p_group = NCCLCommunicationGroup::GetOrCreate(p2p_ranks, device);
    HT_LOG_DEBUG << "P2P Rank[" << p2p_group->rank() << "/" << p2p_group->size()
                 << "] World rank: " << world_rank;
    if (p2p_group->rank() == 0)
      for (int i = 0; i < num_calls; i++)
        p2p_send_arrays.push_back(
          NDArray::full(shape, scalar2 * std::pow(10, i), device, dtype));
    else
      for (int i = 0; i < num_calls; i++)
        p2p_recv_arrays.push_back(NDArray::empty(shape, device, dtype));
  }

  auto collective_fn = [&]() {
    const double reduced_scalar1 =
      ((collective_group->size() + 1) * collective_group->size() / 2) * scalar1;
    for (auto& array : all_reduce_arrays)
      collective_group->AllReduce(array, array);
    collective_group->Sync();
    for (int i = 0; i < num_calls; i++) {
      auto& array = all_reduce_arrays[i];
      HT_LOG_DEBUG << "Collective Rank[" << collective_group->rank() << "/"
                   << collective_group->size() << "] "
                   << "All reduced array " << i << ": " << array;
      assert_fuzzy_eq(array, reduced_scalar1 * std::pow(10, i));
    }
  };
  auto p2p_fn = [&]() {
    if (p2p_group->rank() == 0)
      for (auto& array : p2p_send_arrays)
        p2p_group->Send(array, p2p_group->group_to_world_rank(1));
    else
      for (auto& array : p2p_recv_arrays)
        p2p_group->Recv(array, p2p_group->group_to_world_rank(0));
    p2p_group->Sync();
    if (p2p_group->rank() == 1)
      for (int i = 0; i < num_calls; i++) {
        auto& array = p2p_recv_arrays[i];
        HT_LOG_DEBUG << "P2P Rank[" << p2p_group->rank() << "/"
                     << p2p_group->size() << "] "
                     << "Received array " << i << ": " << array;
        assert_fuzzy_eq(array, scalar2 * std::pow(10, i));
      }
  };

  std::vector<std::thread> threads;
  if (in_p2p_group)
    threads.emplace_back(p2p_fn);
  if (in_collective_group)
    threads.emplace_back(collective_fn);

  for (auto& thread : threads)
    thread.join();
  if (in_collective_group)
    collective_group->Barrier(true);
  if (in_p2p_group)
    p2p_group->Barrier(true);
  HT_LOG_INFO << "Testing Overlap for device " << device_type << " and type "
              << dtype << " done";
  if (in_collective_group)
    collective_group->Barrier(true);
  if (in_p2p_group)
    p2p_group->Barrier(true);
}

int main(int argc, char** argv) {
  for (const auto& device_type : TEST_DEVICE_TYPES) {
    for (const auto& dtype : TEST_DATA_TYPES) {
      TestBroadcastAndReduce(device_type, dtype);
      TestAllReduce(device_type, dtype);
      TestAllGather(device_type, dtype);
      TestReduceScatter(device_type, dtype);
      TestGatherAndScatter(device_type, dtype);
      TestSendAndRecv(device_type, dtype);
      TestOverlap(device_type, dtype);
    }
  }
  return 0;
}
