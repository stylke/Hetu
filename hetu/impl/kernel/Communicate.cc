#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/communication/mpi_comm_group.h"
#include "hetu/impl/utils/common_utils.h"

namespace hetu {
namespace impl {

using namespace hetu::impl::comm;

void AllReduceCpu(const NDArray& input, NDArray& output,
                  const DeviceGroup& device_group, const Stream& stream) {
  auto ranks = DeviceGroupToWorldRanks(device_group);
  auto& comm_group = MPICommunicationGroup::GetOrCreate(ranks, stream);
  comm_group->AllReduce(input, output);
}

void AllGatherCpu(const NDArray& input, NDArray& output,
                   const DeviceGroup& device_group, const Stream& stream) {
  auto ranks = DeviceGroupToWorldRanks(device_group);
  auto& comm_group = MPICommunicationGroup::GetOrCreate(ranks, stream);
  comm_group->AllGather(input, output);                  
}

void ReduceScatterCpu(const NDArray& input, NDArray& output,
                   const DeviceGroup& device_group, const Stream& stream) {
  auto ranks = DeviceGroupToWorldRanks(device_group);
  auto& comm_group = MPICommunicationGroup::GetOrCreate(ranks, stream);
  comm_group->ReduceScatter(input, output);
}

void P2PSendCpu(const NDArray& data, const Device& dst, const Stream& stream) {
  auto src_rank = GetWorldRank();
  auto dst_rank = DeviceToWorldRank(dst);
  std::vector<int> ranks(2);
  ranks[0] = std::min(src_rank, dst_rank);
  ranks[1] = std::max(src_rank, dst_rank);
  auto& comm_group = MPICommunicationGroup::GetOrCreate(ranks, stream);
  comm_group->Send(data, dst_rank);
}

void P2PRecvCpu(NDArray& data, const Device& src, const Stream& stream) {
  auto src_rank = DeviceToWorldRank(src);
  auto dst_rank = GetWorldRank();
  std::vector<int> ranks(2);
  ranks[0] = std::min(src_rank, dst_rank);
  ranks[1] = std::max(src_rank, dst_rank);
  auto& comm_group = MPICommunicationGroup::GetOrCreate(ranks, stream);
  comm_group->Recv(data, src_rank);
}

void BatchedISendIRecvCpu(const NDArrayList& send_datas, const std::vector<Device>& dsts, 
  NDArrayList& recv_datas, const std::vector<Device>& srcs, const Stream& stream) {
  std::vector<int> ranks;
  auto local_rank = GetWorldRank();
  ranks.push_back(local_rank);
  auto push_ranks = [&](const std::vector<Device>& devices) {
    for (auto& device : devices) {
      int rank = DeviceToWorldRank(device);
      if (std::find(ranks.begin(), ranks.end(), rank) == ranks.end()) {
        ranks.push_back(rank);
      }
    }
  };
  push_ranks(dsts); // 要send的目标devices的rank
  push_ranks(srcs); // 要recv的目标devices的rank
  std::sort(ranks.begin(), ranks.end());
  auto& comm_group = MPICommunicationGroup::GetOrCreate(ranks, stream);
  std::vector<Task> tasks;
  for (int i = 0; i < send_datas.size(); i++) {
    tasks.push_back(comm_group->ISend(send_datas[i], DeviceToWorldRank(dsts[i])));
  }
  for (int i = 0; i < recv_datas.size(); i++) {
    tasks.push_back(comm_group->IRecv(recv_datas[i], DeviceToWorldRank(srcs[i])));
  }
  comm_group->BatchedISendIRecv(tasks);
}

} // namespace impl
} // namespace hetu
