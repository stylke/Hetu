#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/communication/nccl_comm_group.h"
#include "hetu/impl/utils/common_utils.h"

#include <thread>

namespace hetu {
namespace impl {

using namespace hetu::impl::comm;

void AllReduceCuda(const NDArray& input, NDArray& output,
                   const DeviceGroup& device_group, const Stream& stream) {
  auto ranks = DeviceGroupToWorldRanks(device_group);
  auto& comm_group = NCCLCommunicationGroup::GetOrCreate(ranks, stream);
  comm_group->AllReduce(input, output);
}

void AllGatherCuda(const NDArray& input, NDArray& output,
                   const DeviceGroup& device_group, const Stream& stream) {
  auto ranks = DeviceGroupToWorldRanks(device_group);
  auto& comm_group = NCCLCommunicationGroup::GetOrCreate(ranks, stream);
  comm_group->AllGather(input, output);                  
}

void ReduceScatterCuda(const NDArray& input, NDArray& output,
                   const DeviceGroup& device_group, const Stream& stream) {
  auto ranks = DeviceGroupToWorldRanks(device_group);
  auto& comm_group = NCCLCommunicationGroup::GetOrCreate(ranks, stream);
  comm_group->ReduceScatter(input, output);
}

void P2PSendCuda(const NDArray& data, const Device& dst, const Stream& stream) {
  auto src_rank = GetWorldRank();
  auto dst_rank = DeviceToWorldRank(dst);
  std::vector<int> ranks(2);
  ranks[0] = std::min(src_rank, dst_rank);
  ranks[1] = std::max(src_rank, dst_rank);
  auto& comm_group = NCCLCommunicationGroup::GetOrCreate(ranks, stream);
  comm_group->Send(data, dst_rank);
}

void P2PRecvCuda(NDArray& data, const Device& src, const Stream& stream) {
  auto src_rank = DeviceToWorldRank(src);
  auto dst_rank = GetWorldRank();
  std::vector<int> ranks(2);
  ranks[0] = std::min(src_rank, dst_rank);
  ranks[1] = std::max(src_rank, dst_rank);
  auto& comm_group = NCCLCommunicationGroup::GetOrCreate(ranks, stream);
  comm_group->Recv(data, src_rank);
}

void BatchedISendIRecvCuda(const NDArrayList& send_datas, 
  const std::vector<Device>& dsts, NDArrayList& recv_datas, 
  const std::vector<Device>& srcs, const std::vector<Device>& comm_deivces, 
  const Stream& stream) {
  std::vector<int> ranks(comm_deivces.size());
  std::transform(comm_deivces.begin(), comm_deivces.end(), ranks.begin(), [&](const Device& device) { return DeviceToWorldRank(device); });
  std::sort(ranks.begin(), ranks.end());
  auto& comm_group = NCCLCommunicationGroup::GetOrCreate(ranks, stream);
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
