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

} // namespace impl
} // namespace hetu
