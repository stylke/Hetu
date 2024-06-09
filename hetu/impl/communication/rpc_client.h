#pragma once

#include "hetu/impl/communication/rpc_client_impl.h"
#include <grpcpp/grpcpp.h>
#include "hetu/impl/communication/rpc/heturpc.grpc.pb.h"

using json = nlohmann::json;

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

namespace hetu {

class DeviceClient : public DeviceClientImpl {
 public:
  DeviceClient() {}
  DeviceClient(std::shared_ptr<Channel> channel)
      : stub_(DeviceController::NewStub(channel)) {}

  int Connect(const std::string& hostname) override;

  int GetRank(const std::string& user) override;

  int CommitHostName(const std::string& hostname, int rank) override;

  std::string GetHostName(int rank) override;

  int CommitDeviceInfo(int type, int index, int multiplex, int rank) override;

  DeviceInfoReply GetDeviceInfo(int rank) override;

  int CommitNcclId(const std::string& nccl_id, const std::vector<int>& world_rank, int stream_id) override;

  std::string GetNcclId(const std::vector<int>& world_rank, int stream_id) override;

  int Exit(int rank) override;

  int PutDouble(const std::string& key, double value) override;

  double GetDouble(const std::string& key) override;

  std::string RemoveDouble(const std::string& key) override;

  int PutInt(const std::string& key, int64_t value) override;

  int64_t GetInt(const std::string& key) override;

  std::string RemoveInt(const std::string& key) override;

  int PutString(const std::string& key, const std::string& value) override;

  std::string GetString(const std::string& key) override;

  std::string RemoveString(const std::string& key) override;

  int PutBytes(const std::string& key, const std::string& value) override;

  std::string GetBytes(const std::string& key) override;

  std::string RemoveBytes(const std::string& key) override;

  int PutJson(const std::string& key, const json& value) override;

  json GetJson(const std::string& key) override;

  std::string RemoveJson(const std::string& key) override;

  int Barrier(int rank, const std::vector<int>& world_rank) override;

 private:
  std::unique_ptr<DeviceController::Stub> stub_;
};

} //namespace hetu
