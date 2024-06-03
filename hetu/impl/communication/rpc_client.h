#pragma once

#include <iostream>
#include <memory>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"

#include <grpcpp/grpcpp.h>

#include "hetu/impl/communication/rpc/heturpc.grpc.pb.h"
#include "hetu/utils/json/json.hpp"

using json = nlohmann::json;

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

namespace hetu {

struct DeviceInfoReply {
  int type;
  int index;
  int multiplex;
};

class DeviceClient {
 public:
  DeviceClient() {}
  DeviceClient(std::shared_ptr<Channel> channel)
      : stub_(DeviceController::NewStub(channel)) {}

  int Connect(const std::string& hostname);

  int GetRank(const std::string& user);

  int CommitHostName(const std::string& hostname, int rank);

  std::string GetHostName(int rank);

  int CommitDeviceInfo(int type, int index, int multiplex, int rank);

  DeviceInfoReply GetDeviceInfo(int rank);

  int CommitNcclId(const std::string& nccl_id, const std::vector<int>& world_rank, int stream_id);

  std::string GetNcclId(const std::vector<int>& world_rank, int stream_id);

  int Exit(int rank);

  int PutDouble(const std::string& key, double value);

  double GetDouble(const std::string& key);

  std::string RemoveDouble(const std::string& key);

  int PutInt(const std::string& key, int64_t value);

  int64_t GetInt(const std::string& key);

  std::string RemoveInt(const std::string& key);

  int PutString(const std::string& key, const std::string& value);

  std::string GetString(const std::string& key);

  std::string RemoveString(const std::string& key);

  int PutBytes(const std::string& key, const std::string& value);

  std::string GetBytes(const std::string& key);

  std::string RemoveBytes(const std::string& key);

  int PutJson(const std::string& key, const json& value);

  json GetJson(const std::string& key);

  std::string RemoveJson(const std::string& key);

  int Barrier(int rank, const std::vector<int>& world_rank);

 private:
  std::unique_ptr<DeviceController::Stub> stub_;
};

} //namespace hetu
