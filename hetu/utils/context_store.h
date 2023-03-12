#pragma once

#include "hetu/common/macros.h"

namespace hetu {

class ContextStore {
 public:
  ContextStore() {}

  void put_bool(const std::string& key, bool value) {
    _ctx.insert({key, value ? "true" : "false"});
  }

  bool get_bool(const std::string& key, bool default_value = false) const {
    auto it = _ctx.find(key);
    return it != _ctx.end() ? (it->second == "true") : default_value;
  }

  void put_int32(const std::string& key, int32_t value) {
    _ctx.insert({key, std::to_string(value)});
  }

  int32_t get_int32(const std::string& key, int32_t default_value = 0) const {
    auto it = _ctx.find(key);
    return it != _ctx.end() ? std::stoi(it->second) : default_value;
  }

  void put_uint32(const std::string& key, uint32_t value) {
    _ctx.insert({key, std::to_string(value)});
  }

  uint32_t get_uint32(const std::string& key,
                      uint32_t default_value = 0) const {
    auto it = _ctx.find(key);
    return it != _ctx.end() ? static_cast<uint32_t>(std::stoul(it->second))
                            : default_value;
  }

  void put_int64(const std::string& key, int64_t value) {
    _ctx.insert({key, std::to_string(value)});
  }

  int64_t get_int64(const std::string& key, int64_t default_value = 0) const {
    auto it = _ctx.find(key);
    return it != _ctx.end() ? std::stoll(it->second) : default_value;
  }

  void put_uint64(const std::string& key, uint64_t value) {
    _ctx.insert({key, std::to_string(value)});
  }

  uint64_t get_uint64(const std::string& key,
                      uint64_t default_value = 0) const {
    auto it = _ctx.find(key);
    return it != _ctx.end() ? std::stoull(it->second) : default_value;
  }

  void put_float32(const std::string& key, float value) {
    _ctx.insert({key, std::to_string(value)});
  }

  float get_float32(const std::string& key, float default_value = 0) const {
    auto it = _ctx.find(key);
    return it != _ctx.end() ? std::stof(it->second) : default_value;
  }

  void put_float64(const std::string& key, double value) {
    _ctx.insert({key, std::to_string(value)});
  }

  double get_float64(const std::string& key, double default_value = 0) const {
    auto it = _ctx.find(key);
    return it != _ctx.end() ? std::stod(it->second) : default_value;
  }

  void put_string(const std::string& key, const std::string& value) {
    _ctx.insert({key, value});
  }

  const std::string& get_string(const std::string& key,
                                const std::string& default_value = "") const {
    auto it = _ctx.find(key);
    return it != _ctx.end() ? it->second : default_value;
  }

 private:
  std::unordered_map<std::string, std::string> _ctx;
};

} // namespace hetu
