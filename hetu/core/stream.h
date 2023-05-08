#pragma once

#include "hetu/core/device.h"

namespace hetu {

using StreamIndex = int32_t;
constexpr int32_t HT_NUM_STREAMS_PER_DEVICE = 1 << 4;
constexpr StreamIndex kUndeterminedStream = -1;
constexpr StreamIndex kBlockingStream = 0;
constexpr StreamIndex kComputingStream = 1;
constexpr StreamIndex kH2DStream = 2;
constexpr StreamIndex kD2HStream = 3;
constexpr StreamIndex kP2PStream = 4;
constexpr StreamIndex kCollectiveStream = 5;

class Stream {
 public:
  Stream(Device device = Device(), StreamIndex id = kBlockingStream)
  : _device(device), _id(id) {
    HT_ASSERT(_device.local())
      << "Cannot create stream for remote device: " << _device;
    HT_ASSERT(_id >= kBlockingStream) << "Invalid stream index: " << _id;
  }

  Stream(const Stream&) = default;
  Stream(Stream&&) = default;
  Stream& operator=(const Stream&) = default;
  Stream& operator=(Stream&&) = default;

  void Sync() const;

  inline Device device() const noexcept {
    return _device;
  }

  inline DeviceType device_type() const noexcept {
    return _device.type();
  }

  inline DeviceIndex device_index() const noexcept {
    return _device.index();
  }

  inline StreamIndex stream_index() const noexcept {
    return _id;
  }

  inline bool is_defined() const noexcept {
    return !_device.is_undetermined();
  }

  inline bool operator==(const Stream& stream) const {
    return _device == stream._device && _id == stream._id;
  }

  inline bool operator!=(const Stream& stream) const {
    return !operator==(stream);
  }

 private:
  Device _device;
  StreamIndex _id;
};

void SynchronizeAllStreams(const Device& device = Device());

std::ostream& operator<<(std::ostream&, const Stream&);

class Event {
 public:
  Event(Device device) : _device(device) {}

  virtual void Record(const Stream& stream) = 0;

  virtual void Sync() = 0;

  virtual void Block(const Stream& stream) = 0;

  virtual int64_t TimeSince(const Event& event) const = 0;

 protected:
  Device _device;
};

class DefaultEvent final : public Event {
 public:
  DefaultEvent(Device device) : Event(device) {
    HT_ASSERT(device.is_cpu())
      << "DefaultEvent should be used with host devices. "
      << "Got " << device;
  }

  inline void Record(const Stream& stream) {
    _tp = std::chrono::steady_clock::now();
    _recorded = true;
  }

  inline void Sync() {
    HT_ASSERT(_recorded) << "Event has not been recorded";
  }

  inline void Block(const Stream& stream) {}

  inline int64_t TimeSince(const Event& event) const {
    const auto& e = reinterpret_cast<const DefaultEvent&>(event);
    HT_ASSERT(e._recorded) << "Start event has not been recorded";
    HT_ASSERT(_recorded) << "Stop event has not been recorded";
    return std::chrono::duration_cast<std::chrono::nanoseconds>(_tp - e._tp)
      .count();
  }

 private:
  std::chrono::time_point<std::chrono::steady_clock> _tp;
  bool _recorded{false};
};

} // namespace hetu

namespace std {
template <>
struct hash<hetu::Stream> {
  std::size_t operator()(const hetu::Stream& stream) const noexcept {
    // Following boost::hash_combine
    auto hash = std::hash<hetu::StreamIndex>()(stream.stream_index());
    hash ^= (std::hash<hetu::Device>()(stream.device()) + 0x9e3779b9 +
             (hash << 6) + (hash >> 2));
    return hash;
  }
};

inline std::string to_string(const hetu::Stream& stream) {
  std::ostringstream os;
  os << stream;
  return os.str();
}

} // namespace std
