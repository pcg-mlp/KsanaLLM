/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <unordered_map>

#include "device/device_utils.h"
#include "models/base/weight_status.h"

namespace ksana_llm {

// The extension type define.
template <int T>
struct DeviceExtensionTypeTraits {
  typedef DummyClass value_type;
};

// Manage device streams and communicators.
template <int T>
class DeviceContextT {
 public:
  explicit DeviceContextT(int node_device_num);
  ~DeviceContextT();

  // Get current device id.
  int GetDeviceId() { return device_id_; }

  // Get device stream.
  Stream& GetMemoryManageStream() { return *memory_manage_stream_; }

  Stream& GetPrimaryComputeStream() { return *primary_compute_stream_; }
  Stream& GetSecondaryComputeStream() { return *secondary_compute_stream_; }

  Stream& GetH2DStream() { return *h2d_stream_; }
  Stream& GetD2HStream() { return *d2h_stream_; }
  Stream& GetD2DStream() { return *d2d_stream_; }

  Stream& GetCommStream() { return *comm_stream_; }
  Stream& GetDistributedSendCommStream() { return *dist_send_comm_stream_; }
  Stream& GetDistributedRecvCommStream() { return *dist_recv_comm_stream_; }

  inline bool IsRunContextDecodeAndDecodeSerially() { return is_contextdecode_and_decode_run_serially_; }
  inline bool IsGemmFp8Supported() { return is_gemm_fp8_supported_; }

  // Weight status management
  void SetWeightStatus(const std::string& key, const WeightStatus& status) { weight_status_map_[key] = status; }

  WeightStatus& GetWeightStatus(const std::string& key) {
    return weight_status_map_[key];  // Will create default WeightStatus if key not found
  }

 private:
  // Initialize all necessary streams.
  void InitStreams();
  void ClearStreams();

 public:
  using FriendType = typename DeviceExtensionTypeTraits<T>::value_type;
  friend FriendType;

  typename DeviceExtensionTypeTraits<T>::value_type* ext = nullptr;

  // Initialize and destroy extension, implemented by device.
  void InitializeExtension();
  void DestroyExtension();

 private:
  // memory operation.
  std::shared_ptr<Stream> memory_manage_stream_ = nullptr;

  // compute.
  std::shared_ptr<Stream> primary_compute_stream_ = nullptr;
  std::shared_ptr<Stream> secondary_compute_stream_ = nullptr;

  // communication.
  std::shared_ptr<Stream> comm_stream_ = nullptr;

  // communication between distributed nodes.
  std::shared_ptr<Stream> dist_send_comm_stream_ = nullptr;
  std::shared_ptr<Stream> dist_recv_comm_stream_ = nullptr;

  // memcpy
  std::shared_ptr<Stream> h2d_stream_ = nullptr;
  std::shared_ptr<Stream> d2h_stream_ = nullptr;
  std::shared_ptr<Stream> d2d_stream_ = nullptr;

  bool is_contextdecode_and_decode_run_serially_{true};
  bool is_gemm_fp8_supported_{false};
  int driver_version_;

  // The device_num of per inference node.
  int node_device_num_ = -1;

  // the device_id in current inference node.
  int device_id_ = -1;

  // Weight status map
  std::unordered_map<std::string, WeightStatus> weight_status_map_;
};

}  // namespace ksana_llm
