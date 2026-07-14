/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "device/device_context_impl.h"

#include "device/device_utils.h"
#include "fmt/format.h"
#include "utils/logger.h"

namespace ksana_llm {

template <int T>
DeviceContextT<T>::DeviceContextT(int node_device_num_) : node_device_num_(node_device_num_) {
  GetDevice(&device_id_);
  InitStreams();

  // Initialize the device extension.
  InitializeExtension();
}

template <int T>
DeviceContextT<T>::~DeviceContextT() {
  DestroyExtension();
}

template <int T>
void DeviceContextT<T>::InitStreams() {
  memory_manage_stream_ = std::make_shared<Stream>();
  primary_compute_stream_ = std::make_shared<Stream>();
  secondary_compute_stream_ = std::make_shared<Stream>();

  comm_stream_ = std::make_shared<Stream>();
  dist_send_comm_stream_ = std::make_shared<Stream>();
  dist_recv_comm_stream_ = std::make_shared<Stream>();

  h2d_stream_ = std::make_shared<Stream>();
  d2h_stream_ = std::make_shared<Stream>();
  d2d_stream_ = std::make_shared<Stream>();
}

template <int T>
void DeviceContextT<T>::ClearStreams() {
  memory_manage_stream_.reset();
  primary_compute_stream_.reset();
  secondary_compute_stream_.reset();

  comm_stream_.reset();
  dist_send_comm_stream_.reset();
  dist_recv_comm_stream_.reset();

  h2d_stream_.reset();
  d2h_stream_.reset();
  d2d_stream_.reset();
}

template class DeviceContextT<ACTIVE_DEVICE_TYPE>;

}  // namespace ksana_llm
