/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <condition_variable>
#include <memory>
#include <vector>

#include "configure/context.h"
#include "configure/environment.h"
#include "device/device_types.h"
#include "distributed/raw_packet.h"
#include "runtime/threadpool.h"
#include "utils/blocking_queue.h"
#include "utils/status.h"
#include "utils/tensor.h"
#include "utils/waiter.h"

namespace ksana_llm {

// Describe the hidden_units buffer
struct HiddenUnitDeviceBuffer {
 public:
  // The device Tensor.
  Tensor tensor;

#ifdef ENABLE_ACL
  // Specially for NPU.
  bool prefill_enabled = false;
  bool decode_enabled = false;
  Tensor prefill_tensor;
#endif
};

// Used for distributed tcp mode only.
// store the hidden units from network before copy to device.
// or hidden units from device before send to network.
struct HiddenUnitHostBuffer {
  // hidden unit shape, for one device, [max_token_num, hidden_unit_size]
  size_t shape_dims[2];

#ifdef ENABLE_ACL
  size_t prefill_shape_dims[2];
#endif

  // The data, for all devices.
  char data[0];
};

// Helper class for distributed tcp mode only.
class HiddenUnitBufferHelper {
 public:
  explicit HiddenUnitBufferHelper(std::shared_ptr<Environment> env);

  // convert host hidden_unit_buffer to device.
  Status ConvertHostBufferToDevice(HiddenUnitDeviceBuffer* hidden_unit_dev, HiddenUnitHostBuffer* hidden_unit_host);

  // convert device hidden_unit_buffer to host.
  Status ConvertDeviceBufferToHost(HiddenUnitHostBuffer* hidden_unit_host, HiddenUnitDeviceBuffer* hidden_unit_dev);

  void ResetDeviceBuffer(HiddenUnitDeviceBuffer* hidden_unit_dev);

  // Create a new host buffer, return a maximum size packet.
  Packet* CreateHostBuffer();
  HiddenUnitDeviceBuffer* CreateDeviceBuffer();

  // Get bytes of host buffer.
  size_t GetHostBufferSize(Packet* packet);

  // Used for allocate packet buffer only.
  static Packet* CreateHostBufferPacket();
  static size_t GetHostBufferPacketSize(Packet* packet);

 private:
  void InitializeHiddenUnitConfig();

 private:
  std::shared_ptr<Environment> env_ = nullptr;

  DataType weight_type_;
  size_t max_token_num_;
  size_t tensor_para_size_;
  size_t hidden_unit_size_;
};

}  // namespace ksana_llm
