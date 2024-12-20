/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <condition_variable>
#include <memory>
#include <vector>

#include "ksana_llm/distributed/raw_packet.h"
#include "ksana_llm/utils/blocking_queue.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/tensor.h"

namespace ksana_llm {

// Describe the hidden_units buffer
struct HiddenUnitDeviceBuffer {
  // The unique id for one schedule step.
  size_t schedule_id;

  // The device Tensor.
  std::vector<Tensor> tensors;

#ifdef ENABLE_ACL
  // Specially for NPU.
  bool prefill_enabled = false;
  bool decode_enabled = false;
  std::vector<Tensor> prefill_tensors;
#endif
};

// Used for distributed mode,
// store the hidden units from network before copy to device.
// or hidden units from device bdefore send to network.
struct HiddenUnitHostBuffer {
  // The unique id for one schedule step.
  size_t schedule_id;

  // hidden unit shape, for one device, [max_token_num, hidden_unit_size]
  size_t shape_dims[2];

#ifdef ENABLE_ACL
  size_t prefill_shape_dims[2];
#endif

  // The device nummber.
  size_t tensor_parallel;

  // The data, for all devices.
  char data[0];
};

// The buffer pool used to manage hidden unit buffers.
class HiddenUnitBufferPool {
 public:
  HiddenUnitBufferPool();

  // Get a hidden unit buffer object, do not create any new object.
  HiddenUnitDeviceBuffer* GetDeviceBuffer();

  // Free the hidden unit buffer to object pool.
  Status FreeDeviceBuffer(HiddenUnitDeviceBuffer* hidden_unit);

  // Get and free the host buffer, create new object if needed.
  // Note: here will return a maximum size packet.
  Packet* GetHostBuffer();
  Status FreeHostBuffer(Packet* hidden_unit_buffer);

  // Put to and get from host received buffer.
  Status PutToHostRecvQueue(Packet* packet);
  Packet* GetFromHostRecvQueue();

  // Put to and get from device received buffer.
  Status PutToDeviceRecvQueue(HiddenUnitDeviceBuffer* hidden_unit);
  HiddenUnitDeviceBuffer* GetFromDeviceRecvQueue();

  // Put to and get from send buffer.
  Status PutToSendQueue(HiddenUnitDeviceBuffer* hidden_unit);
  Packet* GetFromSendQueue();

  Status ConvertHostBufferToDevice(HiddenUnitDeviceBuffer* hidden_unit_dev, HiddenUnitHostBuffer* hidden_unit_host);
  Status ConvertDeviceBufferToHost(HiddenUnitHostBuffer* hidden_unit_host, HiddenUnitDeviceBuffer* hidden_unit_dev);

  // Get bytes of host buffer.
  size_t GetHostPacketSize(Packet* packet);

  // All blocked queue will be returned immediately.
  Status Stop();

 private:
  // Initialize hidden unit device buffer, for max possible memory size.
  Status InitializeHiddenUnitDeviceBuffer(HiddenUnitDeviceBuffer* hidden_unit_buffer);

  void InitializeBufferSize();

 private:
  DataType weight_type_;
  size_t max_token_num_;
  size_t tensor_para_size_;
  size_t hidden_unit_size_;

  // free device buffer, resuable.
  BlockingQueue<HiddenUnitDeviceBuffer*> free_device_buffers_;

  // received device buffer.
  BlockingQueue<HiddenUnitDeviceBuffer*> recv_device_buffers_;

  // Recv buffer.
  BlockingQueue<Packet*> recv_host_buffers_;

  // Send buffer.
  BlockingQueue<Packet*> send_host_buffers_;

  // no used buffers.
  BlockingQueue<Packet*> free_host_buffers_;
};

}  // namespace ksana_llm
