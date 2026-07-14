/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <condition_variable>
#include <memory>
#include <vector>

#include "configure/context.h"
#include "configure/environment.h"
#include "device/device_types.h"
#include "distributed/distributed_data_coordinator.h"
#include "distributed/raw_packet.h"
#include "runtime/hidden_unit_buffer.h"
#include "runtime/threadpool.h"
#include "utils/blocking_queue.h"
#include "utils/status.h"
#include "utils/tensor.h"
#include "utils/waiter.h"

namespace ksana_llm {

class HiddenUnitBufferPool {
 public:
  HiddenUnitBufferPool(std::shared_ptr<DistributedDataCoordinator> distributed_data_coordinator,
                       std::shared_ptr<Environment> env, std::shared_ptr<Context> context);

  // Initialize buffer, for both standalone and distributed.
  void InitializeHiddenUnitBuffer();

  // Get prepared buffer for computation..
  HiddenUnitDeviceBuffer* GetBuffer();

  // Free device buffer.
  void FreeBuffer(HiddenUnitDeviceBuffer* hidden_unit);

  // Append hidden_unit meta to queue.
  void AddHiddenUnitSendMeta(const std::vector<size_t>& shape, DataType data_type);
  void AddHiddenUnitRecvMeta(const std::vector<size_t>& shape, DataType data_type);

 private:
  using HiddenUnitDeviceBufferCallback = std::function<void(HiddenUnitDeviceBuffer*)>;

  // Used for nccl communication.
  struct HiddenUnitMeta {
    std::vector<size_t> shape;
    DataType data_type;
  };

  // Async send task for hidden_unit_buffer.
  struct HiddenUnitSendTask {
    HiddenUnitDeviceBuffer* dev_buffer = nullptr;
    HiddenUnitDeviceBufferCallback cb;
  };

  void AppendReadyQueue(HiddenUnitDeviceBuffer* hidden_unit);
  void AppendFreeQueue(HiddenUnitDeviceBuffer* hidden_unit);

  void ProcessBufferSendLoop();
  void ProcessBufferRecvLoop();
  void AsyncSendBuffer(HiddenUnitDeviceBuffer* hidden_unit, HiddenUnitDeviceBufferCallback cb);

 private:
  std::shared_ptr<Environment> env_ = nullptr;
  std::shared_ptr<Context> context_ = nullptr;
  std::shared_ptr<HiddenUnitBufferHelper> hidden_unit_buffer_helper_ = nullptr;

  std::shared_ptr<DistributedDataCoordinator> distributed_data_coordinator_ = nullptr;

  BlockingQueue<HiddenUnitMeta> send_device_buffer_meta_queue_;
  BlockingQueue<HiddenUnitMeta> recv_device_buffer_meta_queue_;

  BlockingQueue<HiddenUnitDeviceBuffer*> free_device_buffer_queue_;
  BlockingQueue<HiddenUnitDeviceBuffer*> ready_device_buffer_queue_;

  BlockingQueue<HiddenUnitSendTask> send_device_buffer_queue_;

  std::shared_ptr<std::thread> send_buffer_thread_ = nullptr;
  std::shared_ptr<std::thread> recv_buffer_thread_ = nullptr;

  bool terminated_ = false;
};

}  // namespace ksana_llm
