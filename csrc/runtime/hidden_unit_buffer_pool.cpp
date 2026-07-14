/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "runtime/hidden_unit_buffer_pool.h"

#include "configure/environment.h"
#include "device/device_types.h"
#include "device/device_utils.h"
#include "profiler/profile_event.h"
#include "utils/memory_utils.h"
#include "utils/singleton.h"
#include "utils/status.h"

namespace ksana_llm {

using namespace std::placeholders;

HiddenUnitBufferPool::HiddenUnitBufferPool(std::shared_ptr<DistributedDataCoordinator> distributed_data_coordinator,
                                           std::shared_ptr<Environment> env, std::shared_ptr<Context> context) {
  distributed_data_coordinator_ = distributed_data_coordinator;
  context_ = context;
  env_ = env;

  hidden_unit_buffer_helper_ = std::make_shared<HiddenUnitBufferHelper>(env_);

  send_buffer_thread_ =
      std::unique_ptr<std::thread>(new std::thread(&HiddenUnitBufferPool::ProcessBufferSendLoop, this));
  recv_buffer_thread_ =
      std::unique_ptr<std::thread>(new std::thread(&HiddenUnitBufferPool::ProcessBufferRecvLoop, this));
}

void HiddenUnitBufferPool::InitializeHiddenUnitBuffer() {
  // two buffer, for computation and communication.
  constexpr size_t STANDALONE_BUFFER_SIZE = 1;
  constexpr size_t DISTRIBUTED_BUFFER_SIZE = 2;

  size_t buffer_num = context_->global->IsStandalone() ? STANDALONE_BUFFER_SIZE : DISTRIBUTED_BUFFER_SIZE;
  for (size_t i = 0; i < buffer_num; ++i) {
    if (context_->global->IsStandalone() || context_->global->IsDistributedMaster()) {
      ready_device_buffer_queue_.Put(hidden_unit_buffer_helper_->CreateDeviceBuffer());
    } else {
      free_device_buffer_queue_.Put(hidden_unit_buffer_helper_->CreateDeviceBuffer());
    }
  }
}

HiddenUnitDeviceBuffer* HiddenUnitBufferPool::GetBuffer() { return ready_device_buffer_queue_.Get(); }

void HiddenUnitBufferPool::FreeBuffer(HiddenUnitDeviceBuffer* hidden_unit) {
  hidden_unit_buffer_helper_->ResetDeviceBuffer(hidden_unit);

  if (context_->global->IsStandalone()) {
    AppendReadyQueue(hidden_unit);
  } else if (context_->global->IsDistributedMaster()) {
    AsyncSendBuffer(hidden_unit, std::bind(&HiddenUnitBufferPool::AppendReadyQueue, this, _1));
  } else if (context_->global->IsDistributedWorker() && !context_->global->IsDistributedLastWorker()) {
    AsyncSendBuffer(hidden_unit, std::bind(&HiddenUnitBufferPool::AppendFreeQueue, this, _1));
  } else if (context_->global->IsDistributedLastWorker()) {
    AppendFreeQueue(hidden_unit);
  }
}

void HiddenUnitBufferPool::AsyncSendBuffer(HiddenUnitDeviceBuffer* hidden_unit, HiddenUnitDeviceBufferCallback cb) {
  HiddenUnitSendTask send_task = {hidden_unit, cb};
  send_device_buffer_queue_.Put(send_task);
}

void HiddenUnitBufferPool::ProcessBufferSendLoop() {
  SetDevice(context_->device->GetDeviceId());
  while (!terminated_) {
    HiddenUnitSendTask send_task = send_device_buffer_queue_.Get();
    if (!send_task.dev_buffer) {
      KLLM_LOG_WARNING << "ProcessBufferSendLoop nullptr HiddenUnitDeviceBuffer, break.";
      break;
    }

    HiddenUnitMeta hidden_unit_meta = send_device_buffer_meta_queue_.Get();

    // block until hidden_unit could be reused.
    distributed_data_coordinator_->SendData(send_task.dev_buffer, hidden_unit_meta.shape, hidden_unit_meta.data_type);
    send_task.cb(send_task.dev_buffer);
  }
}

void HiddenUnitBufferPool::ProcessBufferRecvLoop() {
  SetDevice(context_->device->GetDeviceId());
  while (!terminated_) {
    HiddenUnitDeviceBuffer* dev_buffer = free_device_buffer_queue_.Get();
    if (!dev_buffer) {
      KLLM_LOG_WARNING << "ProcessBufferRecvLoop nullptr HiddenUnitDeviceBuffer, break.";
      break;
    }

    HiddenUnitMeta hidden_unit_meta = recv_device_buffer_meta_queue_.Get();

    // block until dev buffer is received.
    distributed_data_coordinator_->RecvData(dev_buffer, hidden_unit_meta.shape, hidden_unit_meta.data_type);
    ready_device_buffer_queue_.Put(dev_buffer);
  }
}

void HiddenUnitBufferPool::AddHiddenUnitSendMeta(const std::vector<size_t>& shape, DataType data_type) {
  HiddenUnitMeta hidden_unit_meta = {shape, data_type};
  send_device_buffer_meta_queue_.Put(hidden_unit_meta);
}

void HiddenUnitBufferPool::AddHiddenUnitRecvMeta(const std::vector<size_t>& shape, DataType data_type) {
  HiddenUnitMeta hidden_unit_meta = {shape, data_type};
  recv_device_buffer_meta_queue_.Put(hidden_unit_meta);
}

void HiddenUnitBufferPool::AppendReadyQueue(HiddenUnitDeviceBuffer* hidden_unit) {
  ready_device_buffer_queue_.Put(hidden_unit);
}

void HiddenUnitBufferPool::AppendFreeQueue(HiddenUnitDeviceBuffer* hidden_unit) {
  free_device_buffer_queue_.Put(hidden_unit);
}

}  // namespace ksana_llm
