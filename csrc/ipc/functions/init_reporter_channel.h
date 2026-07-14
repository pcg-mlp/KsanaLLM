/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "configure/environment.h"
#include "distributed/distributed_control_coordinator.h"
#include "ipc/core/ipc_func_types.h"
#include "ipc/core/ipc_func_utils.h"
#include "utils/waiter.h"

namespace ksana_llm {

class InitReporterChannelRequest : public IpcRequest {
 public:
  virtual ~InitReporterChannelRequest() {}
};

class InitReporterChannelResponse : public IpcResponse {
 public:
  virtual ~InitReporterChannelResponse() {}
  virtual Status Serialize(void* ptr, size_t* bytes) const override;
  virtual Status Deserialize(const void* ptr) override;

  // The loading result, succ or failure.
  Status result_status;
};

class InitReporterChannelCallable : public IpcCallable {
 public:
  virtual ~InitReporterChannelCallable() {}
  virtual Status Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) override;

  void SetDistributedControlCoordinator(std::shared_ptr<DistributedControlCoordinator> distributed_control_coordinator);

 private:
  std::shared_ptr<DistributedControlCoordinator> distributed_control_coordinator_ = nullptr;
};

class InitReporterChannelCallback : public IpcCallback, public IpcWaiter {
 public:
  virtual ~InitReporterChannelCallback() {}
  virtual Status Call(const IpcResponse* response, IpcContext* context) override;

  // The final result stat of loading weight.
  Status status;
};

}  // namespace ksana_llm
