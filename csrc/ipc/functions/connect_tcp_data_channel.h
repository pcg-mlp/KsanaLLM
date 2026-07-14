/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "configure/environment.h"
#include "distributed/distributed_data_coordinator.h"
#include "ipc/core/ipc_func_types.h"
#include "ipc/core/ipc_func_utils.h"
#include "utils/waiter.h"

namespace ksana_llm {

class ConnectTcpDataChannelRequest : public IpcRequest {
 public:
  virtual ~ConnectTcpDataChannelRequest() {}
  virtual Status Serialize(void* ptr, size_t* bytes) const override;
  virtual Status Deserialize(const void* ptr) override;

  HostPort executor_downstream_host_port;
};

class ConnectTcpDataChannelResponse : public IpcResponse {
 public:
  virtual ~ConnectTcpDataChannelResponse() {}
  virtual Status Serialize(void* ptr, size_t* bytes) const override;
  virtual Status Deserialize(const void* ptr) override;

  // The connect result, succ or failure.
  Status status;
};

class ConnectTcpDataChannelCallable : public IpcCallable {
 public:
  virtual ~ConnectTcpDataChannelCallable() {}
  virtual Status Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) override;

  void SetDistributedDataCoordinator(std::shared_ptr<DistributedDataCoordinator> distributed_data_coordinator);

 private:
  std::shared_ptr<DistributedDataCoordinator> distributed_data_coordinator_ = nullptr;
};

class ConnectTcpDataChannelCallback : public IpcCallback, public IpcWaiter {
 public:
  virtual ~ConnectTcpDataChannelCallback() {}
  virtual Status Call(const IpcResponse* response, IpcContext* context) override;

  // The final result stat of loading weight.
  Status status;
};

}  // namespace ksana_llm
