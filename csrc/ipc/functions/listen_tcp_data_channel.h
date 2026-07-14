/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "configure/environment.h"
#include "distributed/distributed_data_coordinator.h"
#include "ipc/core/ipc_func_types.h"
#include "ipc/core/ipc_func_utils.h"
#include "utils/waiter.h"

namespace ksana_llm {

class ListenTcpDataChannelRequest : public IpcRequest {
 public:
  virtual ~ListenTcpDataChannelRequest() {}
  virtual Status Serialize(void* ptr, size_t* bytes) const override;
  virtual Status Deserialize(const void* ptr) override;
};

class ListenTcpDataChannelResponse : public IpcResponse {
 public:
  virtual ~ListenTcpDataChannelResponse() {}
  virtual Status Serialize(void* ptr, size_t* bytes) const override;
  virtual Status Deserialize(const void* ptr) override;

  // The result status.
  Status status;

  // The host:port of executor.
  HostPort executor_host_port;
};

class ListenTcpDataChannelCallable : public IpcCallable {
 public:
  virtual ~ListenTcpDataChannelCallable() {}
  virtual Status Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) override;

  void SetDistributedDataCoordinator(std::shared_ptr<DistributedDataCoordinator> distributed_data_coordinator);

 private:
  std::shared_ptr<DistributedDataCoordinator> distributed_data_coordinator_ = nullptr;
};

class ListenTcpDataChannelCallback : public IpcCallback, public IpcWaiter {
 public:
  virtual ~ListenTcpDataChannelCallback() {}
  virtual Status Call(const IpcResponse* response, IpcContext* context) override;

  // The host:port of every executor.
  std::vector<HostPort> executor_host_ports;
  Status status;
};

}  // namespace ksana_llm
