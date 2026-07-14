/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "configure/environment.h"
#include "distributed/distributed_data_coordinator.h"
#include "ipc/core/ipc_func_types.h"
#include "ipc/core/ipc_func_utils.h"
#include "utils/waiter.h"

namespace ksana_llm {

class ConnectNcclDataChannelRequest : public IpcRequest {
 public:
  virtual ~ConnectNcclDataChannelRequest() {}
  virtual Status Serialize(void* ptr, size_t* bytes) const override;
  virtual Status Deserialize(const void* ptr) override;

  ncclUniqueId nccl_unique_id;
};

class ConnectNcclDataChannelResponse : public IpcResponse {
 public:
  virtual ~ConnectNcclDataChannelResponse() {}
  virtual Status Serialize(void* ptr, size_t* bytes) const override;
  virtual Status Deserialize(const void* ptr) override;

  // The loading result, succ or failure.
  Status result_status;
};

class ConnectNcclDataChannelCallable : public IpcCallable {
 public:
  virtual ~ConnectNcclDataChannelCallable() {}
  virtual Status Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) override;

  void SetDistributedDataCoordinator(std::shared_ptr<DistributedDataCoordinator> distributed_data_coordinator);

 private:
  std::shared_ptr<DistributedDataCoordinator> distributed_data_coordinator_ = nullptr;
};

class ConnectNcclDataChannelCallback : public IpcCallback, public IpcWaiter {
 public:
  virtual ~ConnectNcclDataChannelCallback() {}
  virtual Status Call(const IpcResponse* response, IpcContext* context) override;

  // The final result stat of loading weight.
  Status status;
};

}  // namespace ksana_llm
