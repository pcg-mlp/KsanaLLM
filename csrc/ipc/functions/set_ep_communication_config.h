/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <vector>
#include "distributed/control_message.h"
#include "ipc/core/ipc_func_types.h"
#include "ipc/core/ipc_func_utils.h"
#include "utils/status.h"

namespace ksana_llm {

class ModelInstance;

class SetEPCommunicationConfigRequest : public IpcRequest {
 public:
  virtual ~SetEPCommunicationConfigRequest() {}
  virtual Status Serialize(void* ptr, size_t* bytes) const override;
  virtual Status Deserialize(const void* ptr) override;

  // The aggregated EP communicator config from all nodes
  std::vector<EPCommunicationConfig> aggregated_comm_configs;
};

class SetEPCommunicationConfigResponse : public IpcResponse {
 public:
  virtual ~SetEPCommunicationConfigResponse() {}
  virtual Status Serialize(void* ptr, size_t* bytes) const override;
  virtual Status Deserialize(const void* ptr) override;
};

class SetEPCommunicationConfigCallable : public IpcCallable {
 public:
  virtual ~SetEPCommunicationConfigCallable() {}
  virtual Status Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) override;

  void SetModelInstance(std::shared_ptr<ModelInstance> model_instance);

 private:
  std::shared_ptr<ModelInstance> model_instance_ = nullptr;
};

class SetEPCommunicationConfigCallback : public IpcCallback, public IpcWaiter {
 public:
  virtual ~SetEPCommunicationConfigCallback() {}
  virtual Status Call(const IpcResponse* response, IpcContext* context) override;
};

}  // namespace ksana_llm