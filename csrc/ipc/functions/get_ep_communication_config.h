/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <vector>
#include "distributed/control_message.h"
#include "ipc/core/ipc_func_types.h"
#include "ipc/core/ipc_func_utils.h"
#include "utils/status.h"

namespace ksana_llm {

class ModelInstance;
class Environment;

class GetEPCommunicationConfigRequest : public IpcRequest {
 public:
  virtual ~GetEPCommunicationConfigRequest() {}
  virtual Status Serialize(void* ptr, size_t* bytes) const override;
  virtual Status Deserialize(const void* ptr) override;
};

class GetEPCommunicationConfigResponse : public IpcResponse {
 public:
  virtual ~GetEPCommunicationConfigResponse() {}
  virtual Status Serialize(void* ptr, size_t* bytes) const override;
  virtual Status Deserialize(const void* ptr) override;

  // The EP communicator config from this executor
  EPCommunicationConfig executor_comm_config;
};

class GetEPCommunicationConfigCallable : public IpcCallable {
 public:
  virtual ~GetEPCommunicationConfigCallable() {}
  virtual Status Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) override;

  void SetModelInstance(std::shared_ptr<ModelInstance> model_instance);
  void SetEnvironment(std::shared_ptr<Environment> env);

 private:
  std::shared_ptr<ModelInstance> model_instance_;
  std::shared_ptr<Environment> env_;
};

class GetEPCommunicationConfigCallback : public IpcCallback, public IpcWaiter {
 public:
  virtual ~GetEPCommunicationConfigCallback() {}
  virtual Status Call(const IpcResponse* response, IpcContext* context) override;

  // Collected communicator config from all executors
  std::vector<EPCommunicationConfig> executor_comm_configs;
};

}  // namespace ksana_llm