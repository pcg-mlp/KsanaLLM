/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ipc/core/ipc_func_types.h"
#include "ipc/core/ipc_func_utils.h"
#include "runtime/model_instance.h"
#include "utils/waiter.h"

namespace ksana_llm {

class LoadModelRequest : public IpcRequest {
 public:
  virtual ~LoadModelRequest() {}
};

class LoadModelResponse : public IpcResponse {
 public:
  virtual ~LoadModelResponse() {}
  virtual Status Serialize(void* ptr, size_t* bytes) const override;
  virtual Status Deserialize(const void* ptr) override;

  // The loading result, succ or failure.
  Status result_status;
};

class LoadModelCallable : public IpcCallable {
 public:
  virtual ~LoadModelCallable() {}
  virtual Status Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) override;

  void SetModelInstance(std::shared_ptr<ModelInstance> model_instance);

 private:
  std::shared_ptr<ModelInstance> model_instance_ = nullptr;
};

class LoadModelCallback : public IpcCallback, public IpcWaiter {
 public:
  virtual ~LoadModelCallback() {}
  virtual Status Call(const IpcResponse* response, IpcContext* context) override;

  // The final result stat of loading model.
  Status load_status;
};

}  // namespace ksana_llm
