/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "configure/environment.h"
#include "executor/executor_runtime.h"
#include "ipc/core/ipc_func_types.h"
#include "ipc/core/ipc_func_utils.h"
#include "utils/waiter.h"

namespace ksana_llm {

class CreateExecutorRuntimeRequest : public IpcRequest {
 public:
  virtual ~CreateExecutorRuntimeRequest() {}
  virtual Status Serialize(void* ptr, size_t* bytes) const override;
  virtual Status Deserialize(const void* ptr) override;

  // Vocab info for structured output
  VocabInfo vocab_info;
};

class CreateExecutorRuntimeResponse : public IpcResponse {
 public:
  virtual ~CreateExecutorRuntimeResponse() {}
  virtual Status Serialize(void* ptr, size_t* bytes) const override;
  virtual Status Deserialize(const void* ptr) override;

  // The loading result, succ or failure.
  Status result_status;
};

class CreateExecutorRuntimeCallable : public IpcCallable {
 public:
  virtual ~CreateExecutorRuntimeCallable() {}
  virtual Status Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) override;

  void SetExecutorRuntime(std::shared_ptr<ExecutorRuntime> executor_runtime);
  void SetEnvironment(std::shared_ptr<Environment> env);

 private:
  std::shared_ptr<ExecutorRuntime> executor_runtime_ = nullptr;
  std::shared_ptr<Environment> env_ = nullptr;
};

class CreateExecutorRuntimeCallback : public IpcCallback, public IpcWaiter {
 public:
  virtual ~CreateExecutorRuntimeCallback() {}
  virtual Status Call(const IpcResponse* response, IpcContext* context) override;

  // The final result status of creating llm runtime.
  Status create_status;
};

}  // namespace ksana_llm
