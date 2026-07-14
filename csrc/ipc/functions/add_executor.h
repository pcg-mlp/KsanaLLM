/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ipc/core/ipc_func_types.h"
#include "ipc/core/ipc_func_utils.h"
#include "utils/waiter.h"

namespace ksana_llm {

class AddExecutorRequest : public IpcRequest {
 public:
  virtual ~AddExecutorRequest() {}
  virtual Status Serialize(void* ptr, size_t* bytes) const override;
  virtual Status Deserialize(const void* ptr) override;

  // The executor process id.
  int64_t executor_pid = 0;
};

class AddExecutorResponse : public IpcResponse {
 public:
  virtual ~AddExecutorResponse() {}
};

class AddExecutorCallable : public IpcCallable, public IpcWaiter {
 public:
  virtual ~AddExecutorCallable() {}
  virtual Status Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) override;

  // all executor process ids.
  std::vector<int64_t> executor_pids;
};

class AddExecutorCallback : public IpcCallback {
 public:
  virtual ~AddExecutorCallback() {}
};

}  // namespace ksana_llm
