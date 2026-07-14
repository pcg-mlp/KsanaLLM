/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ipc/core/ipc_func_types.h"
#include "ipc/core/ipc_func_utils.h"
#include "runtime/model_instance.h"
#include "utils/waiter.h"

namespace ksana_llm {

class GetDeviceMemoryRequest : public IpcRequest {
 public:
  virtual ~GetDeviceMemoryRequest() {}
};

class GetDeviceMemoryResponse : public IpcResponse {
 public:
  virtual ~GetDeviceMemoryResponse() {}
  virtual Status Serialize(void* ptr, size_t* bytes) const override;
  virtual Status Deserialize(const void* ptr) override;

  size_t device_free = 0;
  size_t device_total = 0;
};

class GetDeviceMemoryCallable : public IpcCallable {
 public:
  virtual ~GetDeviceMemoryCallable() {}
  virtual Status Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) override;
};

class GetDeviceMemoryCallback : public IpcCallback {
 public:
  virtual ~GetDeviceMemoryCallback() {}
};

}  // namespace ksana_llm
