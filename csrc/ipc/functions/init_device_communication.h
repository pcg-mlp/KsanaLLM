/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "configure/context.h"
#include "configure/environment.h"
#include "ipc/core/ipc_func_types.h"
#include "ipc/core/ipc_func_utils.h"
#include "utils/waiter.h"

namespace ksana_llm {

class InitDeviceCommunicationRequest : public IpcRequest {
 public:
  virtual ~InitDeviceCommunicationRequest() {}
  virtual Status Serialize(void* ptr, size_t* bytes) const override;
  virtual Status Deserialize(const void* ptr) override;

#ifdef ENABLE_CUDA
  ncclUniqueId nccl_uid;
#endif
};

class InitDeviceCommunicationResponse : public IpcResponse {
 public:
  virtual ~InitDeviceCommunicationResponse() {}
};

class InitDeviceCommunicationCallable : public IpcCallable {
 public:
  virtual ~InitDeviceCommunicationCallable() {}
  virtual Status Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) override;

  void SetContext(std::shared_ptr<Context> context);

 private:
  std::shared_ptr<Context> context_ = nullptr;
};

class InitDeviceCommunicationCallback : public IpcCallback, public IpcWaiter {
 public:
  virtual ~InitDeviceCommunicationCallback() {}
  virtual Status Call(const IpcResponse* response, IpcContext* context) override;
};

}  // namespace ksana_llm
