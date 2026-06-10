/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "configure/context.h"
#include "configure/environment.h"
#include "ipc/core/ipc_func_types.h"
#include "ipc/core/ipc_func_utils.h"
#include "utils/waiter.h"

namespace ksana_llm {

#ifdef ENABLE_MULTICAST
class InitMulticastCommunicationRequest : public IpcRequest {
 public:
  virtual ~InitMulticastCommunicationRequest() {}
  virtual Status Serialize(void* ptr, size_t* bytes) const override;
  virtual Status Deserialize(const void* ptr) override;

  int64_t unique_op_id{-1};
  int broadcast_id{-1};
};

class InitMulticastCommunicationResponse : public IpcResponse {
 public:
  virtual ~InitMulticastCommunicationResponse() {}
};

class InitMulticastCommunicationCallable : public IpcCallable {
 public:
  virtual ~InitMulticastCommunicationCallable() {}
  virtual Status Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) override;

  void SetContext(std::shared_ptr<Context> context);

 private:
  std::shared_ptr<Context> context_;
};

class InitMulticastCommunicationCallback : public IpcCallback, public IpcWaiter {
 public:
  virtual ~InitMulticastCommunicationCallback() {}
  virtual Status Call(const IpcResponse* response, IpcContext* context) override;
};
#endif

}  // namespace ksana_llm
