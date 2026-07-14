/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ipc/core/ipc_func_types.h"
#include "runtime/schedule_output.h"

namespace ksana_llm {

class DispatchScheduleOutputRequest : public IpcRequest {
 public:
  virtual ~DispatchScheduleOutputRequest() {}
  virtual Status Serialize(void* ptr, size_t* bytes) const override;
  virtual Status Deserialize(const void* ptr) override;

  // The schdule result.
  std::shared_ptr<ScheduleOutput> schedule_output = nullptr;

  // The serialized packet.
  Packet* serialized_packet = nullptr;
};

class DispatchScheduleOutputResponse : public IpcResponse {
 public:
  virtual ~DispatchScheduleOutputResponse() {}
};

class DispatchScheduleOutputCallable : public IpcCallable {
 public:
  virtual ~DispatchScheduleOutputCallable() {}
  virtual Status Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) override;

  void SetRecvQueue(std::shared_ptr<BlockingQueue<std::shared_ptr<ScheduleOutput>>> schedule_output_queue);

 private:
  // Store received schdule output.
  std::shared_ptr<BlockingQueue<std::shared_ptr<ScheduleOutput>>> schedule_output_queue_;
};

class DispatchScheduleOutputCallback : public IpcCallback {
 public:
  virtual ~DispatchScheduleOutputCallback() {}
};

}  // namespace ksana_llm
