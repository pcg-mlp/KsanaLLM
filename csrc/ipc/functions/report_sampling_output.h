/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ipc/core/ipc_func_types.h"
#include "runtime/sampling_output.h"
#include "utils/blocking_queue.h"

namespace ksana_llm {

class ReportSamplingOutputRequest : public IpcRequest {
 public:
  virtual ~ReportSamplingOutputRequest() {}
  virtual Status Serialize(void* ptr, size_t* bytes) const override;
  virtual Status Deserialize(const void* ptr) override;

  // The sampling output.
  std::shared_ptr<SamplingOutput> sampling_output = nullptr;
};

class ReportSamplingOutputResponse : public IpcResponse {
 public:
  virtual ~ReportSamplingOutputResponse() {}
};

class ReportSamplingOutputCallable : public IpcCallable {
 public:
  virtual ~ReportSamplingOutputCallable() {}
  virtual Status Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) override;

  void SetRecvQueue(std::shared_ptr<BlockingQueue<std::shared_ptr<SamplingOutput>>> schedule_output_queue);

 private:
  // Store received schdule output.
  std::shared_ptr<BlockingQueue<std::shared_ptr<SamplingOutput>>> sampling_output_queue_;
};

class ReportSamplingOutputCallback : public IpcCallback {
 public:
  virtual ~ReportSamplingOutputCallback() {}
};

}  // namespace ksana_llm
