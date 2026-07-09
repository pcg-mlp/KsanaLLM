/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstdint>
#include <functional>
#include <mutex>

#include "ipc/core/ipc_func_types.h"

namespace ksana_llm {

// Executor → Engine async report: all in-flight Mooncake submitTransfer
// batches for a pd_v2 request have drained (sender-side barrier). Engine
// receives this signal and is free to send PrefillRequestComplete back
// to the originating Decode peer.
//
// status: 0 = kOk (all batches COMPLETED), 1 = kFailed (any batch
// terminated FAILED / TIMEOUT / CANCELED). kCancelled is reported via
// HandleAbort, not via this barrier.
class PdV2BarrierDoneRequest : public IpcRequest {
 public:
  virtual ~PdV2BarrierDoneRequest() {}
  virtual Status Serialize(void* ptr, size_t* bytes) const override;
  virtual Status Deserialize(const void* ptr) override;

  int64_t req_id = -1;
  uint8_t status = 0;
};

class PdV2BarrierDoneResponse : public IpcResponse {
 public:
  virtual ~PdV2BarrierDoneResponse() {}
};

// Engine-side handler: receives the barrier-done signal and dispatches
// to the registered callback (PdV2PrefillIntakeWrapper::OnBarrierDone).
class PdV2BarrierDoneCallable : public IpcCallable {
 public:
  using BarrierCallback = std::function<void(int64_t req_id, uint8_t status)>;

  virtual ~PdV2BarrierDoneCallable() {}
  virtual Status Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) override;

  void SetCallback(BarrierCallback cb);

 private:
  std::mutex cb_mu_;
  BarrierCallback cb_;
};

class PdV2BarrierDoneCallback : public IpcCallback {
 public:
  virtual ~PdV2BarrierDoneCallback() {}
};

}  // namespace ksana_llm
