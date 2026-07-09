/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ipc/functions/pd_v2_barrier_done.h"

#include "ipc/core/ipc_func_serialization.h"
#include "utils/logger.h"
#include "utils/ret_code.h"
#include "utils/status.h"

namespace ksana_llm {

Status PdV2BarrierDoneRequest::Serialize(void* ptr, size_t* bytes) const {
  SerializeInit(*bytes);
  SerializePod<int64_t>(req_id, ptr, *bytes);
  SerializePod<uint8_t>(status, ptr, *bytes);
  return Status();
}

Status PdV2BarrierDoneRequest::Deserialize(const void* ptr) {
  size_t offset = 0;
  req_id = DeserializePod<int64_t>(ptr, offset);
  status = DeserializePod<uint8_t>(ptr, offset);
  return Status();
}

void PdV2BarrierDoneCallable::SetCallback(BarrierCallback cb) {
  std::lock_guard<std::mutex> lk(cb_mu_);
  cb_ = std::move(cb);
}

Status PdV2BarrierDoneCallable::Call(const IpcRequest* request, IpcResponse* /*response*/, IpcContext* /*context*/) {
  const auto* real_req = reinterpret_cast<const PdV2BarrierDoneRequest*>(request);
  BarrierCallback cb_local;
  {
    std::lock_guard<std::mutex> lk(cb_mu_);
    cb_local = cb_;
  }
  if (cb_local) {
    cb_local(real_req->req_id, real_req->status);
  } else {
    KLLM_LOG_WARNING << "PdV2BarrierDoneCallable: no callback registered, "
                     << "dropping req_id=" << real_req->req_id << " status=" << static_cast<int>(real_req->status);
  }
  return Status();
}

}  // namespace ksana_llm
