/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ipc/functions/get_kv_pool_base_va.h"

#include "ipc/core/ipc_func_serialization.h"
#include "utils/logger.h"
#include "utils/ret_code.h"
#include "utils/status.h"

namespace ksana_llm {

Status GetKvPoolBaseVaResponse::Serialize(void* ptr, size_t* bytes) const {
  SerializeInit(*bytes);
  SerializePod<uint8_t>(registered ? 1 : 0, ptr, *bytes);
  SerializePod<uint64_t>(kv_pool_base_va, ptr, *bytes);
  SerializePod<uint64_t>(total_bytes, ptr, *bytes);
  SerializePod<uint64_t>(block_size, ptr, *bytes);
  SerializePod<uint64_t>(num_blocks, ptr, *bytes);
  SerializePod<uint64_t>(num_layers, ptr, *bytes);
  return Status();
}

Status GetKvPoolBaseVaResponse::Deserialize(const void* ptr) {
  size_t offset = 0;
  registered = DeserializePod<uint8_t>(ptr, offset) != 0;
  kv_pool_base_va = DeserializePod<uint64_t>(ptr, offset);
  total_bytes = DeserializePod<uint64_t>(ptr, offset);
  block_size = DeserializePod<uint64_t>(ptr, offset);
  num_blocks = DeserializePod<uint64_t>(ptr, offset);
  num_layers = DeserializePod<uint64_t>(ptr, offset);
  return Status();
}

void GetKvPoolBaseVaCallable::SetQueryFn(GetKvPoolBaseVaQueryFn fn) { query_fn_ = std::move(fn); }

Status GetKvPoolBaseVaCallable::Call(const IpcRequest* /*request*/, IpcResponse* response, IpcContext* /*context*/) {
  GetKvPoolBaseVaResponse* real_rsp = reinterpret_cast<GetKvPoolBaseVaResponse*>(response);
  if (!query_fn_) {
    real_rsp->registered = false;
    return Status();
  }
  real_rsp->registered = query_fn_(&real_rsp->kv_pool_base_va, &real_rsp->total_bytes, &real_rsp->block_size,
                                   &real_rsp->num_blocks, &real_rsp->num_layers);
  return Status();
}

}  // namespace ksana_llm
