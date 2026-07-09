/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>

#include "ipc/core/ipc_func_types.h"

namespace ksana_llm {

// Engine → Executor query for the pd_v2 KV pool base VA and shape that
// the executor process registered with Mooncake at boot. The engine
// connector uses this to fill DecodeTarget.kv_pool_base_va in outbound
// PrefillRequest frames; the value lives in the executor process address
// space, but Mooncake only needs the VA + segment_id to RDMA-write into
// it (the remote engine's openSegment resolves the SegmentID separately,
// the VA is just a passive offset on the remote side).
class GetKvPoolBaseVaRequest : public IpcRequest {
 public:
  virtual ~GetKvPoolBaseVaRequest() {}
};

class GetKvPoolBaseVaResponse : public IpcResponse {
 public:
  virtual ~GetKvPoolBaseVaResponse() {}
  virtual Status Serialize(void* ptr, size_t* bytes) const override;
  virtual Status Deserialize(const void* ptr) override;

  // OK when the executor has called RegisterKvPool already; if the
  // engine queries too early the executor responds with registered=false
  // and the engine retries (the executor publishes via RegisterKvPool
  // synchronously during boot, so post-boot calls always succeed).
  bool registered = false;
  uint64_t kv_pool_base_va = 0;
  uint64_t total_bytes = 0;
  uint64_t block_size = 0;
  uint64_t num_blocks = 0;
  uint64_t num_layers = 0;
};

// Type-erased query function bound by the executor-side pd_v2 connector
// (signature mirrors PdV2ConnectorInterface — backend lives in a .so).
// Out params follow the order: base_va, total_bytes, block_size,
// num_blocks, num_layers. Returns true if the KV pool is registered.
using GetKvPoolBaseVaQueryFn =
    std::function<bool(uint64_t* /*base_va*/, uint64_t* /*total_bytes*/, uint64_t* /*block_size*/,
                       uint64_t* /*num_blocks*/, uint64_t* /*num_layers*/)>;

class GetKvPoolBaseVaCallable : public IpcCallable {
 public:
  virtual ~GetKvPoolBaseVaCallable() {}
  virtual Status Call(const IpcRequest* request, IpcResponse* response, IpcContext* context) override;

  // Bound by ksana_executor boot after the pd_v2 executor connector is
  // constructed. The callable invokes this fn to answer engine queries;
  // it returns false (registered=false) until RegisterKvPool runs.
  void SetQueryFn(GetKvPoolBaseVaQueryFn fn);

 private:
  GetKvPoolBaseVaQueryFn query_fn_;
};

class GetKvPoolBaseVaCallback : public IpcCallback {
 public:
  virtual ~GetKvPoolBaseVaCallback() {}
};

}  // namespace ksana_llm
