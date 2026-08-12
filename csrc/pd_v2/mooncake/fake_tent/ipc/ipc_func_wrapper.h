/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
// FAKE ipc/ipc_func_wrapper.h — placed on the include path BEFORE the real
// csrc/ipc/ipc_func_wrapper.h for the pd_v2 connector coverage tests. The
// real header pulls the entire executor/distributed/cache_manager stack;
// the connectors only touch two methods, so this minimal stand-in lets the
// real connector .cpp compile + link without that heavy graph.
//
// Behaviour wires the engine-side RPC (PdV2GetKvPoolBaseVa) straight to the
// executor-side query fn (SetPdV2KvPoolQueryFn) so an in-process engine +
// executor connector pair resolves KV-pool VAs end-to-end.
#pragma once

#include <cstdint>
#include <functional>
#include <utility>

#include "utils/ret_code.h"
#include "utils/status.h"

namespace ksana_llm {

class IpcFuncWrapper {
 public:
  using KvQueryFn = std::function<bool(uint64_t*, uint64_t*, uint64_t*, uint64_t*, uint64_t*)>;

  void SetPdV2KvPoolQueryFn(KvQueryFn fn) { kv_query_fn_ = std::move(fn); }

  Status PdV2GetKvPoolBaseVa(int /*rank*/, uint64_t* kv_pool_base_va, uint64_t* total_bytes, uint64_t* block_size,
                             uint64_t* num_blocks, uint64_t* num_layers) {
    if (!kv_query_fn_) return Status(RET_RUNTIME_FAILED, "fake ipc: no kv query fn registered");
    uint64_t b = 0, t = 0, s = 0, n = 0, l = 0;
    bool ok = kv_query_fn_(&b, &t, &s, &n, &l);
    if (kv_pool_base_va) *kv_pool_base_va = b;
    if (total_bytes) *total_bytes = t;
    if (block_size) *block_size = s;
    if (num_blocks) *num_blocks = n;
    if (num_layers) *num_layers = l;
    return ok ? Status() : Status(RET_RUNTIME_FAILED, "fake ipc: kv pool not ready");
  }

  // Barrier report sink. PdV2KvBarrier calls this once a request's in-flight
  // KV batches drain; the fake just records the last report for assertions.
  // Matches the real signature (csrc/ipc/ipc_func_wrapper.h).
  Status ReportPdV2BarrierDone(int64_t req_id, uint8_t status) {
    last_barrier_req_id_ = req_id;
    last_barrier_status_ = status;
    ++barrier_report_count_;
    return Status();
  }

  // Test-visible: the fn registered by MooncakeExecutorConnector.
  KvQueryFn kv_query_fn_;

  // Test-visible: last barrier report (see ReportPdV2BarrierDone).
  int64_t last_barrier_req_id_ = -1;
  uint8_t last_barrier_status_ = 0;
  int barrier_report_count_ = 0;
};

}  // namespace ksana_llm
