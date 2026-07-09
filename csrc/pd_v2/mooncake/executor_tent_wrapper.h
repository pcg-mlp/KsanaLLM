/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "utils/status.h"

namespace mooncake {
class TransferEngine;
}  // namespace mooncake

namespace ksana_llm {
namespace pd_v2 {
namespace mooncake_backend {

// Executor-side wrapper around mooncake::TransferEngine. Used by
// MooncakeExecutorConnector for two things:
//   1. Boot-time: registerLocalMemory(KV pool) so peer Prefill can
//      RDMA-WRITE into it (mooncake handles the network protocol).
//   2. Per-request (Prefill side, future): submitTransfer to push KV
//      bytes to remote Decode KV pool. Decode side never calls
//      submitTransfer — it's a passive receive endpoint.
//
// Threading: registerLocalMemory called once at Boot. submitTransfer
// thread-safe per mooncake docs.
class ExecutorTentWrapper {
 public:
  ExecutorTentWrapper();
  ~ExecutorTentWrapper();

  ExecutorTentWrapper(const ExecutorTentWrapper&) = delete;
  ExecutorTentWrapper& operator=(const ExecutorTentWrapper&) = delete;

  // Boot-time init.
  //   metadata_addr     - same scheme as engine wrapper
  //   local_server_name - e.g. "{ip}:{rpc_port}"; mooncake convention.
  //                       Different from the engine's local_server_name
  //                       so peers can address engine and executor
  //                       separately within the same node.
  //   rpc_port          - per-rank port for Mooncake's TransferEngine.
  //                       Must be unique across all executor processes
  //                       co-located on the same host (TP > 1 deploys
  //                       one executor per GPU on one node, so each
  //                       process needs its own port). Caller is
  //                       responsible for assigning base_port +
  //                       executor_rank or similar; the wrapper does
  //                       not pick a default — passing 0 lets Mooncake
  //                       use its built-in default which collides
  //                       across multi-rank processes.
  Status Initialize(const std::string& metadata_addr, const std::string& local_server_name, uint64_t rpc_port);

  Status Shutdown();

  // Boot-time: register KV pool with mooncake so it's RDMA-accessible
  // by peer engines. `location` is the mooncake memory-class hint
  // ("cuda:0" / "cpu:0"); we'll usually pass "cuda:0" since KV pool
  // lives in device memory.
  Status RegisterKvPool(void* base, size_t total_bytes, const std::string& location);

  // Submit an RDMA WRITE batch to a remote KV pool buffer. Called from
  // PrefillIntakeWrapper / Prefill executor on this connector.
  // `target_server_name` resolves via openSegment internally;
  // `target_va` is the absolute remote VA (= remote_kv_pool_base_va +
  // block_id * block_size + chunk_layer_offset).
  Status SubmitWrite(const std::string& target_server_name, uint64_t target_va, void* source, size_t length);

  // Async variant: submit one-Request batch and return the BatchID
  // immediately without polling. Caller (PdV2KvDispatcher) tracks the
  // BatchID via MooncakeExecutorConnector::TrackInflightBatch and a
  // background barrier thread polls GetBatchStatus until COMPLETED.
  // 0 returned in *out_batch_id on failure with non-OK status.
  Status SubmitWriteAsync(const std::string& target_server_name, uint64_t target_va, void* source, size_t length,
                          uint64_t* out_batch_id);

  // One entry in a coalesced WRITE batch (see SubmitWriteBatchAsync).
  // All entries of a batch share the same target_server_name; only the
  // remote VA / local source / length differ per block.
  struct BatchWriteEntry {
    uint64_t target_va = 0;
    void* source = nullptr;
    size_t length = 0;
  };

  // Batched async variant of SubmitWriteAsync: resolve the segment ONCE,
  // allocate a single BatchID sized to entries.size(), and submit all
  // TransferRequests in one submitTransfer. Replaces N per-block
  // (openSegment + allocateBatchID + submitTransfer) sequences with a
  // single set of calls. Returns the BatchID immediately (no poll); the
  // batch reaches COMPLETED only when every request in it completes.
  // 0 in *out_batch_id on failure or empty input.
  Status SubmitWriteBatchAsync(const std::string& target_server_name, const std::vector<BatchWriteEntry>& entries,
                               uint64_t* out_batch_id);

  // Non-blocking status query for a batch returned from SubmitWriteAsync.
  // *out_completed is true when the batch reached COMPLETED;
  // *out_failed is true on FAILED / TIMEOUT / CANCELED / INVALID. Both
  // can be false simultaneously (still in flight).
  Status GetBatchStatus(uint64_t batch_id, bool* out_completed, bool* out_failed);

  // Best-effort cancel of a Mooncake BatchID via mooncake::TransferEngine
  // ::freeBatchID. Failures are logged but not surfaced.
  Status FreeBatchID(uint64_t batch_id);

  mooncake::TransferEngine* raw() { return engine_.get(); }

 private:
  std::unique_ptr<mooncake::TransferEngine> engine_;
  std::string local_server_name_;
  bool initialized_ = false;
};

}  // namespace mooncake_backend
}  // namespace pd_v2
}  // namespace ksana_llm
