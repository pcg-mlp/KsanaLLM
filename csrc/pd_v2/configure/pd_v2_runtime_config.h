/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstdint>
#include <string>

#include "connector/config.h"  // GroupRole — reused public enum.

namespace ksana_llm {

// Runtime config for pd_v2 (token-based PD disaggregation v2).
//
// Why a separate struct from ConnectorConfig:
//   V1 PD derived `runtime_config_.is_decode_only = (group_role == DECODE)`
//   from ConnectorConfig. pd_v2 Decode runs prefill locally under the
//   local-bypass fast path, so that V1 derivation was wrong for pd_v2 and
//   caused the K7 deep-KV correctness bug (commit ae7cf355). The fix lived
//   inside the parser as an override but the structural problem — sharing
//   one struct with V1 — remained. PdV2RuntimeConfig is the structural
//   answer: pd_v2 read-sites point here, V1 read-sites point at
//   ConnectorConfig, no more cross-contamination of derived runtime flags.
//
// Population: YAML keys stay `setting.connector.*`. The parser populates
// both this struct and ConnectorConfig (twin assignments) so V1 code
// continues to work unchanged. Deletion of the ConnectorConfig pd_v2
// fields is deferred to a follow-up PR.
//
// Relationship to pd_v2::PdV2ConnectorConfig (csrc/pd_v2/public/pd_v2_config.h):
//   PdV2ConnectorConfig is the ABI struct passed across the
//   libpd_v2_connector_*.so boundary at factory time. PdV2RuntimeConfig
//   lives inside the inference_engine / inference_executor processes and
//   is read by callers that previously consulted ConnectorConfig. Naming
//   is intentionally distinct so the two roles don't get conflated.
struct PdV2RuntimeConfig {
  // True iff `setting.connector.backend == "v2"`. Callers gate pd_v2 logic
  // on this. Environment::GetPdV2RuntimeConfig returns RET_CONFIG_NOT_FOUND
  // when this is false, but the field is exposed so callers that received
  // a populated struct can still cheaply assert.
  bool is_pd_v2 = false;

  // Shared with ConnectorConfig (mirror semantics, populated by the same
  // parser block). pd_v2 callers read these from here so they never touch
  // ConnectorConfig.
  GroupRole group_role = GroupRole::NONE;
  std::string cluster_name;
  std::string inference_addr;
  std::string router_addr;
  std::string coordinator_addr;
  int heartbeat_interval_ms = 5000;

  // pd_v2-only. Mooncake metadata server connection string. Empty defaults
  // to "P2PHANDSHAKE" (single-machine, no external service). Supported
  // schemes: http://, etcd://, redis://, P2PHANDSHAKE.
  std::string metadata_addr;

  // pd_v2-only. AUTH password for metadata_addr=redis://...; forwarded by
  // inference_engine / inference_executor bootstrap to env var
  // MC_REDIS_PASSWORD which Mooncake reads at TransferEngine::init time.
  // Empty = no AUTH issued.
  std::string redis_password;

  // pd_v2-only. Number of model layers per Mooncake submitTransfer batch.
  // 0 disables chunking (whole-model in one batch at last layer); positive
  // N triggers a batch every N layers. Default 1 matches the design doc.
  uint32_t transfer_layer_chunk_size = 1;

  // pd_v2-only. Executor Mooncake RPC base port. Per-rank port = base +
  // executor_rank so multi-rank TP deployments don't collide on a single
  // port. Default 20000.
  uint32_t pd_v2_executor_rpc_base_port = 20000;

  // pd_v2-only. Decode-side "remaining prefill ≤ N stays local" threshold
  // (tokens). When (input_tokens.size() − prefix_cache_len) ≤ this value,
  // Decode bypasses peer Prefill and runs the residual prefill + decode
  // locally. 0 disables the fast-path (always forward).
  int pd_v2_local_complete_threshold = 0;

  // pd_v2-only. Decode peer-selection policy. 0=kRoundRobin (default),
  // 1=kLinUCB. kLinUCB turns on the contextual-bandit LOCAL-vs-REMOTE router
  // (the remote peer is still chosen by round-robin).
  int pd_v2_selector_policy = 0;

  // pd_v2-only. LinUCB router knobs, consulted only when
  // pd_v2_selector_policy == 1 (kLinUCB). See pd_v2_config.h EngineConfig for
  // semantics. lambda is fixed at 1.0; these are the only tunables.
  double pd_v2_linucb_alpha = 1.0;
  double pd_v2_linucb_tpot_target_us = 50000.0;
  double pd_v2_linucb_tpot_safety_factor = 0.85;
  double pd_v2_linucb_ttft_threshold_us = 1500000.0;
  double pd_v2_linucb_running_ref = 16.0;
  double pd_v2_linucb_remaining_ref_tokens = 4096.0;
  double pd_v2_linucb_remote_lat_ref_us = 200000.0;

  // pd_v2-only. Control-plane HostStagePool size on the *Prefill* side
  // (bytes). Decode peers RDMA-WRITE the heavy prefill_req payload here
  // before sending the small slot-reference notify. Default 2 GB; raise
  // for multimodal deploys with very large prompts. Must be a multiple
  // of 64 KiB (HostStagePool::kSlotSize). 0 means use the in-source
  // default (HostStagePool::kDefaultPoolBytes).
  uint64_t control_stage_pool_bytes = 0;

  // pd_v2-only. Maximum number of concurrent Decode peers the stage
  // pool partitions across. Static partition: each peer gets
  // pool_bytes / max_peers / 64 KiB slots. Sized for the design's
  // 128-peer fan-in target; raise if you have more Decode senders to
  // a single Prefill, lower to give each peer more slots.
  uint32_t control_stage_max_peers = 128;

  // pd_v2-only. Decode-side AcquireSlots timeout (ms). PdV2DecodeHook's
  // SubmitPrefillRequest blocks at most this long waiting for stage
  // credits before falling back to the failure path. Negative = wait
  // forever (not recommended in production — risks stalling the hook
  // thread on a stuck Prefill peer). Default 5000 (5s).
  int control_stage_acquire_timeout_ms = 5000;

  // pd_v2-only. Decode-side async submit worker thread count. The EDS
  // scheduling thread only enqueues each SubmitPrefillRequest and returns
  // immediately; these workers run the build → acquire stage slots →
  // RDMA-WRITE → notify pipeline off the scheduling thread so a stuck /
  // full / slow Prefill peer never stalls Decode scheduling. Each worker
  // handles one request end-to-end. Default 4; clamped to >= 1.
  int control_stage_async_workers = 4;

  // Debug helper. Mirrors ConnectorConfig::toString() — omits
  // redis_password to keep secrets out of logs.
  std::string toString() const {
    std::string result = "PdV2RuntimeConfig {\n";
    result += "  is_pd_v2: " + std::string(is_pd_v2 ? "true" : "false") + "\n";
    result += "  group_role: " + GroupRoleToString(group_role) + "\n";
    result += "  cluster_name: " + cluster_name + "\n";
    result += "  inference_addr: " + inference_addr + "\n";
    result += "  router_addr: " + router_addr + "\n";
    result += "  coordinator_addr: " + coordinator_addr + "\n";
    result += "  heartbeat_interval_ms: " + std::to_string(heartbeat_interval_ms) + "\n";
    result += "  metadata_addr: " + metadata_addr + "\n";
    result += "  transfer_layer_chunk_size: " + std::to_string(transfer_layer_chunk_size) + "\n";
    result += "  pd_v2_executor_rpc_base_port: " + std::to_string(pd_v2_executor_rpc_base_port) + "\n";
    result += "  pd_v2_local_complete_threshold: " + std::to_string(pd_v2_local_complete_threshold) + "\n";
    result += "  pd_v2_selector_policy: " + std::to_string(pd_v2_selector_policy) + "\n";
    result += "  control_stage_pool_bytes: " + std::to_string(control_stage_pool_bytes) + "\n";
    result += "  control_stage_max_peers: " + std::to_string(control_stage_max_peers) + "\n";
    result += "  control_stage_acquire_timeout_ms: " + std::to_string(control_stage_acquire_timeout_ms) + "\n";
    result += "  control_stage_async_workers: " + std::to_string(control_stage_async_workers) + "\n";
    result += "}";
    return result;
  }
};

}  // namespace ksana_llm
