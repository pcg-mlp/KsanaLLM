/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstdint>

#include "pd_v2/configure/pd_v2_runtime_config.h"
#include "pd_v2/public/pd_v2_config.h"

namespace ksana_llm {
namespace pd_v2 {

// Map the process-internal PdV2RuntimeConfig (populated from YAML) onto the
// cross-.so ABI PdV2ConnectorConfig. Shared by inference_engine (kEngine) and
// inference_executor (kExecutor) so the two bootstraps can't silently drift
// apart. The identity fields common to both deployment roles are filled by
// FillCommonConnectorConfig; the engine / executor builders add their
// role-specific sub-config.

inline void FillCommonConnectorConfig(const PdV2RuntimeConfig& rt, PdV2ConnectorConfig* cfg) {
  cfg->node_role = (rt.group_role == GroupRole::PREFILL) ? NodeRole::kPrefill : NodeRole::kDecode;
  cfg->backend_type = "mooncake";
  cfg->cluster_name = rt.cluster_name;
  cfg->inference_addr = rt.inference_addr;
  cfg->metadata_addr = rt.metadata_addr;
  cfg->router_addr = rt.router_addr;
}

// Engine-side (DeploymentRole::kEngine) connector config. `attn_tp` is this
// engine's attention TP size (RuntimeConfig-derived), used to size the
// DecodeTarget fan-out. LinUCB knobs are only consulted by the connector when
// selector_policy == kLinUCB; default kRoundRobin preserves current behavior.
inline PdV2ConnectorConfig BuildEngineConnectorConfig(const PdV2RuntimeConfig& rt, uint32_t attn_tp) {
  PdV2ConnectorConfig cfg;
  cfg.deployment_role = DeploymentRole::kEngine;
  FillCommonConnectorConfig(rt, &cfg);

  cfg.engine.attn_tp_decode = attn_tp;
  cfg.engine.local_complete_threshold = rt.pd_v2_local_complete_threshold;

  const bool use_linucb = rt.pd_v2_selector_policy == static_cast<int>(SelectorPolicy::kLinUCB);
  cfg.engine.selector_policy = use_linucb ? SelectorPolicy::kLinUCB : SelectorPolicy::kRoundRobin;
  cfg.engine.linucb_alpha = rt.pd_v2_linucb_alpha;
  cfg.engine.linucb_tpot_target_us = rt.pd_v2_linucb_tpot_target_us;
  cfg.engine.linucb_tpot_safety_factor = rt.pd_v2_linucb_tpot_safety_factor;
  cfg.engine.linucb_ttft_threshold_us = rt.pd_v2_linucb_ttft_threshold_us;
  cfg.engine.linucb_running_ref = rt.pd_v2_linucb_running_ref;
  cfg.engine.linucb_remaining_ref_tokens = rt.pd_v2_linucb_remaining_ref_tokens;
  cfg.engine.linucb_remote_lat_ref_us = rt.pd_v2_linucb_remote_lat_ref_us;

  // Control-stage knobs (Prefill sizes its HostStagePool, Decode bounds its
  // AcquireSlots wait). See docs/technology/design/pd_v2_control_stage.md §5/§8.
  cfg.engine.control_stage_pool_bytes = rt.control_stage_pool_bytes;
  cfg.engine.control_stage_max_peers = rt.control_stage_max_peers;
  cfg.engine.control_stage_acquire_timeout_ms = rt.control_stage_acquire_timeout_ms;
  cfg.engine.control_stage_async_workers =
      rt.control_stage_async_workers > 0 ? static_cast<uint32_t>(rt.control_stage_async_workers) : 1u;
  return cfg;
}

// Executor-side (DeploymentRole::kExecutor) connector config. `executor_rank`
// is this process's rank (0..tp-1); the per-rank RPC port is base + rank so
// co-located TP ranks (one executor process per GPU) don't collide on the
// upstream Mooncake default port.
inline PdV2ConnectorConfig BuildExecutorConnectorConfig(const PdV2RuntimeConfig& rt, int executor_rank) {
  PdV2ConnectorConfig cfg;
  cfg.deployment_role = DeploymentRole::kExecutor;
  FillCommonConnectorConfig(rt, &cfg);

  cfg.executor.executor_rank = executor_rank;
  cfg.executor.rpc_port = static_cast<uint64_t>(rt.pd_v2_executor_rpc_base_port) + static_cast<uint64_t>(executor_rank);
  return cfg;
}

}  // namespace pd_v2
}  // namespace ksana_llm
