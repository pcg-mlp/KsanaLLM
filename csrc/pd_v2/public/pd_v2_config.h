/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ksana_llm {
namespace pd_v2 {

// Connector deployment role: which process is loading the connector.
enum class DeploymentRole : uint8_t {
  kEngine = 0,
  kExecutor = 1,
};

// Node role in the disaggregated PD topology.
enum class NodeRole : uint8_t {
  kDecode = 0,
  kPrefill = 1,
};

// PeerSelector policy used by Decode engine to pick a Prefill peer.
enum class SelectorPolicy : uint8_t {
  kRoundRobin = 0,
  // LinUCB learned LOCAL-vs-REMOTE router. The remote peer is still chosen by
  // the kRoundRobin path; this policy only adds a contextual-bandit decision
  // on whether to keep a request's remaining prefill local. Opt-in.
  kLinUCB = 1,
};

// Engine-side configuration. Only meaningful when DeploymentRole == kEngine.
struct EngineConfig {
  // Local port for engine<->engine control plane (PrefillRequest / Ack /
  // Complete / Snapshot).
  uint16_t control_listen_port = 0;

  // Period at which Prefill engine pushes PrefillSnapshot to subscribed
  // Decode peers, used as a backstop against stale health views during idle.
  uint32_t snapshot_push_interval_ms = 1000;

  // Peer selection policy for Decode -> Prefill routing. Ignored on Prefill.
  // Default kRoundRobin spreads requests evenly across registered peers.
  SelectorPolicy selector_policy = SelectorPolicy::kRoundRobin;

  // pd_v2 Decode-side fast-path: when the per-request prefill workload
  // remaining after Decode's local prefix cache hit is ≤ this many
  // tokens, the request bypasses peer Prefill entirely and runs the
  // remaining prefill + autoregressive decode locally on this Decode
  // node. Reasoning: PD round-trip (control-plane notify + RDMA writes
  // for the cached prefix's tail blocks) is fixed cost; small
  // residuals don't earn back the overhead. Tuned per deployment;
  // 0 disables the fast-path (always forward to Prefill, current
  // behavior). Sourced from `setting.connector.pd_v2_local_complete_threshold`.
  int local_complete_threshold = 0;

  // ===== LinUCB router knobs (only consulted when selector_policy ==
  // kLinUCB). The contextual bandit decides, per request, whether to keep the
  // remaining prefill LOCAL vs forward to a Prefill peer. lambda (ridge reg)
  // is fixed at 1.0 and rewards are single-term, so these are the only
  // tunables. Sourced from setting.connector.pd_v2_linucb_*. =====

  // Exploration weight on the UCB bonus sqrt(x^T Ainv x). Larger => explores
  // the less-tried arm more aggressively.
  double linucb_alpha = 1.0;

  // TPOT budget (us). Drives both the tpot_headroom feature and the LOCAL
  // arm's reward: keeping prefill local is "good" only while per-step decode
  // latency stays well under this.
  double linucb_tpot_target_us = 50000.0;

  // LOCAL gate aims at linucb_tpot_target_us * this factor (default 0.85),
  // reserving headroom for the TPOT bump that running prefill locally induces.
  // < 1 prioritizes TPOT harder (prefill kept local less often); 1.0 aims at
  // the raw target.
  double linucb_tpot_safety_factor = 0.85;

  // TTFT budget (us). Drives the LOCAL arm's TtftCredit and the REMOTE arm
  // reward scale: keeping prefill local only earns credit once the remote
  // round-trip (TTFT proxy) would exceed this.
  double linucb_ttft_threshold_us = 1500000.0;

  // running_reqs count treated as "fully busy" — denominator of the idle
  // feature (1 - min(running/ref, 1)). Roughly the per-dp decode batch this
  // node sustains at target TPOT.
  double linucb_running_ref = 16.0;

  // Remaining-prefill token count treated as "too big to keep local" —
  // denominator of the remaining_small feature.
  double linucb_remaining_ref_tokens = 4096.0;

  // Round-trip latency (us) that maps to REMOTE reward 0. Submit->Complete
  // faster than this earns proportionally higher reward; same [0,1] scale as
  // the LOCAL reward so the two arms compare directly.
  double linucb_remote_lat_ref_us = 200000.0;

  // Mock-backend self-driving mode (R5 / M1): when > 0, the mock
  // connector auto-completes every SubmitPrefillRequest after the given
  // delay with a synthetic on_complete(kOk, fake_tokens) — no real
  // Prefill peer needed. fake_tokens is sized to the request's
  // sampling_config.max_new_tokens. The Decode KV pool stays
  // un-initialized so model output is garbage; this exists to exercise
  // the end-to-end Decode pipeline plumbing without standing up a
  // Prefill side. Ignored when backend_type != "mock". Default 0 =
  // disabled (use SimulateRemoteComplete from tests instead).
  uint32_t mock_auto_complete_delay_ms = 0;

  // attention TP size of the local (Decode) engine. Drives how many
  // DecodeTarget entries the Decode connector packs into outbound
  // PrefillRequest frames: one per local attn TP rank (= decode_TP). The
  // Prefill executor's ShouldHandleEntry plug-in fans these out by
  // matching decode_attn_tp_rank_in_group to local Prefill rank.
  // Sourced from RuntimeConfig::parallel_basic_config.attn_tensor_para_size
  // by inference_engine bootstrap. 0 = uninitialized (engine errors).
  uint32_t attn_tp_decode = 0;

  // Control-plane HostStagePool size (bytes), Prefill side. Decode peers
  // RDMA-WRITE the heavy prefill_req payload into this pool before
  // sending a small slot-reference notify (see
  // docs/technology/design/pd_v2_control_stage.md §5.1). Default 2 GB,
  // tunable via setting.connector.control_stage_pool_bytes.
  uint64_t control_stage_pool_bytes = 0;

  // Maximum number of concurrent Decode peers the stage pool partitions
  // across. Default 128 — designed for 1:128 fan-in.
  uint32_t control_stage_max_peers = 128;

  // Decode-side AcquireSlots timeout (ms). Hook thread waits this long
  // for stage credits before falling back to the failure path.
  // Default 5000 (5 s); negative = wait forever.
  int32_t control_stage_acquire_timeout_ms = 5000;

  // Decode-side: number of worker threads that run the async
  // SubmitPrefillRequest pipeline (build wire frame → acquire peer stage
  // slots → RDMA-WRITE staged payload → notify) OFF the EDS scheduling
  // thread. The scheduling thread only enqueues the request and returns
  // immediately; a worker performs the (potentially blocking) stage
  // handshake so a stuck/slow/full peer can never stall Decode scheduling
  // (head-of-line blocking of the whole engine). Each worker handles one
  // request end-to-end, so a single stuck RDMA WRITE only occupies one
  // worker while the rest keep dispatching to healthy peers. Default 4;
  // clamped to >= 1.
  uint32_t control_stage_async_workers = 4;
};

// Executor-side configuration. Only meaningful when DeploymentRole == kExecutor.
//
// All derived sizes (block_size, num_layers, element_size, transmit_this_rank)
// come from the local ksana_llm runtime and are not part of the wire format.
// transfer_layer_chunk_size is a pd_v2-owned knob and is kept here rather
// than reusing a V1 env getter.
struct ExecutorConfig {
  // Underlying transport: "tcp" or "rdma".
  std::string protocol = "rdma";

  // JSON describing the RDMA NIC priority matrix per rank, forwarded to the
  // Mooncake tent backend at Initialize time.
  std::string rdma_nic_priority_matrix_json;

  // Per-rank Mooncake TransferEngine RPC port. Must be unique across
  // all executor processes co-located on a host — under TP > 1 each
  // executor process owns one GPU, so each needs its own port. Set
  // by InferenceExecutor bootstrap as
  //   ConnectorConfig.pd_v2_executor_rpc_base_port + executor_rank
  // (default base 20000). 0 falls back to upstream Mooncake default
  // 12345 — collides across multi-rank, intended only for single-engine
  // single-executor dev launches.
  uint64_t rpc_port = 0;

  // pd_v2-owned chunk size (number of model layers per Mooncake submitTransfer
  // batch). 0 disables chunking and triggers transfer only at the model's
  // last layer.
  uint32_t transfer_layer_chunk_size = 0;

  // Per-process executor rank (0..tp-1). Used to suffix the Mooncake
  // server_name so each rank's KV pool registers as a distinct Redis
  // entry (.exec.r0, .exec.r1, ...). Without this, multi-rank executors
  // sharing one inference_addr stomp each other in metadata and peers
  // resolve a stale base_va that doesn't cover their target_offset.
  int executor_rank = 0;
};

// Top-level connector config; the same struct is loaded on both engine and
// executor processes, with `deployment_role` selecting which sub-config is
// honored.
struct PdV2ConnectorConfig {
  DeploymentRole deployment_role = DeploymentRole::kEngine;
  NodeRole node_role = NodeRole::kDecode;

  // Backend selection. "mooncake" -> libpd_v2_connector_mooncake.so
  // (Mooncake tent backend, real RDMA / TCP fallback). The factory
  // dispatches on this.
  std::string backend_type = "mooncake";

  // Logical cluster name; both ends of a peer pair must agree.
  std::string cluster_name;

  // Node identifier. Value MUST be copied from
  // ConnectorConfig.inference_addr (csrc/connector/config.h) by the
  // engine bootstrap code that constructs PdV2ConnectorConfig — pd_v2
  // does NOT invent a new identity concept. The field exists here only
  // so the connector .so stays decoupled from the Environment singleton
  // at runtime. inference_addr is cluster-unique (it's the host:port
  // each peer node uses to reach this engine), so it doubles as the
  // Mooncake server_name and the per-sender wire key in cross-node
  // PrefillRequest / PrefillRequestComplete frames.
  std::string inference_addr;

  // simple_router /v2/nodes endpoint, e.g. "http://router.local:8080".
  std::string router_addr;

  // Mooncake metadata server connection string. Forwarded to
  // mooncake::TransferEngine::init() at connector Initialize.
  // Supported schemes (per mooncake-transfer-engine v0.3.9):
  //   "http://host:port/"   — HTTP metadata server (this is what the
  //                            cluster-wide mooncake_master daemon
  //                            launches with --enable_http_metadata_server)
  //   "etcd://host:2379"     — etcd backend (USE_ETCD)
  //   "redis://host:6379"    — Redis backend (USE_REDIS)
  //   "P2PHANDSHAKE"         — single-machine, no external service
  //                            (each engine binds its own RPC port and
  //                            peers find each other via openSegment)
  // Empty defaults to "P2PHANDSHAKE" for development.
  std::string metadata_addr;

  EngineConfig engine;
  ExecutorConfig executor;
};

}  // namespace pd_v2
}  // namespace ksana_llm
