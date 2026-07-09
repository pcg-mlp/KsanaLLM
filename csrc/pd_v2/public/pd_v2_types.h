/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "msgpack.hpp"

namespace ksana_llm {
namespace pd_v2 {

// Mirrors `mooncake::tent::Notification` structurally. Every cross-node
// message in pd_v2 (PrefillRequest / Complete / Snapshot / StageAck,
// and the EOP hidden_commit marker) rides on one of these as
// `{name = "pd_v2/<channel>", msg = <serialized payload>}`. The mock
// connector's local-IPC backend synthesizes the same struct; the real
// mooncake backend forwards to `tent::Notification` directly.
struct Notification {
  std::string name;  // channel, e.g. "pd_v2/prefill_req"
  std::string msg;   // opaque payload bytes (typically from
                     // wire/prefill_request_wire.h)
};

// Status reported from the connector executor side back up to the engine
// process via the on_complete callback (Decode side) or via PD_V2_COMPLETE
// IPC (Prefill side).
enum class PdCompleteStatus : uint8_t {
  kOk = 0,
  // Peer was healthy at admission but the in-flight transfer reported a CQE
  // error or the peer was marked unhealthy mid-flight (segment_id revoked).
  kFailed = 1,
  // Decode engine cancelled the request before completion (e.g. client
  // disconnect, abort_waiter signalled).
  kCancelled = 2,
};

// Snapshot of a single peer's resource state, exchanged on
// PrefillRequestComplete / periodic PrefillSnapshot.
// PeerSelector keeps these per peer and uses them as the sole input to
// Select() — there is no reservation counter or in-flight bookkeeping.
//
// Carried inline in several wire messages, so it's msgpack-serializable.
// MAP encoding (field name as key) lets new fields land without breaking
// rolling Prefill/Decode upgrades.
struct PeerHealth {
  uint64_t segment_id = 0;
  std::string server_name;
  uint32_t free_kv_blocks = 0;
  uint64_t last_snapshot_ms = 0;
  bool healthy = false;
  // Load features for snapshot-driven routing (LinUCB et al.). All MAP-encoded
  // so a peer running an older build simply omits them and the reader takes
  // the zero default — rolling upgrade safe.
  uint32_t future_free_kv_blocks = 0;   // freed-but-not-yet-merged blocks (swapout in flight)
  uint32_t waiting_reqs = 0;            // queued (prefilling) request count
  uint32_t running_reqs = 0;            // running (decoding) request count
  uint64_t pending_prefill_tokens = 0;  // sum of planning sequence lengths over waiting reqs
  uint32_t step_token_num = 0;          // EWMA of tokens processed per schedule step
  uint32_t step_latency_us = 0;         // EWMA of per-step wall time (microseconds)

  MSGPACK_DEFINE_MAP(segment_id, server_name, free_kv_blocks, last_snapshot_ms, healthy, future_free_kv_blocks,
                     waiting_reqs, running_reqs, pending_prefill_tokens, step_token_num, step_latency_us);
};

// Per-rank Decode KV-pool descriptor as held inside ksana_llm Request /
// InferRequest / ExecutorInferRequest (i.e. **off the wire**). Separated
// from wire/DecodeTarget to keep `runtime/` free of msgpack — wire side
// converts to/from this when packing/unpacking PrefillRequest.
//
// Semantics identical to wire DecodeTarget; see
// pd_v2/wire/prefill_request_wire.h for the per-field rationale.
struct DecodeTargetLite {
  uint32_t decode_attn_tp_rank_in_group = 0;
  uint64_t kv_pool_base_va = 0;
  std::vector<int> kv_block_ids;
  // pd_v2 prefix-cache hint. Decode side computes:
  //   cached_block_count        = req->prefix_cache_len / block_token_num
  //   decode_prefix_cache_offset = req->prefix_cache_len  (token-level)
  // Prefill dispatcher iterates kv_block_ids[cached_block_count..] for
  // RDMA writes — leading cached blocks already valid on Decode and are
  // skipped. See wire DecodeTarget for the full rationale.
  uint32_t cached_block_count = 0;
  uint32_t decode_prefix_cache_offset = 0;
};

// Prefill-side per-request metrics shipped back to Decode on completion, so the Decode
// access log can attribute the disaggregation cost end-to-end without an offline join.
// All durations are computed on the Prefill node from its OWN monotonic clock (queue /
// compute / send-tail segments of Prefill residency) — they are deltas, never cross-node
// absolute timestamps, which would be meaningless given per-process clock epochs.
struct PdPrefillMetrics {
  int32_t prefix_cache_hit_tokens = 0;  // prefix cache hit on the Prefill node (the real one)
  uint64_t queue_us = 0;                // Prefill 排队: scheduled - arrive
  uint64_t compute_us = 0;              // Prefill 计算: finish - scheduled
  uint64_t send_us = 0;                 // Prefill 收尾: send - finish (barrier ack + sampler drain)
};

// Fired from the connector engine (Decode) side when a Submit'd request
// finishes. The callback receives the tokens Prefill produced
// (1 main + ptp_token_num drafts) so the Decode hook can splice them
// directly into req->output_tokens / draft_tokens before re-entering EDS.
// `generated_tokens` is empty unless status == kOk.
// `prefill_metrics` carries Prefill-side timing/prefix-cache for the access log.
// `prefill_peer` is the selected peer Prefill engine's server_name (empty for local prefill).
// `ctx` is the opaque pointer passed to SubmitPrefillRequest.
using PdRequestCompleteCb = void (*)(uint64_t req_id, PdCompleteStatus status, const std::vector<int>& generated_tokens,
                                     const PdPrefillMetrics& prefill_metrics, const std::string& prefill_peer,
                                     void* ctx);

// Fired by the Prefill-side connector when a PrefillRequest is received
// from a Decode peer. The callback (PdV2PrefillIntakeWrapper) builds a
// minimal InferRequest from the carried sampling config + input_tokens
// and submits it to Prefill EDS with infer_stage=kContext and
// max_new_tokens = sampling.max_new_tokens (already pinned to
// 1 + ptp_token_num by the Decode hook).
//
// decode_engine_inference_addr is the per-node key the connector remembers
// for reverse-routing the eventual prefill_complete back. It's the only
// sender-identity field — no separate
// hash uint64 (server_name is already globally unique within a cluster).
struct IncomingPrefillRequest {
  // Mooncake server_name of the originating Decode engine; used both as
  // the per-sender disambiguation key for concurrent req_ids and as the
  // reverse-route target for replies.
  std::string decode_engine_inference_addr;
  uint64_t req_id;
  std::vector<int> input_tokens;
  // Pointer to the wire SamplingConfigWire owned by the connector for the
  // duration of the callback. PrefillIntakeWrapper casts and translates to
  // ksana_llm::SamplingConfig. Opaque to keep public/ free of wire/ deps.
  const void* sampling_config_wire = nullptr;
  // Pointer to the wire DecodeTarget[] owned by the connector for the
  // duration of the callback. Used by Prefill executor to direct per-rank
  // KV writes back to the Decode KV pool.
  const void* decode_targets_wire = nullptr;
  uint32_t num_decode_targets = 0;
  // Decode-side attn_dp_group_id this request belongs to. Combined with
  // each DecodeTarget's decode_attn_tp_rank_in_group it yields the
  // physical Decode device_id so the Prefill executor can address the
  // correct `.exec.r<device_id>` Mooncake server_name.
  uint32_t decode_attn_dp_group_id = 0;
};

using PdIncomingReqCb = void (*)(const IncomingPrefillRequest& req, void* ctx);

}  // namespace pd_v2
}  // namespace ksana_llm
