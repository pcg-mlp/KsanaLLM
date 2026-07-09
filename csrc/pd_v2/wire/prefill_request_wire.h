/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "msgpack.hpp"

#include "pd_v2/public/pd_v2_types.h"
#include "pd_v2/wire/sampling_config_wire.h"

// Strong-typed enums are msgpack-friendly via MSGPACK_ADD_ENUM. Must be at
// global scope, before any MSGPACK_DEFINE_MAP that references the enum.
MSGPACK_ADD_ENUM(ksana_llm::pd_v2::PdCompleteStatus);

namespace ksana_llm {
namespace pd_v2 {

// All wire types use msgpack MAP encoding (field names as keys), so a
// rolling Prefill/Decode upgrade can land new fields without breaking
// either side: unknown fields are dropped, missing fields fall back to
// the in-source default. Frame envelope (u32 type + u32 body_len + body)
// is added by WrapFrame below; msgpack handles only the body bytes.

// Per-rank receive descriptor inside a PrefillRequest. One DecodeTarget
// per Decode attn_TP rank in the destination group; size of
// decode_targets[] equals attn_TP_decode (not attn_TP_prefill).
//
// Token-based: only the KV pool side is described — Prefill
// returns generated_tokens via PrefillRequestComplete, no hidden buffer
// transfer.
struct DecodeTarget {
  // The Decode attn_TP rank (within the destination DP group) that this
  // entry targets. The Prefill executor's ShouldHandleEntry plug-in
  // matches this against the local rank to decide whether the local rank
  // participates.
  uint32_t decode_attn_tp_rank_in_group = 0;

  // Remote VA of block 0 of the Decode KV pool. Prefill executor writes
  // via Mooncake submitTransfer(target_id = openSegment(
  //   decode_engine_inference_addr), target_offset = kv_pool_base_va +
  //   kv_block_ids[i] * block_size + chunk_layer_offset).
  // The Mooncake SegmentID is per-engine (one per remote process), not
  // per-buffer, and is resolved by the Prefill side at openSegment time;
  // it is NOT carried on the wire. block_size / num_layers come from the
  // RegisterKvPool boot metadata (kept in the engine's local cache table).
  uint64_t kv_pool_base_va = 0;

  // Logical block ids in the Decode KV pool. Receiver assigned them at
  // admission; sender does not interpret them, only uses them as offsets.
  std::vector<int> kv_block_ids;

  // pd_v2 prefix-cache awareness. When Decode admits a request and finds
  // the leading N tokens already cached in its local KV pool (shared from
  // a prior request that had the same prompt prefix), it carries those
  // block_ids in `kv_block_ids` (so this list is always the FULL request
  // span — debugging stays simple) but tells Prefill "skip writing the
  // first `cached_block_count` of them; their K/V is already valid on
  // me". Prefill executor's dispatcher iterates `kv_block_ids` from
  // index `cached_block_count` onwards for per-block RDMA writes.
  //
  // `decode_prefix_cache_offset` is the same notion expressed in tokens
  // (= `cached_block_count * block_token_num` for block-aligned hits;
  // the runtime path on Decode side uses token granularity for prefix
  // cache match). Both fields are derivable from each other given
  // block_token_num, but block_token_num isn't carried on the wire and
  // both sides assume the same value, so we ship both for clarity.
  // Prefill dispatcher uses cached_block_count (block-level granularity
  // matches its loop). Token-level field exists for logging / future
  // partial-block prefix cache.
  uint32_t cached_block_count = 0;
  uint32_t decode_prefix_cache_offset = 0;

  // kv_block_ids removed from wire serialization — carried in
  // PrefillStagePayload to avoid exceeding the 64 KB Tent notify limit
  // on 128K+ token requests (8 targets × ~6 KB block IDs × 1.33 base64
  // blows past 64 KB). Receiver populates this field from the stage
  // payload after RDMA read.
  MSGPACK_DEFINE_MAP(decode_attn_tp_rank_in_group, kv_pool_base_va, cached_block_count, decode_prefix_cache_offset);

  // Converter from wire to runtime-local form (DecodeTargetLite from
  // pd_v2/public/pd_v2_types.h). PrefillIntakeWrapper deserializes the
  // wire frame, then converts each entry into the lite struct that lives
  // on the synthesized Request — so runtime/ never pulls msgpack.
  DecodeTargetLite ToLite() const {
    DecodeTargetLite lite;
    lite.decode_attn_tp_rank_in_group = decode_attn_tp_rank_in_group;
    lite.kv_pool_base_va = kv_pool_base_va;
    lite.kv_block_ids = kv_block_ids;
    lite.cached_block_count = cached_block_count;
    lite.decode_prefix_cache_offset = decode_prefix_cache_offset;
    return lite;
  }
};

// Engine-to-engine PrefillRequest frame. Sent over the Mooncake notify
// control plane at admission time. Prefill receives it on its connector
// engine thread, reads the heavy stage payload (input_tokens) from the
// HostStagePool slot range referenced by stage_*, builds a minimal
// Prefill-side InferRequest (with sampling_config translated from
// sampling_config_wire and max_new_tokens already pinned to
// 1 + ptp_token_num by Decode), and submits it to EDS.
//
// v5.2 (control-stage): input_tokens are NOT carried inline — they live
// in the stage payload (PrefillStagePayload, msgpack), which Decode RDMA
// WRITEs into the slot range described by stage_start_slot /
// stage_n_slots / stage_payload_bytes. This keeps the notify body well
// under the 64 KB Tent notify limit even for 32K+ token prefills. See
// docs/technology/design/pd_v2_control_stage.md §5.1.
struct PrefillRequest {
  // Mooncake server_name of the Decode engine that sent this request.
  // Serves two purposes simultaneously: (1) the per-sender disambiguation
  // key for concurrent req_ids from different Decode nodes, and (2) the
  // reverse-route target for prefill_complete notifies sent back via
  // engine_mooncake.OpenPeer(decode_engine_inference_addr).
  std::string decode_engine_inference_addr;

  // Decode-side req_id (engine-local counter); not unique across Decode
  // nodes — paired with decode_engine_inference_addr to form a global key.
  uint64_t req_id = 0;

  // Stage payload location in the *Prefill-side* HostStagePool. The
  // partition base for this Decode peer is fixed at peer registration
  // time, so start_slot here is the absolute slot index (already
  // partition-base-added) for direct SlotPtr() lookup on the receiver.
  // n_slots = ceil(payload_bytes / kSlotSize); payload_bytes is the
  // exact byte length to feed into DeserializePrefillStagePayload — the
  // tail of the last slot is uninitialized/garbage.
  uint32_t stage_start_slot = 0;
  uint32_t stage_n_slots = 0;
  uint32_t stage_payload_bytes = 0;

  // Sampling intent (mirror of ksana_llm::SamplingConfig essential
  // fields). Decode hook overrides max_new_tokens to 1 + ptp_token_num
  // before sending so Prefill produces exactly the tokens Decode needs.
  SamplingConfigWire sampling_config;

  // size = attn_TP_decode (number of Decode ranks in the receiving DP
  // group), not attn_TP_prefill. ShouldHandleEntry plug-in maps each
  // entry to the appropriate Prefill rank.
  std::vector<DecodeTarget> decode_targets;

  // Which Decode attn_dp group owns this request — assigned at admission
  // by BasicWorkloadBalancer. Combined with
  // decode_targets[r].decode_attn_tp_rank_in_group it yields the physical
  // Decode device_id (= decode_attn_dp_group_id * attn_TP_decode + r),
  // which is the suffix index in the executor's Mooncake server_name
  // (`<inference_addr>.exec.r<device_id>`). Prefill side composes the
  // target server_name with this offset; without it, Decode attn_DP > 1
  // collapses every request to attn_dp_group=0.
  uint32_t decode_attn_dp_group_id = 0;

  MSGPACK_DEFINE_MAP(decode_engine_inference_addr, req_id, stage_start_slot, stage_n_slots, stage_payload_bytes,
                     sampling_config, decode_targets, decode_attn_dp_group_id);
};

// Body written by Decode into the Prefill HostStagePool slot range
// described by PrefillRequest{stage_start_slot, stage_n_slots,
// stage_payload_bytes}. Standalone msgpack so the Prefill connector can
// decode it without parsing the full PrefillRequest envelope. Kept
// minimal on purpose: only fields heavy enough to risk crossing the 64
// KB notify limit go here. Today that's `input_tokens` (32K prompt =
// 128 KB int32s); other large fields (multimodal embeddings) will be
// added here when introduced rather than re-introducing a multi-channel
// path.
struct PrefillStagePayload {
  std::vector<int> input_tokens;
  std::vector<int> kv_block_ids;

  MSGPACK_DEFINE_MAP(input_tokens, kv_block_ids);
};

// Sent by Prefill to Decode after the stage payload has been consumed,
// releasing the slot range so Decode's PeerCredits ring can hand the
// slots back to the next request. Routed implicitly by the notify name
// (Decode receives only its own messages), so no decode_engine_inference_addr
// field is needed.
//
// err_code is opaque to the wire; 0 = success, non-zero values are
// listed in docs/technology/design/pd_v2_control_stage.md §6.2 (the
// only case Decode currently treats specially is "deserialization
// failed"; other errors fall through to the existing PrefillRequestComplete
// failure path).
struct StageAck {
  uint64_t req_id = 0;
  uint32_t start_slot = 0;
  uint32_t n_slots = 0;
  uint32_t err_code = 0;

  MSGPACK_DEFINE_MAP(req_id, start_slot, n_slots, err_code);
};

// Completion message sent by Prefill back to Decode after Prefill's
// first-step LM_head + sampler produced the requested tokens.
struct PrefillRequestComplete {
  // Decode-local req_id paired with the implicit notify route. Decode
  // engine receives only its own messages (notify is addressed by
  // Mooncake server_name), so req_id alone is enough to look up the
  // in-flight admission on the receiver side.
  uint64_t req_id = 0;
  PdCompleteStatus status = PdCompleteStatus::kOk;
  PeerHealth snapshot;

  // 1 main token + (max_new_tokens - 1) draft tokens, produced by
  // Prefill's first-step LM_head + sampler. Decode hook splices into
  // req->output_tokens and req->draft_tokens. Empty iff status != kOk.
  std::vector<int> generated_tokens;

  // Prefill-side metrics for the Decode access log (see PdPrefillMetrics). Durations are
  // Prefill-clock deltas (queue/compute/send-tail); prefix-cache is the Prefill hit count.
  int32_t pf_prefix_cache_hit_tokens = 0;
  uint64_t pf_queue_us = 0;
  uint64_t pf_compute_us = 0;
  uint64_t pf_send_us = 0;

  MSGPACK_DEFINE_MAP(req_id, status, snapshot, generated_tokens, pf_prefix_cache_hit_tokens, pf_queue_us, pf_compute_us,
                     pf_send_us);
};

// Per-recipient HostStagePool partition assignment carried inside
// PrefillSnapshot. Each PrefillSnapshot is unicast (one per Decode peer),
// so the assignment is filled in by Prefill against that specific peer's
// pre-allocated stage partition before sending. Decode caches the
// assignment keyed by Prefill peer addr and feeds it into PeerCredits
// (base_slot / n_slots) to allocate stage slots before RDMA WRITE; the
// pool_base_va + slot_size pair lets Decode compute the absolute
// target_offset = pool_base_va + start_slot * slot_size.
//
// Default-zero values mean "no assignment yet" — Decode treats this as
// the peer not being ready for stage writes and refuses admission to
// that peer until a non-zero assignment arrives. This keeps the wire
// schema-evolution safe: a Prefill that hasn't been upgraded yet will
// emit zeros and Decode will simply not route to that peer (failing
// fast over silently writing to slot 0 of an unregistered partition).
struct StageAssignment {
  // Mooncake-registered base VA of the Prefill HostStagePool. Decode
  // writes to target_offset = pool_base_va + start_slot * slot_size.
  uint64_t pool_base_va = 0;

  // Slot granularity of the pool (= HostStagePool::kSlotSize, currently
  // 64 KiB). Carried so the wire stays decoupled from the in-source
  // constant — a future pool size tweak doesn't require Decode rebuild.
  uint32_t slot_size = 0;

  // Absolute first slot index this Decode peer owns inside the pool.
  // Slots [base_slot, base_slot + n_slots) are reserved exclusively for
  // this Decode peer; PeerCredits ring on the Decode side allocates
  // within this range and emits absolute slot indices on the wire.
  uint32_t base_slot = 0;

  // Number of slots this peer owns. Capacity in bytes = n_slots *
  // slot_size. With the default 16 GB pool / 64 KiB slot / 128 peer
  // partition, n_slots = 2048 per peer.
  uint32_t n_slots = 0;

  MSGPACK_DEFINE_MAP(pool_base_va, slot_size, base_slot, n_slots);
};

// Periodic snapshot push from Prefill to subscribed Decode peers, used
// as a backstop against stale health views during idle periods (when no
// PrefillRequest / Complete is flowing). Sent every
// EngineConfig::snapshot_push_interval_ms.
//
// PrefillSnapshot doubles as the carrier for each Decode peer's stage
// partition assignment (StageAssignment). Each unicast snapshot is
// addressed to one specific Decode peer, so Prefill fills `stage` with
// that peer's partition before serialization. Combining health + stage
// assignment into one message avoids a separate handshake — Decode's
// admission gate (healthy && stage.n_slots > 0) is satisfied by exactly
// the messages it already subscribes to.
struct PrefillSnapshot {
  PeerHealth snapshot;
  StageAssignment stage;

  MSGPACK_DEFINE_MAP(snapshot, stage);
};

// Discriminator for the wire envelope. Receivers read the type from the
// frame header and dispatch to the matching deserializer.
enum class PdV2WireMsgType : uint32_t {
  kPrefillRequest = 1,
  kStageAck = 2,
  kPrefillRequestComplete = 3,
  kPrefillSnapshot = 4,
};

// Wire envelope: [u32 msg_type LE][u32 body_len LE][msgpack body...].
// Two 32-bit LE fields keep the header naturally aligned and avoid mixing
// u16/u32 widths in the codec. The msgpack body is self-describing
// (length is implicit in the encoding) but we keep an explicit body_len
// so a stream-mode receiver can know how many bytes to buffer before
// handing to the msgpack parser.
constexpr size_t kPdV2WireHeaderSize = 8;  // u32 + u32.

// Body codec: thin wrappers over msgpack. Every wire struct above has
// MSGPACK_DEFINE_MAP so a single template handles both directions.
template <typename T>
inline void SerializeMsg(const T& msg, std::vector<uint8_t>* out) {
  msgpack::sbuffer sbuf;
  msgpack::pack(sbuf, msg);
  const auto* p = reinterpret_cast<const uint8_t*>(sbuf.data());
  out->assign(p, p + sbuf.size());
}

template <typename T>
inline size_t DeserializeMsg(const uint8_t* in, size_t len, T* out) {
  if (in == nullptr || out == nullptr) return 0;
  try {
    std::size_t off = 0;
    msgpack::object_handle h = msgpack::unpack(reinterpret_cast<const char*>(in), len, off);
    h.get().convert(*out);
    return off;
  } catch (const std::exception&) {
    return 0;
  }
}

// Convenience aliases keeping the per-type call sites stable across the
// rest of pd_v2/. New wire types only need to add MSGPACK_DEFINE_MAP on
// the struct; no codec function need be written by hand.
inline void SerializePrefillRequest(const PrefillRequest& req, std::vector<uint8_t>* out) { SerializeMsg(req, out); }
inline size_t DeserializePrefillRequest(const uint8_t* in, size_t len, PrefillRequest* out) {
  return DeserializeMsg(in, len, out);
}

inline void SerializePrefillRequestComplete(const PrefillRequestComplete& msg, std::vector<uint8_t>* out) {
  SerializeMsg(msg, out);
}
inline size_t DeserializePrefillRequestComplete(const uint8_t* in, size_t len, PrefillRequestComplete* out) {
  return DeserializeMsg(in, len, out);
}

inline void SerializePrefillSnapshot(const PrefillSnapshot& msg, std::vector<uint8_t>* out) { SerializeMsg(msg, out); }
inline size_t DeserializePrefillSnapshot(const uint8_t* in, size_t len, PrefillSnapshot* out) {
  return DeserializeMsg(in, len, out);
}

inline void SerializePrefillStagePayload(const PrefillStagePayload& msg, std::vector<uint8_t>* out) {
  SerializeMsg(msg, out);
}
inline size_t DeserializePrefillStagePayload(const uint8_t* in, size_t len, PrefillStagePayload* out) {
  return DeserializeMsg(in, len, out);
}

inline void SerializeStageAck(const StageAck& msg, std::vector<uint8_t>* out) { SerializeMsg(msg, out); }
inline size_t DeserializeStageAck(const uint8_t* in, size_t len, StageAck* out) { return DeserializeMsg(in, len, out); }

// Frame helpers (out-of-line in .cpp).
void WrapFrame(PdV2WireMsgType type, const std::vector<uint8_t>& body, std::vector<uint8_t>* out);
bool PeekFrameHeader(const uint8_t* in, size_t len, PdV2WireMsgType* type, uint32_t* body_len);

}  // namespace pd_v2
}  // namespace ksana_llm
