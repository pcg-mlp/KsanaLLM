/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "pd_v2/wire/prefill_request_wire.h"

#include <gtest/gtest.h>

namespace ksana_llm {
namespace pd_v2 {

namespace {

PeerHealth MakeSnapshot() {
  PeerHealth h;
  h.segment_id = 0xCAFEBABEDEADBEEFULL;
  h.server_name = "prefill-node-7";
  h.free_kv_blocks = 4096;
  h.last_snapshot_ms = 1234567890123ULL;
  h.healthy = true;
  // Load features (rolling-upgrade-safe MAP fields).
  h.future_free_kv_blocks = 512;
  h.waiting_reqs = 7;
  h.running_reqs = 13;
  h.pending_prefill_tokens = 987654ULL;
  h.step_token_num = 2048;
  h.step_latency_us = 31000;
  return h;
}

void ExpectPeerHealthEq(const PeerHealth& a, const PeerHealth& b) {
  EXPECT_EQ(a.segment_id, b.segment_id);
  EXPECT_EQ(a.server_name, b.server_name);
  EXPECT_EQ(a.free_kv_blocks, b.free_kv_blocks);
  EXPECT_EQ(a.last_snapshot_ms, b.last_snapshot_ms);
  EXPECT_EQ(a.healthy, b.healthy);
  EXPECT_EQ(a.future_free_kv_blocks, b.future_free_kv_blocks);
  EXPECT_EQ(a.waiting_reqs, b.waiting_reqs);
  EXPECT_EQ(a.running_reqs, b.running_reqs);
  EXPECT_EQ(a.pending_prefill_tokens, b.pending_prefill_tokens);
  EXPECT_EQ(a.step_token_num, b.step_token_num);
  EXPECT_EQ(a.step_latency_us, b.step_latency_us);
}

SamplingConfigWire MakeSamplingConfig() {
  SamplingConfigWire cfg;
  cfg.topk = 32;
  cfg.topp = 0.95f;
  cfg.temperature = 0.8f;
  cfg.repetition_penalty = 1.1f;
  cfg.max_new_tokens = 4;  // 1 main + 3 drafts
  cfg.stop_token_ids = {2, 0};
  cfg.stop_strings = {"</s>", "END"};
  cfg.ignore_eos = false;
  cfg.json_schema = R"({"type":"object","properties":{"x":{"type":"integer"}}})";
  cfg.enable_structured_output = true;
  cfg.enable_thinking = false;
  cfg.ptp_token_num = 3;
  return cfg;
}

void ExpectSamplingConfigEq(const SamplingConfigWire& a, const SamplingConfigWire& b) {
  EXPECT_EQ(a.topk, b.topk);
  EXPECT_FLOAT_EQ(a.topp, b.topp);
  EXPECT_FLOAT_EQ(a.temperature, b.temperature);
  EXPECT_FLOAT_EQ(a.repetition_penalty, b.repetition_penalty);
  EXPECT_EQ(a.max_new_tokens, b.max_new_tokens);
  EXPECT_EQ(a.stop_token_ids, b.stop_token_ids);
  EXPECT_EQ(a.stop_strings, b.stop_strings);
  EXPECT_EQ(a.ignore_eos, b.ignore_eos);
  EXPECT_EQ(a.json_schema, b.json_schema);
  EXPECT_EQ(a.enable_structured_output, b.enable_structured_output);
  EXPECT_EQ(a.enable_thinking, b.enable_thinking);
  EXPECT_EQ(a.ptp_token_num, b.ptp_token_num);
}

}  // namespace

TEST(PrefillRequestWireTest, RoundTripPrefillRequestMultiTarget) {
  PrefillRequest req;
  req.decode_engine_inference_addr = "ksana-prod/decode-host-7/engine";
  req.req_id = 42;
  req.stage_start_slot = 1024;
  req.stage_n_slots = 3;
  req.stage_payload_bytes = 1024 * 130;  // ~32K tokens, 3 64KB slots
  req.sampling_config = MakeSamplingConfig();
  // Decode attn_DP=4, this request lands on group 2.
  req.decode_attn_dp_group_id = 2;

  DecodeTarget t0;
  t0.decode_attn_tp_rank_in_group = 0;
  t0.kv_pool_base_va = 100;
  // kv_block_ids not serialized on wire — carried in PrefillStagePayload
  req.decode_targets.push_back(t0);

  DecodeTarget t1;
  t1.decode_attn_tp_rank_in_group = 1;
  t1.kv_pool_base_va = 101;
  req.decode_targets.push_back(t1);

  std::vector<uint8_t> buf;
  SerializePrefillRequest(req, &buf);
  ASSERT_FALSE(buf.empty());

  PrefillRequest out;
  size_t consumed = DeserializePrefillRequest(buf.data(), buf.size(), &out);
  EXPECT_EQ(consumed, buf.size());

  EXPECT_EQ(out.decode_engine_inference_addr, req.decode_engine_inference_addr);
  EXPECT_EQ(out.req_id, req.req_id);
  EXPECT_EQ(out.stage_start_slot, req.stage_start_slot);
  EXPECT_EQ(out.stage_n_slots, req.stage_n_slots);
  EXPECT_EQ(out.stage_payload_bytes, req.stage_payload_bytes);
  ExpectSamplingConfigEq(out.sampling_config, req.sampling_config);
  EXPECT_EQ(out.decode_attn_dp_group_id, 2u);
  ASSERT_EQ(out.decode_targets.size(), 2u);
  EXPECT_EQ(out.decode_targets[0].decode_attn_tp_rank_in_group, 0u);
  EXPECT_EQ(out.decode_targets[0].kv_pool_base_va, 100u);
  EXPECT_TRUE(out.decode_targets[0].kv_block_ids.empty());
  EXPECT_EQ(out.decode_targets[1].decode_attn_tp_rank_in_group, 1u);
  EXPECT_TRUE(out.decode_targets[1].kv_block_ids.empty());
}

TEST(PrefillRequestWireTest, RoundTripPrefillStagePayloadLargeTokens) {
  // Mimic a 32K-token prefill — the size that originally drove the
  // control-stage redesign. Body should land well above the 64 KB
  // notify limit (proving the stage path is needed) but still be
  // round-trip clean through msgpack.
  PrefillStagePayload payload;
  payload.input_tokens.reserve(32 * 1024);
  for (int i = 0; i < 32 * 1024; ++i) payload.input_tokens.push_back(i & 0xFFFF);
  payload.kv_block_ids.reserve(1225);
  for (int i = 0; i < 1225; ++i) payload.kv_block_ids.push_back(i);

  std::vector<uint8_t> buf;
  SerializePrefillStagePayload(payload, &buf);
  EXPECT_GT(buf.size(), 64u * 1024u) << "stage payload should be > 64KB";

  PrefillStagePayload out;
  EXPECT_GT(DeserializePrefillStagePayload(buf.data(), buf.size(), &out), 0u);
  EXPECT_EQ(out.input_tokens, payload.input_tokens);
  EXPECT_EQ(out.kv_block_ids, payload.kv_block_ids);
}

TEST(PrefillRequestWireTest, RoundTripStageAckOk) {
  StageAck ack;
  ack.req_id = 4242;
  ack.start_slot = 1024;
  ack.n_slots = 3;
  ack.err_code = 0;

  std::vector<uint8_t> buf;
  SerializeStageAck(ack, &buf);
  // Header + ack body must fit comfortably under the 64 KB notify limit.
  EXPECT_LT(buf.size(), 256u);

  StageAck out;
  EXPECT_GT(DeserializeStageAck(buf.data(), buf.size(), &out), 0u);
  EXPECT_EQ(out.req_id, ack.req_id);
  EXPECT_EQ(out.start_slot, ack.start_slot);
  EXPECT_EQ(out.n_slots, ack.n_slots);
  EXPECT_EQ(out.err_code, 0u);
}

TEST(PrefillRequestWireTest, RoundTripStageAckWithError) {
  StageAck ack;
  ack.req_id = 99;
  ack.start_slot = 0;
  ack.n_slots = 2;
  ack.err_code = 7;  // arbitrary non-zero err code

  std::vector<uint8_t> buf;
  SerializeStageAck(ack, &buf);
  StageAck out;
  EXPECT_GT(DeserializeStageAck(buf.data(), buf.size(), &out), 0u);
  EXPECT_EQ(out.err_code, 7u);
  EXPECT_EQ(out.n_slots, ack.n_slots);
}

TEST(PrefillRequestWireTest, RoundTripCompleteWithTokens) {
  PrefillRequestComplete msg;
  msg.req_id = 1234;
  msg.status = PdCompleteStatus::kOk;
  msg.snapshot = MakeSnapshot();
  msg.generated_tokens = {2024, 7, 8, 9};  // 1 main + 3 drafts

  std::vector<uint8_t> buf;
  SerializePrefillRequestComplete(msg, &buf);

  PrefillRequestComplete out;
  EXPECT_GT(DeserializePrefillRequestComplete(buf.data(), buf.size(), &out), 0u);
  EXPECT_EQ(out.req_id, msg.req_id);
  EXPECT_EQ(out.status, PdCompleteStatus::kOk);
  ExpectPeerHealthEq(out.snapshot, msg.snapshot);
  EXPECT_EQ(out.generated_tokens, msg.generated_tokens);
}

TEST(PrefillRequestWireTest, RoundTripCompleteFailedHasEmptyTokens) {
  PrefillRequestComplete msg;
  msg.req_id = 5678;
  msg.status = PdCompleteStatus::kFailed;
  msg.snapshot = MakeSnapshot();
  // generated_tokens left empty — Decode hook should not consume on failure.

  std::vector<uint8_t> buf;
  SerializePrefillRequestComplete(msg, &buf);

  PrefillRequestComplete out;
  EXPECT_GT(DeserializePrefillRequestComplete(buf.data(), buf.size(), &out), 0u);
  EXPECT_EQ(out.status, PdCompleteStatus::kFailed);
  EXPECT_TRUE(out.generated_tokens.empty());
}

TEST(PrefillRequestWireTest, RoundTripSnapshot) {
  PrefillSnapshot snap;
  snap.snapshot = MakeSnapshot();
  // Per-recipient stage partition assignment: 16 GB pool / 64 KiB slot,
  // peer 5 of 128 -> base_slot = 5 * 2048, n_slots = 2048.
  snap.stage.pool_base_va = 0x7FAB000000000000ULL;
  snap.stage.slot_size = 64u * 1024u;
  snap.stage.base_slot = 5u * 2048u;
  snap.stage.n_slots = 2048u;
  std::vector<uint8_t> sb;
  SerializePrefillSnapshot(snap, &sb);
  PrefillSnapshot snap_out;
  EXPECT_GT(DeserializePrefillSnapshot(sb.data(), sb.size(), &snap_out), 0u);
  ExpectPeerHealthEq(snap_out.snapshot, snap.snapshot);
  EXPECT_EQ(snap_out.stage.pool_base_va, snap.stage.pool_base_va);
  EXPECT_EQ(snap_out.stage.slot_size, snap.stage.slot_size);
  EXPECT_EQ(snap_out.stage.base_slot, snap.stage.base_slot);
  EXPECT_EQ(snap_out.stage.n_slots, snap.stage.n_slots);
}

TEST(PrefillRequestWireTest, FrameWrapAndPeek) {
  PrefillRequest req;
  req.decode_engine_inference_addr = "test";
  req.req_id = 2;
  req.stage_start_slot = 0;
  req.stage_n_slots = 1;
  req.stage_payload_bytes = 64;
  req.sampling_config.max_new_tokens = 1;

  std::vector<uint8_t> body;
  SerializePrefillRequest(req, &body);

  std::vector<uint8_t> framed;
  WrapFrame(PdV2WireMsgType::kPrefillRequest, body, &framed);
  EXPECT_EQ(framed.size(), kPdV2WireHeaderSize + body.size());

  PdV2WireMsgType type;
  uint32_t body_len = 0;
  ASSERT_TRUE(PeekFrameHeader(framed.data(), framed.size(), &type, &body_len));
  EXPECT_EQ(type, PdV2WireMsgType::kPrefillRequest);
  EXPECT_EQ(body_len, body.size());

  PrefillRequest out;
  EXPECT_GT(DeserializePrefillRequest(framed.data() + kPdV2WireHeaderSize, framed.size() - kPdV2WireHeaderSize, &out),
            0u);
  EXPECT_EQ(out.req_id, 2u);
}

TEST(PrefillRequestWireTest, RejectsTruncatedFrame) {
  PdV2WireMsgType type;
  uint32_t body_len = 0;
  uint8_t buf[3] = {0, 0, 0};
  EXPECT_FALSE(PeekFrameHeader(buf, sizeof(buf), &type, &body_len));
}

TEST(PrefillRequestWireTest, RejectsTruncatedPrefillRequest) {
  PrefillRequest req;
  req.decode_engine_inference_addr = "test";
  req.req_id = 2;
  req.stage_start_slot = 0;
  req.stage_n_slots = 2;
  req.stage_payload_bytes = 96;
  req.sampling_config.max_new_tokens = 1;

  std::vector<uint8_t> buf;
  SerializePrefillRequest(req, &buf);

  PrefillRequest out;
  // Lop off the last byte; msgpack should refuse to decode cleanly.
  EXPECT_EQ(DeserializePrefillRequest(buf.data(), buf.size() - 1, &out), 0u);
}

// Schema-evolution smoke test: an older receiver decoding a newer message
// with extra unknown fields, or a newer receiver decoding an older message
// with missing fields, should both work. msgpack MAP decoding ignores
// unknown keys and leaves missing-key fields at their in-source default.
TEST(PrefillRequestWireTest, SchemaEvolutionMissingFieldsDefault) {
  // Pack a PrefillRequestComplete-shaped MAP with ONLY req_id present
  // (generated_tokens omitted) plus an unknown extra field.
  msgpack::sbuffer sbuf;
  msgpack::packer<msgpack::sbuffer> pk(&sbuf);
  pk.pack_map(2);
  pk.pack(std::string("req_id"));
  pk.pack(static_cast<uint64_t>(7));
  pk.pack(std::string("unknown_future"));
  pk.pack(static_cast<uint64_t>(99));

  PrefillRequestComplete out;
  size_t consumed =
      DeserializeMsg<PrefillRequestComplete>(reinterpret_cast<const uint8_t*>(sbuf.data()), sbuf.size(), &out);
  EXPECT_GT(consumed, 0u);
  EXPECT_EQ(out.req_id, 7u);
}

// Pre-stage Prefill (only `snapshot` field, no `stage` key) still
// deserializes cleanly on the upgraded Decode side, with stage defaulting
// to zero — Decode treats that as "no assignment yet" and refuses to
// route to that peer until a non-zero StageAssignment lands.
TEST(PrefillRequestWireTest, SchemaEvolutionSnapshotMissingStageDefaultsZero) {
  msgpack::sbuffer sbuf;
  msgpack::packer<msgpack::sbuffer> pk(&sbuf);
  pk.pack_map(1);  // only `snapshot`, no `stage`
  pk.pack(std::string("snapshot"));
  pk.pack(MakeSnapshot());

  PrefillSnapshot out;
  // Pre-fill with non-zero values to ensure the decoder leaves stage
  // either at default (msgpack behavior on missing key) or at the
  // pre-decode value — both are acceptable, just don't crash.
  out.stage.pool_base_va = 0xDEADBEEFULL;
  out.stage.n_slots = 9999;
  size_t consumed = DeserializeMsg<PrefillSnapshot>(reinterpret_cast<const uint8_t*>(sbuf.data()), sbuf.size(), &out);
  EXPECT_GT(consumed, 0u);
  ExpectPeerHealthEq(out.snapshot, MakeSnapshot());
}

// Rolling upgrade: an old Prefill packs only the original 5 PeerHealth
// fields (no load features). The upgraded reader decodes into a freshly
// default-constructed PeerHealth, so every new MAP field that the wire
// omits keeps its in-class default of 0. This is the exact pattern the
// production decode paths use (a fresh PeerHealth per message), which is why
// it is rolling-upgrade safe. Note: msgpack MAP decode leaves missing-key
// fields *untouched* rather than actively zeroing them — the zero comes from
// the in-class initializer, not the decoder, so callers must not reuse a
// dirty target object across decodes.
TEST(PrefillRequestWireTest, PeerHealthSchemaEvolutionLoadFieldsDefaultZero) {
  msgpack::sbuffer sbuf;
  msgpack::packer<msgpack::sbuffer> pk(&sbuf);
  pk.pack_map(5);  // only the legacy fields
  pk.pack(std::string("segment_id"));
  pk.pack(static_cast<uint64_t>(42));
  pk.pack(std::string("server_name"));
  pk.pack(std::string("legacy-prefill"));
  pk.pack(std::string("free_kv_blocks"));
  pk.pack(static_cast<uint32_t>(100));
  pk.pack(std::string("last_snapshot_ms"));
  pk.pack(static_cast<uint64_t>(555));
  pk.pack(std::string("healthy"));
  pk.pack(true);

  PeerHealth out;  // fresh, in-class defaults are all 0
  size_t consumed = DeserializeMsg<PeerHealth>(reinterpret_cast<const uint8_t*>(sbuf.data()), sbuf.size(), &out);
  EXPECT_GT(consumed, 0u);
  EXPECT_EQ(out.segment_id, 42u);
  EXPECT_EQ(out.server_name, "legacy-prefill");
  EXPECT_EQ(out.free_kv_blocks, 100u);
  EXPECT_EQ(out.healthy, true);
  EXPECT_EQ(out.future_free_kv_blocks, 0u);
  EXPECT_EQ(out.waiting_reqs, 0u);
  EXPECT_EQ(out.running_reqs, 0u);
  EXPECT_EQ(out.pending_prefill_tokens, 0u);
  EXPECT_EQ(out.step_token_num, 0u);
  EXPECT_EQ(out.step_latency_us, 0u);
}

TEST(PrefillRequestWireTest, NotifyBodyFitsUnder64KBWith8Targets) {
  PrefillRequest req;
  req.decode_engine_inference_addr = "10.0.0.1:8089";
  req.req_id = 31931136;
  req.stage_start_slot = 0;
  req.stage_n_slots = 6;
  req.stage_payload_bytes = 360000;
  req.sampling_config = MakeSamplingConfig();
  req.decode_attn_dp_group_id = 0;
  for (uint32_t r = 0; r < 8; ++r) {
    DecodeTarget t;
    t.decode_attn_tp_rank_in_group = r;
    t.kv_pool_base_va = 0x7ef7ae000000ULL + r * 0x100000000ULL;
    t.cached_block_count = 0;
    t.decode_prefix_cache_offset = 0;
    req.decode_targets.push_back(t);
  }
  std::vector<uint8_t> buf;
  SerializePrefillRequest(req, &buf);
  // After base64 the body must stay under 64 KB.
  const size_t base64_size = (buf.size() + 2) / 3 * 4;
  EXPECT_LT(base64_size, 64u * 1024u) << "notify body after base64 = " << base64_size << " bytes, must be < 64 KB";
}

// Pre-stage PrefillStagePayload (only `input_tokens`, no `kv_block_ids` key)
// still deserializes cleanly on the upgraded receiver, with kv_block_ids
// defaulting to empty — rolling-upgrade safety net.
TEST(PrefillRequestWireTest, SchemaEvolutionStagePayloadMissingKvBlockIds) {
  msgpack::sbuffer sbuf;
  msgpack::packer<msgpack::sbuffer> pk(&sbuf);
  pk.pack_map(1);
  pk.pack(std::string("input_tokens"));
  pk.pack(std::vector<int>{1, 2, 3});

  PrefillStagePayload out;
  size_t consumed =
      DeserializeMsg<PrefillStagePayload>(reinterpret_cast<const uint8_t*>(sbuf.data()), sbuf.size(), &out);
  EXPECT_GT(consumed, 0u);
  EXPECT_EQ(out.input_tokens, (std::vector<int>{1, 2, 3}));
  EXPECT_TRUE(out.kv_block_ids.empty());
}

}  // namespace pd_v2
}  // namespace ksana_llm
