/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
// MooncakeEngineConnector Prefill-role reception path: the incoming-request
// hook registration, OnPrefillReq (reads the RDMA-staged payload out of the
// HostStagePool, acks the slot range, and dispatches to the hook), and
// SendPrefillComplete (packs the reply back to the originating Decode).
#include <string>
#include <utility>
#include <vector>

#include "pd_v2/mooncake/mooncake_engine_connector.h"
#include "pd_v2/mooncake/mooncake_engine_connector_internal.h"
#include "utils/logger.h"

namespace ksana_llm {
namespace pd_v2 {
namespace mooncake_backend {

void MooncakeEngineConnector::RegisterIncomingHook(PdIncomingReqCb on_incoming, void* ctx) {
  std::lock_guard<std::mutex> lk(state_mu_);
  on_incoming_ = on_incoming;
  on_incoming_ctx_ = ctx;
}

Status MooncakeEngineConnector::SendPrefillComplete(const std::string& decode_engine_inference_addr, uint64_t req_id,
                                                    PdCompleteStatus status, const std::vector<int>& generated_tokens,
                                                    const PeerHealth& snapshot,
                                                    const PdPrefillMetrics& prefill_metrics) {
  {
    std::lock_guard<std::mutex> lk(state_mu_);
    incoming_routes_.erase({decode_engine_inference_addr, req_id});
  }
  PrefillRequestComplete msg;
  msg.req_id = req_id;
  msg.status = status;
  msg.snapshot = snapshot;
  msg.generated_tokens = generated_tokens;
  msg.pf_prefix_cache_hit_tokens = prefill_metrics.prefix_cache_hit_tokens;
  msg.pf_queue_us = prefill_metrics.queue_us;
  msg.pf_compute_us = prefill_metrics.compute_us;
  msg.pf_send_us = prefill_metrics.send_us;
  std::vector<uint8_t> body;
  SerializePrefillRequestComplete(msg, &body);
  Notification notify{kChPrefillComplete, std::string(body.begin(), body.end())};
  // PD 链路关键点[4/4]: Prefill 处理完请求, 回发 PrefillComplete 通知给 decode (保留 INFO 便于阻塞定位)。
  KLLM_LOG_INFO << "MooncakeEngineConnector::SendPrefillComplete req_id=" << req_id
                << " decode_addr=" << decode_engine_inference_addr << " status=" << static_cast<int>(status)
                << " tokens=" << generated_tokens.size();
  return tent_->SendNotification(decode_engine_inference_addr, notify);
}

void MooncakeEngineConnector::OnPrefillReq(const Notification& n) {
  PrefillRequest req;
  if (DeserializePrefillRequest(reinterpret_cast<const uint8_t*>(n.msg.data()), n.msg.size(), &req) == 0) {
    KLLM_LOG_WARNING << "OnPrefillReq: malformed wire frame, dropped (msg.size=" << n.msg.size() << ")";
    return;
  }

  // Stage payload sits at [stage_start_slot, stage_start_slot+stage_n_slots)
  // in our HostStagePool. Decode RDMA-wrote it before sending this notify.
  // Deserialize the PrefillStagePayload (msgpack of {input_tokens}) and
  // ack back so the Decode peer can return the slot range to its
  // PeerCredits ring. We send the ack even on payload-decode failure with
  // err_code != 0 — Prefill couldn't use the payload, but the slot range
  // is consumed and must be released, otherwise Decode's ring leaks.
  PrefillStagePayload stage_payload;
  uint32_t stage_err = 0;
  if (stage_pool_ == nullptr) {
    KLLM_LOG_WARNING << "OnPrefillReq: stage_pool_ is null (not Prefill role?); dropping req=" << req.req_id;
    return;
  }
  const size_t pool_total_slots = stage_pool_->total_slots();
  if (req.stage_n_slots == 0 || req.stage_start_slot >= pool_total_slots ||
      static_cast<size_t>(req.stage_start_slot) + req.stage_n_slots > pool_total_slots ||
      req.stage_payload_bytes == 0 || req.stage_payload_bytes > req.stage_n_slots * HostStagePool::kSlotSize) {
    KLLM_LOG_WARNING << "OnPrefillReq: bad stage descriptor req=" << req.req_id
                     << " start_slot=" << req.stage_start_slot << " n_slots=" << req.stage_n_slots
                     << " payload_bytes=" << req.stage_payload_bytes << " (pool_total_slots=" << pool_total_slots
                     << ")";
    stage_err = 2;  // bad descriptor
  } else {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(stage_pool_->SlotPtr(req.stage_start_slot));
    size_t consumed = DeserializePrefillStagePayload(p, req.stage_payload_bytes, &stage_payload);
    if (consumed == 0) {
      KLLM_LOG_WARNING << "OnPrefillReq: PrefillStagePayload decode failed req=" << req.req_id
                       << " payload_bytes=" << req.stage_payload_bytes;
      stage_err = 1;  // deserialization failed
    } else {
      for (auto& t : req.decode_targets) {
        t.kv_block_ids = stage_payload.kv_block_ids;
      }
    }
  }

  // Always send the StageAck so Decode can release the partition slots,
  // regardless of payload success. Sent before invoking the incoming hook
  // to minimize Decode-side credit hold time.
  if (tent_ != nullptr && !req.decode_engine_inference_addr.empty()) {
    StageAck ack;
    ack.req_id = req.req_id;
    ack.start_slot = req.stage_start_slot;
    ack.n_slots = req.stage_n_slots;
    ack.err_code = stage_err;
    std::vector<uint8_t> body;
    SerializeStageAck(ack, &body);
    Notification ackn{kChStageAck, std::string(body.begin(), body.end())};
    Status as = tent_->SendNotification(req.decode_engine_inference_addr, ackn);
    if (!as.OK()) {
      KLLM_LOG_WARNING << "OnPrefillReq: StageAck send to " << req.decode_engine_inference_addr
                       << " failed: " << as.GetMessage();
    }
  }

  if (stage_err != 0) {
    // No payload to feed Prefill; surface as a request-level failure via
    // PrefillRequestComplete{kFailed} so Decode's hook fails the request
    // cleanly instead of waiting on Prefill's hook that will never fire.
    if (tent_ != nullptr && !req.decode_engine_inference_addr.empty()) {
      PrefillRequestComplete complete;
      complete.req_id = req.req_id;
      complete.status = PdCompleteStatus::kFailed;
      std::vector<uint8_t> body;
      SerializePrefillRequestComplete(complete, &body);
      Notification cn{kChPrefillComplete, std::string(body.begin(), body.end())};
      (void)tent_->SendNotification(req.decode_engine_inference_addr, cn);
    }
    return;
  }

  PdIncomingReqCb cb = nullptr;
  void* ctx = nullptr;
  {
    std::lock_guard<std::mutex> lk(state_mu_);
    incoming_routes_[{req.decode_engine_inference_addr, req.req_id}] = true;
    cb = on_incoming_;
    ctx = on_incoming_ctx_;
  }
  if (cb == nullptr) {
    KLLM_LOG_WARNING << "OnPrefillReq: no incoming hook registered, dropped";
    return;
  }
  IncomingPrefillRequest in;
  in.decode_engine_inference_addr = req.decode_engine_inference_addr;
  in.req_id = req.req_id;
  in.input_tokens = std::move(stage_payload.input_tokens);
  in.sampling_config_wire = static_cast<const void*>(&req.sampling_config);
  in.decode_targets_wire = req.decode_targets.empty() ? nullptr : static_cast<const void*>(req.decode_targets.data());
  in.num_decode_targets = static_cast<uint32_t>(req.decode_targets.size());
  in.decode_attn_dp_group_id = req.decode_attn_dp_group_id;
  // PD 链路关键点[3/4]: Prefill 收到 decode 发来的请求并派发给 hook (保留 INFO 便于阻塞定位)。
  KLLM_LOG_INFO << "MooncakeEngineConnector::OnPrefillReq dispatching to hook: "
                << "req_id=" << req.req_id << " decode_addr=" << req.decode_engine_inference_addr
                << " input_tokens=" << in.input_tokens.size() << " decode_targets=" << req.decode_targets.size()
                << " decode_attn_dp_group_id=" << req.decode_attn_dp_group_id;
  cb(in, ctx);
}

}  // namespace mooncake_backend
}  // namespace pd_v2
}  // namespace ksana_llm
