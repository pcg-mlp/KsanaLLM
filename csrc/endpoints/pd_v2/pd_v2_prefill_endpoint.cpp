/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "endpoints/pd_v2/pd_v2_prefill_endpoint.h"

#include "endpoints/local/local_endpoint.h"
#include "pd_v2/wire/prefill_request_wire.h"
#include "pd_v2/wire/sampling_config_wire.h"
#include "utils/logger.h"
#include "utils/request.h"
#include "utils/waiter.h"

namespace ksana_llm {
namespace pd_v2 {

namespace {

SamplingConfig WireToSamplingConfig(const SamplingConfigWire& w) {
  SamplingConfig sc;
  sc.topk = w.topk;
  sc.topp = w.topp;
  sc.temperature = w.temperature;
  sc.repetition_penalty = w.repetition_penalty;
  sc.max_new_tokens = w.max_new_tokens;
  sc.stop_token_ids.assign(w.stop_token_ids.begin(), w.stop_token_ids.end());
  sc.stop_strings = w.stop_strings;
  sc.ignore_eos = w.ignore_eos;
  sc.json_schema = w.json_schema;
  sc.regex = w.regex;
  sc.grammar = w.grammar;
  sc.structural_tag = w.structural_tag;
  sc.enable_structured_output = w.enable_structured_output;
  sc.enable_thinking = w.enable_thinking;
  sc.ptp_token_num = w.ptp_token_num;
  return sc;
}

}  // namespace

PdV2PrefillEndpoint::PdV2PrefillEndpoint(Deps deps) : deps_(std::move(deps)) {
  PdV2PrefillCompletion::Deps completion_deps;
  completion_deps.connector = deps_.connector;
  completion_deps.snapshot_fn = deps_.snapshot_fn;
  completion_ = std::make_shared<PdV2PrefillCompletion>(std::move(completion_deps));

  if (deps_.local_endpoint == nullptr) {
    KLLM_LOG_ERROR << "PdV2PrefillEndpoint: local_endpoint is null";
  }
  if (deps_.connector == nullptr) {
    KLLM_LOG_ERROR << "PdV2PrefillEndpoint: connector is null";
  }
}

PdV2PrefillEndpoint::~PdV2PrefillEndpoint() = default;

void PdV2PrefillEndpoint::Register() {
  if (deps_.connector == nullptr) return;
  deps_.connector->RegisterIncomingHook(&PdV2PrefillEndpoint::OnIncomingTrampoline, this);
}

void PdV2PrefillEndpoint::OnIncomingTrampoline(const IncomingPrefillRequest& req, void* ctx) {
  auto* self = static_cast<PdV2PrefillEndpoint*>(ctx);
  if (self == nullptr) return;
  self->OnIncoming(req);
}

void PdV2PrefillEndpoint::OnIncoming(const IncomingPrefillRequest& req) {
  if (deps_.local_endpoint == nullptr || deps_.connector == nullptr) {
    KLLM_LOG_ERROR << "PdV2PrefillEndpoint::OnIncoming: missing deps, drop req_id=" << req.req_id;
    return;
  }

  if (deps_.prefill_attn_tp > 0 && req.num_decode_targets > deps_.prefill_attn_tp) {
    KLLM_LOG_ERROR << "PdV2PrefillEndpoint::OnIncoming: unsupported topology"
                   << " req_id=" << req.req_id << " d_attn_tp=" << req.num_decode_targets
                   << " > prefill_attn_tp=" << deps_.prefill_attn_tp
                   << " (matrix §6 'p_attn_TP < d_attn_TP' TBD slice/replicate"
                   << " row); rejecting with kFailed";
    PeerHealth health;
    if (deps_.snapshot_fn) {
      health = deps_.snapshot_fn();
    } else {
      health.healthy = true;
    }
    deps_.connector->SendPrefillComplete(req.decode_engine_inference_addr, req.req_id, PdCompleteStatus::kFailed,
                                         /*generated_tokens=*/{}, health, /*prefill_metrics=*/{});
    return;
  }

  if (req.sampling_config_wire == nullptr) {
    KLLM_LOG_ERROR << "PdV2PrefillEndpoint::OnIncoming: null sampling_config_wire, drop req_id=" << req.req_id;
    return;
  }
  const auto* wire_sc = static_cast<const SamplingConfigWire*>(req.sampling_config_wire);

  auto request =
      std::make_shared<Request>(static_cast<int64_t>(req.req_id), req.input_tokens, WireToSamplingConfig(*wire_sc));

  const std::string peer_addr = req.decode_engine_inference_addr;

  // 确认tokens已经全部返回后，调用回调函数
  request->waiter = std::make_shared<Waiter>(/*num_wait=*/1, completion_->MakeRequestFinishedCallback(req.req_id));

  std::vector<DecodeTargetLite> targets_lite;
  if (req.decode_targets_wire != nullptr && req.num_decode_targets > 0) {
    const auto* targets_wire = static_cast<const DecodeTarget*>(req.decode_targets_wire);
    targets_lite.reserve(req.num_decode_targets);
    for (uint32_t i = 0; i < req.num_decode_targets; ++i) {
      targets_lite.push_back(targets_wire[i].ToLite());
    }
  }

  request->is_pd_v2_prefill = true;
  request->pd_v2_decode_targets = std::move(targets_lite);
  request->pd_v2_decode_peer_addr = peer_addr;
  request->pd_v2_decode_attn_dp_group_id = req.decode_attn_dp_group_id;

  if (deps_.connector != nullptr) {
    auto* connector_raw = deps_.connector.get();
    const uint64_t req_id_capture = req.req_id;
    request->pd_v2_on_destroy = [connector_raw, req_id_capture]() { connector_raw->Release(req_id_capture); };
  }

  // 在alive map 中绑定 Request 和 调用回调函数，回调函数会先查询请求是否活着然后
  // 调用回调
  completion_->Admit(req.req_id, request, peer_addr, req.num_decode_targets);

  Status submit_status = deps_.local_endpoint->SubmitAsync(request);
  if (!submit_status.OK()) {
    KLLM_LOG_ERROR << "PdV2PrefillEndpoint::OnIncoming SubmitAsync failed req_id=" << req.req_id
                   << " err=" << submit_status.GetMessage();
    request->finish_status = submit_status;
    if (request->waiter) {
      request->waiter->Notify();
    }
    return;
  }

  KLLM_LOG_DEBUG << "PdV2PrefillEndpoint admitted req_id=" << req.req_id << " tokens=" << req.input_tokens.size()
                 << " max_new_tokens=" << wire_sc->max_new_tokens;
}

void PdV2PrefillEndpoint::OnBarrierDone(int64_t req_id, uint8_t status) { completion_->OnBarrierDone(req_id, status); }

size_t PdV2PrefillEndpoint::NumIncomingAdmittedForTest() const { return completion_->NumAdmittedForTest(); }

size_t PdV2PrefillEndpoint::NumCompletedForTest() const { return completion_->NumCompletedForTest(); }

size_t PdV2PrefillEndpoint::NumAliveForTest() const { return completion_->NumAliveForTest(); }

}  // namespace pd_v2
}  // namespace ksana_llm
