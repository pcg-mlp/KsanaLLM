/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "runtime/speculative/ptp/ptp_decoder_adapter.h"

#include "configure/schedule_config_parser.h"
#include "runtime/executor_infer_request.h"
#include "runtime/speculative/common/speculative_resource_profile.h"

namespace ksana_llm {

PtpDecoderAdapter::PtpDecoderAdapter(std::shared_ptr<PtpGenerator> generator) : generator_(std::move(generator)) {}

SpecSchedulingInfo PtpDecoderAdapter::GetSchedulingInfo() const { return generator_->GetSchedulingInfo(); }

void PtpDecoderAdapter::BeforeForward(SpecContext& ctx) {
  if (!ctx.reqs) return;
  for (auto& req : *ctx.reqs) {
    generator_->GenerateDraft(req);
  }
}

void PtpDecoderAdapter::Verify(SpecContext& ctx) {
  if (!ctx.reqs) return;
  for (auto& req : *ctx.reqs) {
    generator_->FilterDraftTokens(req);
  }
}

bool PtpDecoderAdapter::IsCombinableWith(const SpeculativeDecoderInterface* other) const {
  // PTP is incompatible with MTP and speculative decoding (Trie/ASR-custom).
  if (!other) return true;
  const char* other_name = other->GetName();
  return (strcmp(other_name, "MTP") != 0 && strcmp(other_name, "Trie") != 0 && strcmp(other_name, "ASR") != 0);
}

// Reverse static registration for PTP resource estimation.
namespace {

SpeculativeResourceProfile PtpEstimateProfile(const BatchSchedulerConfig& cfg) {
  SpeculativeResourceProfile profile;
  if (cfg.ptp_step_num > 0) {
    // PTP: ptp_step_num verify tokens + ptp_step_num placeholder tokens.
    // PTP does not need extra KV tokens — draft tokens are verify+placeholder only.
    profile.extra_decode_tokens_per_req = 2 * cfg.ptp_step_num;
    profile.workload_multiplier = 2 * cfg.ptp_step_num;
    profile.extra_token_capacity = 2 * cfg.ptp_step_num;
    profile.extra_kv_tokens = 0;
  }
  return profile;
}

const bool kPtpResourceRegistered = []() {
  SpeculativeRegistry::Get().Register(PtpEstimateProfile);
  return true;
}();

const bool kPtpDecoderRegistered = []() {
  SpeculativeRegistry::Get().RegisterDecoder("ptp", [](const BatchSchedulerConfig& cfg) {
    DecoderFactoryResult result;
    if (cfg.ptp_step_num > 0) {
      auto gen = std::make_shared<PtpGenerator>(cfg.ptp_step_num, cfg.ptp_token_id);
      result.generator = gen;
      result.adapter = std::make_unique<PtpDecoderAdapter>(gen);
    }
    return result;
  });
  return true;
}();

}  // namespace

}  // namespace ksana_llm