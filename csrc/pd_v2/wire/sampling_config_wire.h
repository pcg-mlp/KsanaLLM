/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "msgpack.hpp"

namespace ksana_llm {
namespace pd_v2 {

// Wire-side mirror of the ksana_llm SamplingConfig fields that Prefill needs
// to reproduce the user's sampling intent locally. Decode hook fills this
// from req->sampling_config before SubmitPrefillRequest and overrides
// max_new_tokens to 1 + ptp_token_num so Prefill returns exactly the
// tokens Decode needs to bootstrap its loop.
//
// Fields are a subset of ksana_llm::SamplingConfig — kept narrow because
// (a) the wire payload is sent per request and (b) some fields (beam search,
// logprobs, length_penalty) are not meaningful for the 1 + M-draft Prefill
// micro-batch. R7 will revisit if a regression test exposes a gap.
//
// Wire format: msgpack MAP (field names as keys), so adding a field is
// backward-compatible across a rolling Prefill/Decode upgrade. See
// csrc/utils/request_serial.h for the same MSGPACK_DEFINE_MAP convention
// already used by trpc_endpoint / request_packer.
struct SamplingConfigWire {
  int32_t topk = 1;
  float topp = 1.0f;
  float temperature = 1.0f;
  float repetition_penalty = 1.0f;

  // Number of tokens Prefill must produce. Decode hook sets this to
  // 1 (main) + max(0, ptp_token_num) (drafts).
  int32_t max_new_tokens = 1;

  std::vector<int32_t> stop_token_ids;
  std::vector<std::string> stop_strings;
  bool ignore_eos = false;

  std::string json_schema;
  bool enable_structured_output = false;
  bool enable_thinking = true;

  // Per-request PTP draft token number. -1 = unset (use global default on
  // Prefill side), 0 = disable PTP, >0 = use the specified number.
  int32_t ptp_token_num = -1;

  MSGPACK_DEFINE_MAP(topk, topp, temperature, repetition_penalty, max_new_tokens, stop_token_ids, stop_strings,
                     ignore_eos, json_schema, enable_structured_output, enable_thinking, ptp_token_num);
};

}  // namespace pd_v2
}  // namespace ksana_llm
