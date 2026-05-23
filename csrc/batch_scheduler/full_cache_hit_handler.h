/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "configure/environment.h"
#include "runtime/infer_request.h"
#include "utils/logger.h"
#include "utils/string_utils.h"

namespace ksana_llm {

// When prefix cache fully covers the input sequence, retain the last N token(s) for prefill
// to compute logits for next-token generation.
// For SWA models, prefill_start_offset is floor-aligned to block_token_num so that
// prefix_len is a multiple of block_token_num.
// For non-SWA models, prefill_token_num is kStepGenerateTokenNum (1),
// and switches the request to decode stage.
// Caller is responsible for checking the full-hit condition (shared_token_num >= seq_token_num) before calling.
inline void HandleFullPrefixCacheHit(const std::shared_ptr<InferRequest>& req, ScheduleTaskWorkload& planning_workload,
                                     size_t seq_token_num) {
  const bool is_swa = !req->swa_kv_cache_blocks.empty();
  size_t prefill_start_offset;
  size_t prefill_token_num;
  if (is_swa) {
    prefill_start_offset = (seq_token_num - kStepGenerateTokenNum) / req->block_token_num * req->block_token_num;
    prefill_token_num = seq_token_num - prefill_start_offset;
  } else {
    prefill_token_num = kStepGenerateTokenNum;
    prefill_start_offset = seq_token_num - prefill_token_num;
  }
  KLLM_CHECK_WITH_INFO(seq_token_num >= prefill_token_num,
                       FormatStr("req_id=%ld seq_token_num(%zu) < prefill_token_num(%zu)", req->req_id, seq_token_num,
                                 prefill_token_num));
  planning_workload.prefill_start_offset = prefill_start_offset;
  planning_workload.prefill_token_num = prefill_token_num;
  req->SetPlanningWorkload(planning_workload);
  req->SetRemainingWorkload(planning_workload);
  if (!is_swa) {
    req->infer_stage = InferStage::kDecode;
  }
  KLLM_LOG_SCHEDULER << "HandleFullPrefixCacheHit: req_id=" << req->req_id << ", seq_token_num=" << seq_token_num
                     << ", is_swa=" << is_swa << ", prefill_token_num=" << prefill_token_num
                     << ", prefill_start_offset=" << planning_workload.prefill_start_offset;
}

}  // namespace ksana_llm
