/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/runtime/infer_request.h"

#include <atomic>
#include <limits>
#include <vector>
#include "ksana_llm/block_manager/block_manager.h"
#include "ksana_llm/runtime/infer_stage.h"
#include "ksana_llm/utils/memory_utils.h"
#include "ksana_llm/utils/request.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/string_utils.h"
#include "ksana_llm/utils/tensor.h"

namespace ksana_llm {
InferRequest::InferRequest(std::shared_ptr<Request> &request, int index)
    : req_id(request->req_ids[index]),
      model_name(request->model_name),
      logits_custom_length(request->logits_custom_length),
      input_tokens(request->input_tokens),
      input_refit_embedding(request->input_refit_embedding),
      output_tokens(std::get<0>(request->output_group[index])),
      complete_output_tokens(request->input_tokens),
      logprobs(std::get<1>(request->output_group[index])),
      request_target(request->request_target),
      response(request->response),
      cumulative_score(0),
      sampling_config(request->sampling_config),
      waiter(request->waiter),
      step_waiter(request->step_waiter),
      abort_waiter(request->abort_waiter),
      finished(request->finisheds[index]),
      aborted(request->aborted),
      finish_status(request->finish_status),
      output_mutex(request->output_mutex),
      padded_size(request->padded_size),
      beam_search_group(request->beam_search_group),
      span_context(request->span_context),
      timestamp_in_ms(request->timestamp_in_ms),
      is_cudagraph_capture_request(request->is_cudagraph_capture_request),
      req_ctx(request->req_ctx),
      req_fsm(request->req_fsm) {}

InferRequest::~InferRequest() { KLLM_LOG_DEBUG << "req " << req_id << " destroyed."; }

void InferRequest::Notify() {
  for (size_t i = 0; i < req_group.size(); i++) {
    if (!req_group[i]->finished) return;
  }

  if (sampling_config.num_beams > 1) {
    std::sort(beam_search_group.begin(), beam_search_group.end(),
              [](const OutputTuple &a, const OutputTuple &b) { return std::get<2>(a) > std::get<2>(b); });

    for (size_t i = 0; i < req_group.size() && i < beam_search_group.size(); i++) {
      req_group[i]->output_tokens = std::move(std::get<0>(beam_search_group[i]));
      req_group[i]->logprobs = std::move(std::get<1>(beam_search_group[i]));
    }
  }

  for (size_t i = 0; i < req_group.size(); i++) {
    req_group[i]->ClearReqGroup();
  }

  // After a notification, the corresponding request may be destructed.
  // So we return early to avoid accessing any variables referencing it.
  if (aborted) {
    abort_waiter->Notify();
    return;
  }
  if (waiter) {
    waiter->Notify();
    return;
  }
  if (step_waiter) {
    step_waiter->Notify();
  }
}

void InferRequest::NotifyStep() {
  if (sampling_config.num_beams > 1) {
    int output_tokens_len = -1;
    for (size_t i = 0; i < req_group.size(); i++) {
      if (req_group[i]->finished) continue;
      output_tokens_len = output_tokens_len == -1 ? req_group[i]->output_tokens.size() : output_tokens_len;
      if (req_group[i]->output_tokens.size() != (size_t)output_tokens_len) return;
    }
  }

  if (step_waiter) {
    step_waiter->Notify();
  }
}

std::vector<float *> InferRequest::GetLogitsPtr() { return model_instance->GetLogitsPtr(); }

std::vector<std::vector<void *>> InferRequest::GetBlockPtrs() {
  std::vector<std::vector<void *>> block_ptrs;
  for (size_t rank = 0; rank < kv_cache_blocks.size(); ++rank) {
    std::vector<void *> block_ptr(kv_cache_blocks[rank].size());
    GetBlockManager()->SetDeviceId(rank);
    GetBlockManager()->GetBlockPtrs(kv_cache_blocks[rank], block_ptr);
    block_ptrs.push_back(block_ptr);
  }
  return block_ptrs;
}

}  // namespace ksana_llm
