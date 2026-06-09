/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "runtime/infer_request.h"

#include <atomic>
#include <limits>
#include <sstream>
#include <vector>
#include "profiler/timer.h"
#include "runtime/infer_stage.h"
#include "utils/memory_utils.h"
#include "utils/request.h"
#include "utils/singleton.h"
#include "utils/status.h"
#include "utils/string_utils.h"
#include "utils/tensor.h"

namespace ksana_llm {
InferRequest::InferRequest(std::shared_ptr<Request> &request, const int index)
    : req_id(request->req_ids[index]),
      logits_custom_length(request->logits_custom_length),
      input_tokens(request->input_tokens),
      input_refit_embedding(request->input_refit_embedding),
      output_tokens(std::get<0>(request->output_group[index])),
      logprobs(std::get<1>(request->output_group[index])),
      return_cache_stat(request->return_cache_stat),
      cache_stat(request->cache_stat),
      request_target(request->request_target),
      response(request->response),
      sampling_config(request->sampling_config),
      waiter(request->waiter),
      step_waiter(request->step_waiter),
      abort_waiter(request->abort_waiter),
      structured_generator_config(request->structured_generator_config),
      finished(request->finisheds[index]),
      aborted(request->aborted),
      finish_status(request->finish_status),
      output_mutex(request->output_mutex),
      kv_comm_group_key(request->kv_comm_group_key),
      beam_search_group(request->beam_search_group),
      timestamp_in_us(ProfileTimer::GetCurrentTimeInUs()),
      req_ctx(request->req_ctx),
      prefill_entering_forward(request->prefill_entering_forward) {
  draft_tokens.spec = request->draft_tokens;
}

InferRequest::~InferRequest() { KLLM_LOG_DEBUG << "req " << req_id << " destroyed."; }

std::string InferRequest::PrintKVBlockIds(bool print_details) const {
  std::ostringstream ss;
  ss << ", kv_cache_blocks_size:" << kv_cache_blocks.size() << ", kv_cache_blocks: {";
  if (print_details) {
    ss << "{ ";
    for (auto blk_id : kv_cache_blocks) {
      ss << blk_id << ", ";
    }
    ss << "}, ";
  }
  ss << "}";
  return ss.str();
}

std::string InferRequest::ToString(bool print_details) const {
  std::ostringstream oss;
  oss << " req(req_id:" << req_id << ", step:" << step << ", sampling_token_num:" << sampling_token_num
      << ", kv_cached_token_num:" << kv_cached_token_num << ", prefix_cache_len:" << prefix_cache_len
      << ", input_tokens_size:" << input_tokens.size() << ", output_tokens_size:" << output_tokens.size()
      << ", forwarding_tokens_size:" << forwarding_tokens.size() << ", draft_tokens_size:" << draft_tokens.size()
      << ", accepted_tokens_size:" << accepted_tokens.size() << ", generated_token_size:" << generated_tokens.size()
      << PrintKVBlockIds(print_details) << ", finished:" << finished << ", aborted:" << aborted
      << ", finish_status:" << finish_status.ToString() << ", infer_stage:" << static_cast<int>(infer_stage) << " ) ";
  return oss.str();
}

std::ostream &operator<<(std::ostream &os, const InferRequest &req) {
  os << req.ToString();
  return os;
}

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

const std::vector<int> &InferRequest::GetInflightSequence() const { return forwarding_tokens; }

size_t InferRequest::GetInflightSequenceLen() const { return forwarding_tokens.size(); }

size_t InferRequest::GetInflightQueryLen() const { return forwarding_tokens.size() - kv_cached_token_num; }

size_t InferRequest::GetInflightSamplingTokenNum() const { return sampling_token_num; }

size_t InferRequest::GetPlanningSequenceLen() const {
  if (planning_workload_.prefill_token_num > 0) {
    return planning_workload_.prefill_start_offset + planning_workload_.GetTokenNum();
  }
  // Decoding
  return forwarding_tokens.size() + planning_workload_.GetTokenNum();
}

size_t InferRequest::GetPlanningQueryLen() const { return planning_workload_.GetTokenNum(); }

size_t InferRequest::GetPlanningSamplingTokenNum() const { return planning_workload_.sampling_token_num; }

void InferRequest::SetKvCachedTokenNum(size_t num) {
  kv_cached_token_num = num;
  prefix_cache_len = num;
}

void InferRequest::SetCacheHitStatus(size_t shared_token_num, bool is_first_prefill_step) {
  if (!return_cache_stat || infer_stage != InferStage::kContext || !is_first_prefill_step) {
    return;
  }

  cache_stat.clear();
  cache_stat.emplace_back(shared_token_num, shared_token_num);

  // return if not flexible cached
  if (flexible_cached_copy_tasks.empty()) {
    return;
  }

  std::set<int> flexible_cached_token_idx;
  for (const auto &task : flexible_cached_copy_tasks) {
    flexible_cached_token_idx.insert(task.dst_token_idx_);
  }

  int start = *flexible_cached_token_idx.begin();
  int prev = start;
  for (auto it = std::next(flexible_cached_token_idx.begin()); it != flexible_cached_token_idx.end(); ++it) {
    int curr = *it;
    if (curr != prev + 1) {
      cache_stat.emplace_back(start, prev + 1);
      start = curr;
    }
    prev = curr;
  }
  cache_stat.emplace_back(start, prev + 1);
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

std::vector<int> InferRequest::GetKVOccupiedDevices() {
  std::vector<int> kv_occupied_devices;
  kv_occupied_devices = cache_manager->GetBlockAllocatorGroup()->GetBlockAllocatorDevices();
  KLLM_LOG_DEBUG << "req_id: " << req_id << ", kv_occupied_devices: " << Vector2Str(kv_occupied_devices) << ".";
  return kv_occupied_devices;
}

void InferRequest::ResetPrefillingTokens() {
  infer_stage = InferStage::kContext;
  prefilling_tokens_ = output_tokens;
  kv_cached_token_num = 0;
  step = 0;
  suggested_draft_num = 0;
  prefix_cache_len = 0;
  remaining_workload_.Reset();
  remaining_workload_.prefill_token_num = prefilling_tokens_.size();

  const size_t draft_token_num = draft_tokens.size();
  // Assumption: if logits_custom_length > 0, then sampling_token_num = logits_custom_length
  //             because user asked to generate logits for a specific number of tokens
  //             In runtime implementation, sampling_token_num is used to determine the number of logits to generate
  remaining_workload_.sampling_token_num = std::max(kStepGenerateTokenNum + draft_token_num, logits_custom_length);

  planning_workload_ = remaining_workload_;
  sampling_token_num = planning_workload_.sampling_token_num;
}

void InferRequest::AppendCachedTokens(const std::vector<int> &tokens) {
  std::lock_guard<std::mutex> guard(output_mutex);
  KLLM_LOG_DEBUG << "InferRequest::AppendCachedTokens: req_id=" << req_id << ", appending " << tokens.size()
                 << " tokens"
                 << ", output_tokens_before=" << output_tokens.size()
                 << ", prefilling_tokens_before=" << prefilling_tokens_.size();
  for (int token : tokens) {
    output_tokens.push_back(token);
    prefilling_tokens_.push_back(token);
  }
  KLLM_LOG_DEBUG << "InferRequest::AppendCachedTokens: req_id=" << req_id
                 << ", output_tokens_after=" << output_tokens.size()
                 << ", prefilling_tokens_after=" << prefilling_tokens_.size();
}

void InferRequest::SetInflightTaskGenResultEstimation(size_t generated_token_num, size_t draft_token_num) {
  KLLM_CHECK(!inflight_task_.IsEmpty());
  planning_workload_.generated_token_num = generated_token_num;
  planning_workload_.draft_token_num = draft_token_num;
  remaining_workload_.generated_token_num = generated_token_num;
  remaining_workload_.draft_token_num = draft_token_num;
}

void InferRequest::SetRemainingWorkload(const ScheduleTaskWorkload &workload) { remaining_workload_ = workload; }

void InferRequest::SetPlanningWorkload(const ScheduleTaskWorkload &workload) { planning_workload_ = workload; }

void InferRequest::SetPlanningTask() {
  KLLM_CHECK(planning_task_.IsEmpty());
  planning_task_.workload = planning_workload_;

  KLLM_CHECK(remaining_workload_.prefill_token_num >= planning_workload_.prefill_token_num);
  remaining_workload_.prefill_token_num -= planning_workload_.prefill_token_num;
  remaining_workload_.prefill_start_offset += planning_workload_.prefill_token_num;
  remaining_workload_.generated_token_num = 0;
  remaining_workload_.draft_token_num = 0;
  KLLM_CHECK((remaining_workload_.prefill_token_num + remaining_workload_.prefill_start_offset) ==
             prefilling_tokens_.size());
  planning_workload_.Reset();
}

void InferRequest::UpdateAfterInflightTaskFinished() {
  // If task is prefill task and not the last step, drop draft_tokens.
  // generated_tokens should be empty.
  if (IsSplitPrefillStep()) {
    generated_tokens.clear();
    draft_tokens.clear();
  } else {
    infer_stage = InferStage::kDecode;

    output_mutex.lock();
    if (inflight_task_.workload.draft_token_num > 0) {
      // replace draft tokens with accepted tokens.
      forwarding_tokens.resize(forwarding_tokens.size() - forwarding_tokens_draft_num + accepted_tokens.size());
      output_tokens.insert(output_tokens.end(), accepted_tokens.begin(), accepted_tokens.end());
      accepted_tokens.clear();
    }

    // append new tokens to output_tokens
    output_tokens.insert(output_tokens.end(), generated_tokens.begin(), generated_tokens.end());
    // GenerationController makes sure eos only appears at the end of accepted_tokens + generated_tokens
    if (std::find(sampling_config.stop_token_ids.begin(), sampling_config.stop_token_ids.end(), output_tokens.back()) !=
        sampling_config.stop_token_ids.end()) {
      KLLM_LOG_DEBUG << "req " << req_id << " finished. output_tokens.size=" << output_tokens.size()
                     << ", eos token=" << output_tokens.back();
      is_eos_generated_ = true;
    }
    output_mutex.unlock();
  }
  // current token has kv_cache
  kv_cached_token_num = forwarding_tokens.size();

  // generated token and draft token are new workload to be processed
  remaining_workload_.generated_token_num = generated_tokens.size();
  remaining_workload_.draft_token_num = draft_tokens.size();

  // Adjust planning workload for scheduling
  if (remaining_workload_.prefill_token_num > 0 &&
      (remaining_workload_.generated_token_num > 0 || remaining_workload_.draft_token_num > 0)) {
    KLLM_LOG_SCHEDULER << ScheduleStateToStr();
    assert(false);
  }

  planning_workload_.Reset();
  if (remaining_workload_.prefill_token_num > 0) {
    planning_workload_.prefill_token_num = remaining_workload_.prefill_token_num;
  } else {
    planning_workload_.generated_token_num = generated_tokens.size();
    planning_workload_.draft_token_num = draft_tokens.size();
  }
}

void InferRequest::LaunchPlanningTask() {
  KLLM_CHECK_WITH_INFO(!planning_task_.IsEmpty(), "No planning_task" + ScheduleStateToStr());
  KLLM_CHECK_WITH_INFO(inflight_task_.IsEmpty(), "Has inflight_task" + ScheduleStateToStr());
  assert(planning_task_.workload.GetTokenNum() > 0);

  inflight_task_ = planning_task_;
  planning_task_.Reset();
  SetKvCachedTokenNum(forwarding_tokens.size());
  if (inflight_task_.workload.prefill_token_num > 0) {
    size_t forwarded_token_num = forwarding_tokens.size();
    if (forwarded_token_num == inflight_task_.workload.prefill_start_offset) {
      forwarding_tokens.insert(
          forwarding_tokens.end(), prefilling_tokens_.begin() + forwarded_token_num,
          prefilling_tokens_.begin() + forwarded_token_num + inflight_task_.workload.prefill_token_num);
    } else {
      // Prefix cache hit
      forwarding_tokens.assign(prefilling_tokens_.begin(), prefilling_tokens_.begin() +
                                                               inflight_task_.workload.prefill_start_offset +
                                                               inflight_task_.workload.prefill_token_num);
      SetKvCachedTokenNum(inflight_task_.workload.prefill_start_offset);
    }

    // Append draft tokens at the last step of prefill
    if (!IsSplitPrefillStep()) {
      const std::vector<int> &all_draft_tokens = draft_tokens.GetDraftTokens();
      const size_t draft_token_num = all_draft_tokens.size();
      forwarding_tokens.insert(forwarding_tokens.end(), all_draft_tokens.begin(), all_draft_tokens.end());
      inflight_task_.workload.draft_token_num = draft_token_num;
      inflight_task_.workload.sampling_token_num = kStepGenerateTokenNum + draft_token_num;
    }
  } else {
    auto merged_draft_tokens = draft_tokens.GetDraftTokens();
    size_t resource_ready_token_num =
        inflight_task_.workload.generated_token_num + inflight_task_.workload.draft_token_num;
    KLLM_CHECK(generated_tokens.size() <= resource_ready_token_num);
    forwarding_tokens.insert(forwarding_tokens.end(), generated_tokens.begin(), generated_tokens.end());

    resource_ready_token_num -= generated_tokens.size();
    size_t draft_token_num = std::min(resource_ready_token_num, merged_draft_tokens.size());
    forwarding_tokens.insert(forwarding_tokens.end(), merged_draft_tokens.begin(),
                             merged_draft_tokens.begin() + draft_token_num);

    // Change inflight_task_ workload to real workload
    inflight_task_.workload.generated_token_num = generated_tokens.size();
    inflight_task_.workload.draft_token_num = draft_token_num;
    inflight_task_.workload.sampling_token_num = kStepGenerateTokenNum + draft_token_num;
    if (draft_token_num < merged_draft_tokens.size()) {
      draft_tokens.TruncDraft(draft_token_num);
    }
  }
  sampling_token_num = inflight_task_.workload.sampling_token_num;
  forwarding_tokens_draft_num = inflight_task_.workload.draft_token_num;
  KLLM_LOG_SCHEDULER << ScheduleStateToStr();
  assert(inflight_task_.workload.GetTokenNum() > 0);
  assert(forwarding_tokens.size() > kv_cached_token_num);
}

std::string InferRequest::ScheduleStateToStr() const {
  std::stringstream ss;
  ss << " schedule_state={ req_id=" << req_id << ", is_stopped=" << is_stopped_
     << ", inflight_task_=" << inflight_task_.workload.ToString()
     << ", planning_task_=" << planning_task_.workload.ToString()
     << ", remaining_workload_=" << remaining_workload_.ToString()
     << ", planning_workload_= " << planning_workload_.ToString()
     << ", forwarding_tokens.size=" << forwarding_tokens.size() << ", generated_tokens=" << Vector2Str(generated_tokens)
     << ", accepted_tokens=" << Vector2Str(accepted_tokens)
     << ", draft_tokens=" << Vector2Str(draft_tokens.GetDraftTokens())
     << ", output_tokens.size=" << output_tokens.size() << ", is_eos_generated=" << is_eos_generated_;

  // Print request context headers
  if (req_ctx) {
    auto it_ip = req_ctx->find("x-remote-ip");
    auto it_group = req_ctx->find("kv-comm-group-key");
    auto it_reqid = req_ctx->find("request-id");
    auto it_trace = req_ctx->find("traceparent");
    ss << ", x_remote_ip=" << (it_ip != req_ctx->end() ? it_ip->second : "-")
       << ", kv_comm_group_key=" << (it_group != req_ctx->end() ? it_group->second : "-")
       << ", request_id=" << (it_reqid != req_ctx->end() ? it_reqid->second : "-")
       << ", traceparent=" << (it_trace != req_ctx->end() ? it_trace->second : "-");
  }
  ss << " }";
  return ss.str();
}

}  // namespace ksana_llm
