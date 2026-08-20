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

void StampOnce(std::atomic<uint64_t> &slot) {
  if (slot.load(std::memory_order_relaxed) != 0) return;
  uint64_t expected = 0;
  slot.compare_exchange_strong(expected, ProfileTimer::GetCurrentTimeInUs(), std::memory_order_relaxed);
}

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
      pd_v2(*request),
      beam_search_group(request->beam_search_group),
      timestamp_in_us(ProfileTimer::GetCurrentTimeInUs()),
      req_ctx(request->req_ctx),
      prefill_entering_forward(request->prefill_entering_forward),
      scheduled_time_us(request->scheduled_time_us),
      first_token_time_us(request->first_token_time_us),
      finish_time_us(request->finish_time_us),
      prefix_cache_hit_tokens(request->prefix_cache_hit_tokens) {
  draft_tokens.spec = request->draft_tokens;
}

PdV2RequestContext::PdV2RequestContext(Request &request)
    : blocks_allocated_time_us(request.pd_v2_blocks_allocated_time_us),
      prefill_dispatched_time_us(request.pd_v2_prefill_dispatched_time_us),
      prefill_complete_time_us(request.pd_v2_prefill_complete_time_us),
      pf_prefix_cache_hit_tokens(request.pd_v2_pf_prefix_cache_hit_tokens),
      pf_queue_us(request.pd_v2_pf_queue_us),
      pf_compute_us(request.pd_v2_pf_compute_us),
      pf_send_us(request.pd_v2_pf_send_us) {}

InferRequest::~InferRequest() {
  KLLM_LOG_DEBUG << "req " << req_id << " destroyed.";
  // 触发 pd_v2 等模块注册的 teardown 回调，确保异常路径也能清理 connector 状态。
  // 析构函数保持 no-throw，回调异常只记录日志。
  for (auto &cb : on_destroy_callbacks_) {
    try {
      if (cb) cb();
    } catch (const std::exception &e) {
      KLLM_LOG_WARNING << "InferRequest::~InferRequest: on_destroy callback threw: " << e.what()
                       << " (req_id=" << req_id << ")";
    } catch (...) {
      KLLM_LOG_WARNING << "InferRequest::~InferRequest: on_destroy callback threw "
                       << "unknown exception (req_id=" << req_id << ")";
    }
  }
}

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
    if (abort_waiter) {
      abort_waiter->Notify();
      return;
    }
    // PD-V2 path may set aborted=true on a request that never had an abort_waiter
    // (only local_endpoint allocates it). Fall through so any waiter/step_waiter
    // still gets notified instead of dereferencing null.
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

  StampOnce(first_token_time_us);

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
  KLLM_CHECK(inflight_count_ > 0);
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

void InferRequest::RetireChunkedReqs() {
  KLLM_CHECK_WITH_INFO(inflight_count_ > 0 && IsSplitPrefillStep(),
                       "RetireChunkedReqs precondition violated" + ScheduleStateToStr());
  KLLM_CHECK(generated_tokens.empty());
  KLLM_CHECK(accepted_tokens.empty());
  KLLM_CHECK(draft_tokens.size() == 0);

  kv_cached_token_num = forwarding_tokens.size();
  remaining_workload_.generated_token_num = 0;
  remaining_workload_.draft_token_num = 0;
  planning_workload_.Reset();
  planning_workload_.prefill_token_num = remaining_workload_.prefill_token_num;

  ResetInflightTask();
  ++inflight_update_skip_count_;
}

bool InferRequest::UpdateAfterInflightTaskFinished() {
  if (inflight_update_skip_count_ > 0) {
    generated_tokens.clear();
    accepted_tokens.clear();
    draft_tokens.clear();
    --inflight_update_skip_count_;
    return false;
  }
  KLLM_CHECK(inflight_count_ > 0);
  const ScheduleTask &oldest_inflight = inflight_tasks_[0];

  // fast path 深度 2 下, step N 产出 EOS 后仍可能已 launch step N+1 (占位步).
  // 丢弃该多余 step 的 token, 并同步 pop placeholder 队列, 避免 inflight 与 placeholder 长度失配.
  // slow path 在 EOS 后 inflight 立即归零, 不会进入此分支.
  if (is_eos_generated_) {
    generated_tokens.clear();
    accepted_tokens.clear();
    draft_tokens.clear();
    if (!pending_placeholder_positions_.empty()) {
      pending_placeholder_positions_.pop_front();
    }
    return true;
  }

  // placeholder 回填后需清空 generated_tokens, 防止下次 LaunchPlanningTask 重复 append (见函数末尾).
  bool fast_path_placeholder_consumed = false;

  // If task is prefill task and not the last step, drop draft_tokens.
  // generated_tokens should be empty.
  if (IsSplitPrefillStep()) {
    accepted_tokens.clear();
    generated_tokens.clear();
    draft_tokens.clear();
  } else {
    infer_stage = InferStage::kDecode;

    output_mutex.lock();
    if (oldest_inflight.workload.draft_token_num > 0) {
      // replace draft tokens with accepted tokens.
      forwarding_tokens.resize(forwarding_tokens.size() - forwarding_tokens_draft_num + accepted_tokens.size());
      output_tokens.insert(output_tokens.end(), accepted_tokens.begin(), accepted_tokens.end());
      // Accumulate MTP/draft acceptance stats for access-log hit-rate.
      mtp_draft_token_num_ += forwarding_tokens_draft_num;
      mtp_hit_token_num_ += accepted_tokens.size();
      if (req_ctx) {
        (*req_ctx)["mtp_draft_tokens"] = std::to_string(mtp_draft_token_num_);
        (*req_ctx)["mtp_hit_tokens"] = std::to_string(mtp_hit_token_num_);
      }
      accepted_tokens.clear();
    }

    // fast path: launch 时写入的 placeholder 对应「下一 inflight step」的末位;
    // 本函数 retire 的是 oldest inflight, 用其 generated_tokens[0] 回填 placeholder.
    if (!pending_placeholder_positions_.empty() && !generated_tokens.empty()) {
      const size_t pos = pending_placeholder_positions_.front();
      pending_placeholder_positions_.pop_front();
      KLLM_CHECK_WITH_INFO(pos < forwarding_tokens.size(), "Placeholder pos out of range" + ScheduleStateToStr());
      forwarding_tokens[pos] = generated_tokens[0];
      fast_path_placeholder_consumed = true;
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

  // generated_tokens[0] 已写入 forwarding_tokens 对应位置, 须 clear:
  // 否则下次 LaunchPlanningTask 会再次 append, 与 device 侧 ReplaceKernel 时序冲突导致 token 错乱.
  if (fast_path_placeholder_consumed) {
    generated_tokens.clear();
  }
  return true;
}

void InferRequest::ResetInflightTask() {
  // 队首 oldest inflight retire 后左滑队列; slow path (inflight_count_=1) 语义与原单槽 inflight_task_ 一致.
  if (inflight_count_ == 0) {
    return;
  }
  for (size_t i = 0; i + 1 < inflight_count_; ++i) {
    inflight_tasks_[i] = inflight_tasks_[i + 1];
  }
  inflight_tasks_[inflight_count_ - 1].Reset();
  --inflight_count_;
  // pending_placeholder_positions_ 在 UpdateAfterInflightTaskFinished 中随 oldest retire 一并 pop, 此处不处理.
}

void InferRequest::LaunchPlanningTask(bool use_placeholder_for_last_token) {
  KLLM_CHECK_WITH_INFO(!planning_task_.IsEmpty(), "No planning_task" + ScheduleStateToStr());
  // inflight 容量由 BatchScheduler 的 max_depth 约束 (slow=1, fast=2).
  // placeholder 仅当已有在飞 step: 其真实末位 token 尚未经 IPC 回到 Engine.
  KLLM_CHECK_WITH_INFO(inflight_count_ < kMaxInflightDepth, "Inflight full" + ScheduleStateToStr());
  if (use_placeholder_for_last_token) {
    KLLM_CHECK_WITH_INFO(inflight_count_ > 0, "Placeholder launch requires existing inflight" + ScheduleStateToStr());
  }
  assert(planning_task_.workload.GetTokenNum() > 0);

  // 选取要写入的 inflight slot.
  ScheduleTask &new_inflight = inflight_tasks_[inflight_count_];
  new_inflight = planning_task_;
  planning_task_.Reset();
  // pd_v2: when kv_from_remote == true the peer Prefill RDMA-wrote K/V
  // for the prompt and PdV2DecodeHook::OnPrefillComplete pre-populated
  //   - forwarding_tokens = input_tokens         (size N = prompt len)
  //   - generated_tokens  = [Paris]              (just-sampled token, fed
  //                                               below at line 342 into
  //                                               forwarding_tokens for the
  //                                               first decode-step query)
  //   - kv_cached_token_num = N                  (positions 0..N-1 cached)
  //   - infer_stage = kDecode
  // The default reset `kv_cached_token_num = forwarding_tokens.size()`
  // would clobber it back to N (no-op on first step where the two are
  // already equal — but defensive: subsequent decode steps' line 285
  // bumps kv_cached, and we want this branch to be a coherent inverse
  // for the non-reset case anyway). Skip the reset on pd_v2 reqs to
  // make the contract explicit and to keep the branch obviously safe
  // even if the hook's pre-populated state diverges by 1 from the
  // forwarding_tokens length.
  if (!pd_v2.kv_from_remote) {
    SetKvCachedTokenNum(forwarding_tokens.size());
  }
  if (new_inflight.workload.prefill_token_num > 0) {
    size_t forwarded_token_num = forwarding_tokens.size();
    if (forwarded_token_num == new_inflight.workload.prefill_start_offset) {
      forwarding_tokens.insert(
          forwarding_tokens.end(), prefilling_tokens_.begin() + forwarded_token_num,
          prefilling_tokens_.begin() + forwarded_token_num + new_inflight.workload.prefill_token_num);
    } else {
      // Prefix cache hit
      forwarding_tokens.assign(prefilling_tokens_.begin(), prefilling_tokens_.begin() +
                                                               new_inflight.workload.prefill_start_offset +
                                                               new_inflight.workload.prefill_token_num);
      SetKvCachedTokenNum(new_inflight.workload.prefill_start_offset);
    }

    // 末步 prefill 追加 draft; IsSplitPrefillStep 读 oldest inflight (slow path 下即本次新 push 的 task).
    if (!IsSplitPrefillStep()) {
      const std::vector<int> &all_draft_tokens = draft_tokens.GetDraftTokens();
      const size_t draft_token_num = all_draft_tokens.size();
      forwarding_tokens.insert(forwarding_tokens.end(), all_draft_tokens.begin(), all_draft_tokens.end());
      new_inflight.workload.draft_token_num = draft_token_num;
      new_inflight.workload.sampling_token_num =
          std::max(kStepGenerateTokenNum + draft_token_num, logits_custom_length);
    }
  } else {
    auto merged_draft_tokens = draft_tokens.GetDraftTokens();
    size_t resource_ready_token_num = new_inflight.workload.generated_token_num + new_inflight.workload.draft_token_num;
    KLLM_CHECK(generated_tokens.size() <= resource_ready_token_num);

    // decode 末位 token 语义:
    //   slow path — generated_tokens 已由上一步 IPC 回流, 直接 append;
    //   fast path — 上一步采样结果尚在 device, Engine 用 placeholder 占位, Executor ReplaceKernel 回填.
    // 当前版本 kStepGenerateTokenNum=1 (FastPathController 启动期约束), placeholder 仅占 1 位.
    size_t effective_generated_num = generated_tokens.size();
    if (use_placeholder_for_last_token) {
      KLLM_CHECK_WITH_INFO(generated_tokens.size() <= 1,
                           "Fast path placeholder currently only supports kStepGenerateTokenNum=1, generated_tokens>1" +
                               ScheduleStateToStr());
      if (generated_tokens.empty()) {
        forwarding_tokens.push_back(kFastPathPlaceholderTokenId);
        pending_placeholder_positions_.push_back(forwarding_tokens.size() - 1);
        effective_generated_num = 1;  // workload 按 1 个 generated token 计
      } else {
        // 防御: 已有真实 token 时优先使用 (正常 fast path 流水线不应走到此分支).
        forwarding_tokens.insert(forwarding_tokens.end(), generated_tokens.begin(), generated_tokens.end());
      }
    } else {
      forwarding_tokens.insert(forwarding_tokens.end(), generated_tokens.begin(), generated_tokens.end());
    }

    // 资源 budget 用 effective_generated_num (placeholder 计 1, slow path 用 generated_tokens.size).
    resource_ready_token_num -= effective_generated_num;
    size_t draft_token_num = std::min(resource_ready_token_num, merged_draft_tokens.size());
    forwarding_tokens.insert(forwarding_tokens.end(), merged_draft_tokens.begin(),
                             merged_draft_tokens.begin() + draft_token_num);

    // Change new_inflight workload to real workload
    // fast path placeholder 时 effective_generated_num=1 (代表 forward 处理 1 个 slot, 与 slow path 等价).
    new_inflight.workload.generated_token_num = effective_generated_num;
    new_inflight.workload.draft_token_num = draft_token_num;
    new_inflight.workload.sampling_token_num = kStepGenerateTokenNum + draft_token_num;
    if (draft_token_num < merged_draft_tokens.size()) {
      draft_tokens.TruncDraft(draft_token_num);
    }
  }
  sampling_token_num = new_inflight.workload.sampling_token_num;
  forwarding_tokens_draft_num = new_inflight.workload.draft_token_num;

  ++inflight_count_;
  KLLM_LOG_SCHEDULER << ScheduleStateToStr();
  assert(new_inflight.workload.GetTokenNum() > 0);
  assert(forwarding_tokens.size() > kv_cached_token_num);
}

std::string InferRequest::ScheduleStateToStr() const {
  std::stringstream ss;
  ss << " schedule_state={ req_id=" << req_id << ", is_stopped=" << is_stopped_
     << ", inflight_count_=" << inflight_count_ << ", inflight_tasks_[0]=" << inflight_tasks_[0].workload.ToString()
     << ", inflight_tasks_[1]=" << inflight_tasks_[1].workload.ToString()
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
