/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/batch_scheduler/strategy/continuous_batching.h"
#include <cmath>
#include <memory>

#include "base_strategy.h"
#include "ksana_llm/cache_manager/prefix_cache_manager.h"
#include "ksana_llm/profiler/reporter.h"
#include "ksana_llm/profiler/trace_event_recorder.h"
#include "ksana_llm/runtime/infer_request.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/memory_utils.h"
#include "ksana_llm/utils/request_packer.h"
#include "ksana_llm/utils/ret_code.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/stop_checker.h"

namespace ksana_llm {

ContinuousBatchingStrategy::ContinuousBatchingStrategy(const BatchSchedulerConfig &batch_scheduler_config, int tp_num,
                                                       std::shared_ptr<BatchState> batch_state)
    : BaseScheduleStrategy(batch_scheduler_config, tp_num, batch_state) {}

bool ContinuousBatchingStrategy::CheckRequestTimeout(const std::shared_ptr<InferRequest> req) {
  return batch_state_->schedule_time_in_ms >= req->timestamp_in_ms + batch_scheduler_config_.waiting_timeout_in_ms;
}

bool ContinuousBatchingStrategy::CheckRequestFinish(const std::shared_ptr<InferRequest> req) {
  std::vector<int> &stop_token_ids = req->sampling_config.stop_token_ids;
#ifdef CLEAR_CACHE
  if (req->input_tokens.size() == 2 && req->input_tokens[0] == 0 && req->input_tokens[1] == 0) {
    size_t free_block_num;
    auto prefix_cache_manager_ptr = dynamic_cast<PrefixCacheManager *>(cache_manager_.get());
    if (prefix_cache_manager_ptr != nullptr) {
      prefix_cache_manager_ptr->FreeCachedBlocks(1e8, free_block_num);
      KLLM_LOG_WARNING << "cache_manager free " << free_block_num << " blocks.";
    }
  }
#endif
  if (std::find(stop_token_ids.begin(), stop_token_ids.end(), req->output_tokens.back()) != stop_token_ids.end() ||
      (req->sampling_config.max_new_tokens > 0 &&
       req->output_tokens.size() >= req->input_tokens.size() + req->sampling_config.max_new_tokens) ||
      req->output_tokens.size() >= batch_scheduler_config_.max_token_len ||
      (req->req_fsm != nullptr && req->req_fsm->IsStopState(req->fsm_state_id))) {
    stop_checker_->CheckCompleteStopStrings(req, tokenizer_);
    return true;
  }

  // When stop strings are checked and matched, stop early
  return stop_checker_->CheckIncrementalStopStrings(req, tokenizer_);
}

void ContinuousBatchingStrategy::ExtendTokensWithRetokenization(std::shared_ptr<InferRequest> req) {
  // First, convert the output_tokens into a string A (excluding the input_tokens part).
  // Then, concatenate the original prompt, the string A, and the fixed string defined in the structured output regex.
  // Finally, the combined string is encoded using a tokenizer to obtain the new output_tokens.
  std::shared_ptr<RequestPacker> request_packer = Singleton<RequestPacker>::GetInstance();
  std::shared_ptr<FiniteStateNode> state = req->req_fsm->GetState(req->fsm_state_id);

  // detokenize the newly generated token sequence to obtain the corresponding string.
  std::string structure_text = "";
  std::vector<int> new_tokens;
  if (req->output_tokens.size() > req->input_tokens.size()) {
    new_tokens = std::vector<int>(req->output_tokens.begin() + req->input_tokens.size(), req->output_tokens.end());
    request_packer->DeTokenize(new_tokens, structure_text);
  }

  // collapse consecutive non-generation states into one.
  while (state->state_type_ == FiniteStateType::NON_GENERATION_STATE) {
    structure_text += state->GetEdge().first;
    state = req->req_fsm->GetState(state->GetNextStateId());
  }

  // Clear the request for assignment.
  req->output_tokens = req->input_tokens;
  new_tokens.clear();

  // Assign values.
  request_packer->Tokenize(structure_text, new_tokens, false);
  req->output_tokens.insert(req->output_tokens.end(), new_tokens.begin(), new_tokens.end());
  req->complete_output_tokens = req->output_tokens;
  req->NotifyStep();

  // State transition.
  req->fsm_state_id = state->state_id_;
}

void ContinuousBatchingStrategy::ExtendTokensWithoutRetokenization(std::shared_ptr<InferRequest> req) {
  // Unlike the ExtendTokensWithRetokenization, in this function, the fixed string defined in the structured
  // output regex has already been preconverted into the corresponding token_id_list. Therefore, we only need to
  // append the new token to the output_tokens.
  std::shared_ptr<FiniteStateNode> state = req->req_fsm->GetState(req->fsm_state_id);

  while (state->state_type_ == FiniteStateType::NON_GENERATION_STATE) {
    std::vector<int> jump_tokens = req->req_fsm->GetStringTokens(state->GetEdge().first);
    req->output_tokens.insert(req->output_tokens.end(), jump_tokens.begin(), jump_tokens.end());
    state = req->req_fsm->GetState(state->GetNextStateId());
  }

  req->complete_output_tokens = req->output_tokens;

  req->NotifyStep();
  req->fsm_state_id = state->state_id_;
}

void ContinuousBatchingStrategy::CheckJumpForwardRequest(std::shared_ptr<InferRequest> req) {
  if (req->req_fsm.get() == nullptr) {
    return;
  }
  std::string output_text = "";
  std::shared_ptr<FiniteStateNode> state = req->req_fsm->GetState(req->fsm_state_id);
  if (state->state_type_ == FiniteStateType::NON_GENERATION_STATE) {
    // non-generation states
    // Refer to https://lmsys.org/blog/2024-02-05-compressed-fsm/#tokenization-boundary-handling
    // We use retokenizer to avoid issues related to tokenization boundaries during constrained decoding.
    ExtendTokensWithRetokenization(req);
    // ExtendTokensWithoutRetokenization(req);
    if (req->req_fsm->IsStopState(req->fsm_state_id)) {
      return;
    }
    req->infer_stage = InferStage::STAGE_CONTEXT;
  } else if (state->state_type_ == FiniteStateType::GENERATION_STATE) {
    // generation states
    size_t next_state_id = state->GetNextStateId(req->output_tokens.back());
    if (next_state_id != req->fsm_state_id) {
      req->prefix_cache_len = req->output_tokens.size() - 1;
      req->req_fsm->CheckFSMPopToken(req->fsm_state_id, req->output_tokens);
      req->fsm_state_id = next_state_id;
      CheckJumpForwardRequest(req);
    }
  } else if (state->state_type_ == FiniteStateType::FINAL_STATE) {
    // stop states
    KLLM_LOG_DEBUG << fmt::format("Req {} got an EOF status, will stop soon.", req->req_id);
  } else {
    KLLM_LOG_ERROR << fmt::format("Status {} got an invalid status type : {}", req->fsm_state_id, state->state_type_);
  }
  return;
}

void ContinuousBatchingStrategy::RecomputeRequest(std::shared_ptr<InferRequest> req) {
  // Add request to the begining of waiting queue.
  req->kv_cache_blocks.clear();
  req->kv_cache_blocks.resize(tp_num_);
  req->infer_stage = InferStage::STAGE_CONTEXT;
  req->step = 0;
  req->complete_output_tokens = req->output_tokens;
}

void ContinuousBatchingStrategy::StopRequest(std::shared_ptr<InferRequest> req, Status req_status) {
  req->finish_status = req_status;
  req->finished = true;

#ifdef ENABLE_RECORD_EVENT
  if (req_status.GetCode() == RET_SUCCESS) {
    RECORD_TRACE_EVENT_TAG("FinishReq", TraceEventType::FinishReq, std::to_string(req->req_id),
                           TRACE_THREAD_NAME_PREFILL_DECODE);
  } else {
    RECORD_TRACE_EVENT_TAG(req_status.GetMessage(), TraceEventType::DropReq, std::to_string(req->req_id),
                           TRACE_THREAD_NAME_PREFILL_DECODE);
  }
#endif

  req->Notify();
}

void ContinuousBatchingStrategy::UpdateRunningRequests(size_t &total_needed_block_num) {
  for (auto it = batch_state_->running_queue.begin(); it != batch_state_->running_queue.end();) {
    auto req = *it;
    // All req here should be decode now.
    req->infer_stage = InferStage::STATE_DECODE;
    opentelemetry::common::KeyValueIterableView<std::unordered_map<std::string, std::string>> attributes(*req->req_ctx);

    // Always update cache manager, even if request is finished.
    Status status = cache_manager_->UpdateRequestTokens(req->req_id, req->output_tokens, req->kv_cache_blocks);
    if (req->output_tokens.size() <= req->complete_output_tokens.size()) {
      req->infer_stage = InferStage::STAGE_CONTEXT;
      req->output_tokens = req->complete_output_tokens;
      batch_state_->waiting_queue.insert(batch_state_->waiting_queue.begin(), req);
      it = batch_state_->running_queue.erase(it);
      continue;
    }

    if (!status.OK()) {
      KLLM_LOG_ERROR << "UpdateRequestTokens req " << req->req_id
                     << " error, recompute it, info: " << status.GetMessage();

      cache_manager_->DestroyFinishedRequest(req->req_id);

      RecomputeRequest(req);

      batch_state_->waiting_queue.insert(batch_state_->waiting_queue.begin(), req);
      it = batch_state_->running_queue.erase(it);
      continue;
    }

    CheckJumpForwardRequest(req);

    // Check if finished.
    if (CheckRequestFinish(req)) {
      cache_manager_->DestroyFinishedRequest(req->req_id);
      auto end_time = ProfileTimer::GetCurrentTimeInMs();
      size_t output_token_num = req->output_tokens.size();
      uint64_t duration = end_time - req->timestamp_in_ms;

      REPORT_METRIC(forward_cost_time_ms, duration, attributes);
      REPORT_METRIC(metric_output_token_num, output_token_num, attributes);

      // TODO(shawnding): Adjust to microsecond precision
      if (duration != 0) {
        REPORT_METRIC(time_to_per_output_token_ms, output_token_num / duration, attributes);
      } else {
        KLLM_LOG_WARNING << fmt::format(
            "Req duration is zero, req_id: {}, input_token_num: {} ,output_token_num: {} "
            "req start time is: {} req end time is: {}",
            req->req_id, req->input_tokens.size(), output_token_num, req->timestamp_in_ms, end_time);
        REPORT_METRIC(time_to_per_output_token_ms, output_token_num, attributes);
      }

      StopRequest(req, Status(RET_SUCCESS));
      it = batch_state_->running_queue.erase(it);

      continue;
    }

    // Check timeout
    if (CheckRequestTimeout(req)) {
      KLLM_LOG_DEBUG << "req " << req->req_id << " timeout in running.";

      cache_manager_->DestroyFinishedRequest(req->req_id);

      StopRequest(req, Status(RET_TIMEOUT, "running timeout."));
      it = batch_state_->running_queue.erase(it);
      REPORT_COUNTER(forward_req_timeout_num, static_cast<size_t>(1), attributes);

      continue;
    }

    // Check abort.
    if (req->aborted) {
      KLLM_LOG_DEBUG << "req " << req->req_id << " aborted in running.";

      cache_manager_->DestroyFinishedRequest(req->req_id);

      StopRequest(req, Status(RET_TERMINATED, "client aborted."));
      it = batch_state_->running_queue.erase(it);
      REPORT_COUNTER(forward_req_aborted_num, static_cast<size_t>(1), attributes);

      continue;
    }

    // Not finished, notify streaming iterator.
    req->NotifyStep();

    total_needed_block_num += cache_manager_->GetRequestStepBlockNumber(req->req_id, req->output_tokens.size());

    ++it;
  }
}

Status ContinuousBatchingStrategy::AllocateRequestBlocksWithRetry(std::shared_ptr<InferRequest> req,
                                                                  size_t &total_needed_block_num,
                                                                  size_t &step_block_num, bool &allocate_block_succ,
                                                                  bool &skip_swapout_check) {
  Status status = cache_manager_->AllocateRequestBlocks(req->req_id, step_block_num, req->kv_cache_blocks);
  if (!status.OK()) {
    KLLM_LOG_ERROR << "Alllocate blocks error, info: " << status.GetMessage();
    KLLM_LOG_DEBUG << "Waiting all pending swapout requests done, and retry.";
    MergePendingSwapoutRequests(true, false);

    // Try the allocation again after all swapout finished.
    status = cache_manager_->AllocateRequestBlocks(req->req_id, step_block_num, req->kv_cache_blocks);
    if (!status.OK()) {
      KLLM_LOG_ERROR << "Alllocate blocks error again, recompute it, info: " << status.GetMessage();
      allocate_block_succ = false;
    } else {
      total_needed_block_num -= step_block_num;
      skip_swapout_check = true;
    }
  } else {
    total_needed_block_num -= step_block_num;
    skip_swapout_check = true;
  }
  return status;
}

void ContinuousBatchingStrategy::ProcessRunningQueue() {
  KLLM_LOG_DEBUG << "ProcessRunningQueue invoked, running queue size:" << batch_state_->running_queue.size()
                 << ", free block num:" << cache_manager_->GetUsableBlockNumber()
                 << ", future block num:" << cache_manager_->GetFutureBlockNumber();

  // Merge pending swapin requests, continue running.
  Status status = MergePendingSwapinRequests(false, true);
  if (!status.OK()) {
    KLLM_LOG_ERROR << "ProcessRunningQueue error, info: " << status.GetMessage();
  }

  size_t total_needed_block_num = 0;

  // The requests that should be recomputed.
  std::vector<std::shared_ptr<InferRequest>> recompute_requests;

  // Update cache manager, process finished and timeout requests.
  UpdateRunningRequests(total_needed_block_num);

  if (batch_state_->running_queue.empty() && batch_state_->waiting_queue.empty()) {
    // If running & waiting queue in current step is empty, wait all swapin jobs done if existed.
    // In order to make sure the schedule result not empty.
    if (!batch_state_->swapin_pending_requests_.empty()) {
      Status status = MergePendingSwapinRequests(true, false);
      if (!status.OK()) {
        KLLM_LOG_ERROR << "MergePendingSwapinRequests error, info: " << status.GetMessage();
      }

      KLLM_LOG_DEBUG << "ProcessRunningQueue update, running queue size:" << batch_state_->running_queue.size()
                     << ", free block num:" << cache_manager_->GetUsableBlockNumber()
                     << ", future block num:" << cache_manager_->GetFutureBlockNumber();
    }
  }

  // Swapout necessary blocks.
  bool skip_swapout_check = false;
  for (size_t running_batch_size = batch_state_->running_queue.size(); running_batch_size > 0; --running_batch_size) {
    auto it = batch_state_->running_queue.begin() + running_batch_size - 1;
    auto req = *it;

    // No need to check max_batch_size and max_step_tokens here.
    size_t swapout_block_threshold = std::ceil(running_batch_size * batch_scheduler_config_.swapout_block_threshold);

    // Whether the allocation operation is successful.
    bool allocate_block_succ = true;

    size_t step_block_num = cache_manager_->GetRequestStepBlockNumber(req->req_id, req->output_tokens.size());
    size_t total_free_block_num = cache_manager_->GetUsableBlockNumber();
    size_t future_free_block_num = cache_manager_->GetFutureBlockNumber();

    // If last request have not enough blocks, wait all swapout done.
    if (running_batch_size == 1 && step_block_num > total_free_block_num) {
      KLLM_LOG_DEBUG << "Not enough blocks for last req " << req->req_id
                     << ", waiting all pending swapout requests done.";
      MergePendingSwapoutRequests(true, false);

      // Update block num.
      total_free_block_num = cache_manager_->GetUsableBlockNumber();
      future_free_block_num = cache_manager_->GetFutureBlockNumber();
    }

    // never swap out last request.
    if (skip_swapout_check ||
        (step_block_num <= total_free_block_num && total_needed_block_num <= total_free_block_num &&
         total_needed_block_num + swapout_block_threshold <= total_free_block_num + future_free_block_num)) {
      KLLM_LOG_DEBUG << "running req " << req->req_id << " continue running, step_block_num:" << step_block_num
                     << ", current_block_num:" << req->kv_cache_blocks[0].size()
                     << ", current_token_size:" << req->output_tokens.size();

      status = AllocateRequestBlocksWithRetry(req, total_needed_block_num, step_block_num, allocate_block_succ,
                                              skip_swapout_check);
      if (status.OK()) {
        continue;
      }
    }

    // If allocation failed, disable split fuse.
    if (batch_state_->waiting_queue.size() > 0 && batch_scheduler_config_.split_fuse_token_num > 0) {
      for (auto it = batch_state_->waiting_queue.begin(); it != batch_state_->waiting_queue.end(); it++) {
        auto req = *it;
        if (req->kv_cache_blocks[0].size() > 0) {
          cache_manager_->DestroyFinishedRequest(req->req_id);
          req->kv_cache_blocks.clear();
          req->kv_cache_blocks.resize(tp_num_);
          KLLM_LOG_WARNING << fmt::format("Split fuse disabled due to allocation failure for request ID {}",
                                          req->req_id);
        }
      }
      batch_scheduler_config_.split_fuse_token_num = 0;
      KLLM_LOG_WARNING << "Split fuse has been disabled.";
    }

    // If allocation failed, skip swapout, recompute request directly.
    if (allocate_block_succ) {
      // No more blocks, skip swap in and waiting launch.
      batch_state_->step_sched_finish = true;
      KLLM_LOG_DEBUG << "No more free blocks, skip swapped and waiting queue.";

      if (batch_scheduler_config_.preempt_mode == PreemptMode::SWAP) {
        KLLM_LOG_DEBUG << "running req " << req->req_id << " swapout async"
                       << ", current_block_num:" << req->kv_cache_blocks[0].size()
                       << ", current_token_size:" << req->output_tokens.size();

        // Merge all swapin request before swapout.
        if (!batch_state_->swapin_pending_requests_.empty()) {
          KLLM_LOG_DEBUG << "Pending swapin requests exists, merge it first.";
          MergePendingSwapinRequests(true, false);
        }
        RECORD_TRACE_EVENT_BEGIN("SO", TraceEventType::SwapOut, std::to_string(req->req_id), TRACE_THREAD_NAME_SWAP);
        size_t free_block_num = 0;
        size_t swapped_block_num = 0;
        status = cache_manager_->SwapoutRequestAsync(req->req_id, swapped_block_num, free_block_num);
        if (status.OK()) {
          batch_state_->swapout_pending_requests_[req->req_id] = req;
          batch_state_->running_queue.erase(it);

          total_needed_block_num -= step_block_num;
          continue;
        }
        KLLM_LOG_ERROR << "Swap out request error, recompute it. info: " << status.GetMessage();
      }
    }

    if (!status.OK() || batch_scheduler_config_.preempt_mode == PreemptMode::RECOMPUTE) {
      KLLM_LOG_DEBUG << "running req " << req->req_id << " recompute.";

      size_t freeable_block_num = 0;
      cache_manager_->GetRequestFreeableBlockNum(req->req_id, freeable_block_num);
      cache_manager_->DestroyFinishedRequest(req->req_id);

      // Add recomputed request to the begining of waiting queue.
      req->kv_cache_blocks.clear();
      req->kv_cache_blocks.resize(tp_num_);
      req->infer_stage = InferStage::STAGE_CONTEXT;
      req->step = 0;
      req->complete_output_tokens = req->output_tokens;
      batch_state_->waiting_queue.insert(batch_state_->waiting_queue.begin(), req);
      batch_state_->running_queue.erase(it);

      total_needed_block_num -= step_block_num;
      continue;
    }

    KLLM_LOG_DEBUG << "running req " << req->req_id << " should not arrive here.";
  }

  batch_state_->MergeRunningBufferQueue();
}

void ContinuousBatchingStrategy::ProcessSwappedQueue() {
  KLLM_LOG_DEBUG << "ProcessSwappedQueue invoked, swap queue size:" << batch_state_->swapped_queue.size()
                 << ", free block num:" << cache_manager_->GetUsableBlockNumber()
                 << ", future block num:" << cache_manager_->GetFutureBlockNumber();

  if (batch_scheduler_config_.preempt_mode != SWAP) {
    return;
  }

  // Merge pending swapout requests.
  Status status = MergePendingSwapoutRequests(false, true);
  if (!status.OK()) {
    KLLM_LOG_ERROR << "ProcessSwappedQueue error, info: " << status.GetMessage();
  }

  if (batch_state_->swapped_queue.empty()) {
    return;
  }

  if (batch_state_->step_sched_finish) {
    KLLM_LOG_DEBUG << "Skip swapped queue.";
    return;
  }

  size_t step_batch_size = batch_state_->running_queue.size();
  size_t step_token_num = batch_state_->running_queue.size();
  for (auto it = batch_state_->swapped_queue.begin(); it != batch_state_->swapped_queue.end();) {
    auto req = it->second;

    // Check timeout, no finished req in swapped queue.
    if (CheckRequestTimeout(req)) {
      KLLM_LOG_DEBUG << "req " << req->req_id << " timeout in swapped.";

      cache_manager_->DestroySwapedRequest(req->req_id);

      StopRequest(req, Status(RET_TIMEOUT, "running timeout."));
      it = batch_state_->swapped_queue.erase(it);
      continue;
    }

    // Check abort.
    if (req->aborted) {
      KLLM_LOG_DEBUG << "req " << req->req_id << " aborted in swapped.";

      cache_manager_->DestroySwapedRequest(req->req_id);

      StopRequest(req, Status(RET_TERMINATED, "client aborted."));
      it = batch_state_->swapped_queue.erase(it);
      continue;
    }

    size_t swapin_needed_block_num = 0;
    cache_manager_->GetRequestNeededBlockNum(req->req_id, swapin_needed_block_num);

    size_t swapin_block_threshold = std::ceil(step_batch_size * batch_scheduler_config_.swapin_block_threshold);

    size_t total_free_block_num = cache_manager_->GetUsableBlockNumber();
    size_t step_needed_block_num = cache_manager_->GetRequestStepBlockNumber(req->req_id, req->output_tokens.size());

    if (step_batch_size + 1 <= batch_scheduler_config_.max_batch_size &&
        step_token_num + 1 <= batch_scheduler_config_.max_step_tokens &&
        swapin_needed_block_num + step_needed_block_num + swapin_block_threshold <= total_free_block_num) {
      KLLM_LOG_DEBUG << "swapped req " << req->req_id << " swapin async"
                     << ", current_block_num:" << req->kv_cache_blocks[0].size()
                     << ", current_token_size:" << req->output_tokens.size();

      // Merge pending swapout requests before swap in.
      if (!batch_state_->swapout_pending_requests_.empty()) {
        KLLM_LOG_DEBUG << "Pending swapout requests exists, merge it first.";
        MergePendingSwapoutRequests(true, false);
      }
      RECORD_TRACE_EVENT_BEGIN("SI", TraceEventType::SwapIn, std::to_string(req->req_id), TRACE_THREAD_NAME_SWAP);
      status = cache_manager_->SwapinRequestAsync(req->req_id, swapin_needed_block_num, req->kv_cache_blocks);
      if (status.OK()) {
        step_batch_size += 1;
        step_token_num += 1;

        batch_state_->swapin_pending_requests_[req->req_id] = req;
        it = batch_state_->swapped_queue.erase(it);
        continue;
      }
      KLLM_LOG_ERROR << "Swap in request error, info: " << status.GetMessage();
      ++it;
    }

    // Swapped job still existed, skip launch waiting.
    batch_state_->step_sched_finish = true;
    KLLM_LOG_DEBUG << "Swapped queue not empty, skip waiting queue.";
    break;
  }
}

/**
 * Processes a request to determine the appropriate number of tokens to split or fuse based on the current
 * batching strategy configuration. This function adjusts the number of output tokens in the request to match
 * the calculated split or fuse token count, and updates the shared and unique block counts accordingly.
 *
 * The function aims to optimize the processing of requests by dynamically adjusting the number of tokens
 * to be processed together, based on the configured thresholds and the current state of the request and
 * batch scheduler.
 */
bool ContinuousBatchingStrategy::ProcessSplitFuseToken(std::shared_ptr<InferRequest> req, size_t &shared_block_num,
                                                       size_t &unique_block_num, size_t &shared_token_num,
                                                       size_t step_token_num, size_t decode_request_num) {
  size_t split_fuse_token_num = batch_scheduler_config_.split_fuse_token_num;
  size_t current_token_num = req->output_tokens.size();
  size_t tokens_to_split = 0;
  if (step_token_num < split_fuse_token_num) {
    tokens_to_split = std::min((current_token_num - shared_token_num + req->block_token_num - 1),
                               (split_fuse_token_num - step_token_num)) /
                      req->block_token_num * req->block_token_num;
  }

  if (split_fuse_token_num > 0 && decode_request_num != 0 && tokens_to_split < current_token_num - shared_token_num) {
    if (tokens_to_split == 0) {
      // End the scheduling of context decode.
      return true;
    }
    KLLM_LOG_DEBUG << "req " << req->req_id << " shared_block_num " << shared_block_num << " unique_block_num "
                   << unique_block_num << " shared_token_num " << shared_token_num << " current_token_num "
                   << current_token_num << " tokens_to_split " << tokens_to_split;
    req->output_tokens.resize(shared_token_num + tokens_to_split);
    cache_manager_->GetRequestPrefixBlockNumber(req->req_id, req->output_tokens, shared_block_num, unique_block_num,
                                                shared_token_num);
    // The `unique_block_num` is decremented here because, during split_fuse operations,
    // there is no need to pre-allocate a block for decoding purposes.
    unique_block_num--;
  }
  return false;
}

void ContinuousBatchingStrategy::ProcessWaitingQueue() {
  KLLM_LOG_DEBUG << "ProcessWaitingQueue invoked, waiting queue size:" << batch_state_->waiting_queue.size()
                 << ", free block num:" << cache_manager_->GetUsableBlockNumber()
                 << ", future block num:" << cache_manager_->GetFutureBlockNumber();

  if (batch_state_->waiting_queue.empty()) {
    return;
  }

  if (batch_state_->step_sched_finish) {
    KLLM_LOG_DEBUG << "Skip waiting queue.";
    return;
  }

  size_t total_logits_extra_length = 0;
  size_t decode_request_num = batch_state_->running_queue.size();

  size_t step_batch_size = batch_state_->running_queue.size();
  size_t step_token_num = batch_state_->running_queue.size();

  for (auto it = batch_state_->waiting_queue.begin(); it != batch_state_->waiting_queue.end();) {
    auto req = *it;
    CheckJumpForwardRequest(req);

    if (req->req_fsm != nullptr && CheckRequestFinish(req)) {
      cache_manager_->DestroyFinishedRequest(req->req_id);
      StopRequest(req, Status(RET_SUCCESS));
      it = batch_state_->waiting_queue.erase(it);
      continue;
    }
    ++it;
  }

  for (auto it = batch_state_->waiting_queue.begin(); it != batch_state_->waiting_queue.end();) {
    auto req = *it;
    req->output_tokens = req->complete_output_tokens;

    // When the logits_custom_length is greater than 0, the size of logits to be calculated is logits_custom_length.
    auto logits_extra_length = (req->logits_custom_length - 1);
    if (req->logits_custom_length > 0) {
      total_logits_extra_length += logits_extra_length;
    }

    // Check timeout, no finished req in waiting queue.
    if (CheckRequestTimeout(req)) {
      KLLM_LOG_DEBUG << "req " << req->req_id << " timeout in waiting.";

      StopRequest(req, Status(RET_TIMEOUT, "running timeout."));
      it = batch_state_->waiting_queue.erase(it);
      continue;
    }

    // Check abort.
    if (req->aborted) {
      KLLM_LOG_DEBUG << "req " << req->req_id << " aborted in waiting.";

      StopRequest(req, Status(RET_TERMINATED, "client aborted."));
      it = batch_state_->waiting_queue.erase(it);
      continue;
    }

    size_t shared_block_num = 0;
    size_t unique_block_num = 0;
    size_t shared_token_num = 0;
    cache_manager_->GetRequestPrefixBlockNumber(req->req_id, req->output_tokens, shared_block_num, unique_block_num,
                                                shared_token_num);
    if (ProcessSplitFuseToken(req, shared_block_num, unique_block_num, shared_token_num, step_token_num,
                              decode_request_num)) {
      break;
    }

    req->prefix_cache_len = shared_token_num;
    req->prefix_cache_blocks_number = shared_block_num;
    req->is_use_prefix_cache = shared_token_num > 0;

    if (req->is_use_prefix_cache) {
      REPORT_COUNTER(prefix_cache_hit_req_num, static_cast<size_t>(1));
      REPORT_COUNTER(prefix_cache_hit_token_num, shared_token_num);
      REPORT_COUNTER(prefix_cache_hit_block_num, shared_block_num);
    }

    size_t current_token_num = req->output_tokens.size();
    size_t launch_block_threshold = std::ceil(step_batch_size * batch_scheduler_config_.launch_block_threshold);

    // Get usable block number every time, the blocks matched by req are not reusable here.
    size_t total_free_block_num = cache_manager_->GetRequestUsableBlockNumber(req->req_id);
    if (step_batch_size + 1 + total_logits_extra_length <= batch_scheduler_config_.max_batch_size &&
        step_token_num + current_token_num <= batch_scheduler_config_.max_step_tokens &&
        unique_block_num + launch_block_threshold <= total_free_block_num) {
      Status status = cache_manager_->AllocateRequestBlocks(req->req_id, unique_block_num, req->kv_cache_blocks);
      KLLM_LOG_DEBUG << "waiting req " << req->req_id << " launch, prefix_len:" << req->prefix_cache_len
                     << ", share_block_num:" << req->prefix_cache_blocks_number
                     << ", unique_block_num:" << unique_block_num << ", free_block_num:" << total_free_block_num;
      if (status.OK()) {
        step_batch_size += 1;
        step_token_num += current_token_num;

        // if full matched, skip decode and put it to the end of decode list.
        if (shared_token_num == req->output_tokens.size()) {
          KLLM_LOG_DEBUG << "Full matched, skip prefill, req " << req->req_id;
          REPORT_COUNTER(full_prompt_matched_req_num, static_cast<size_t>(1));
          REPORT_COUNTER(full_prompt_matched_block_num, shared_block_num);
          REPORT_METRIC(time_to_first_token_ms, ProfileTimer::GetCurrentTimeInMs() - req->timestamp_in_ms);

          req->infer_stage = InferStage::STATE_DECODE;
          req->prefix_cache_len = 0;
          req->prefix_cache_blocks_number = 0;
          req->is_use_prefix_cache = false;
          batch_state_->running_queue.insert(batch_state_->running_queue.begin() + decode_request_num, req);
        } else {
          batch_state_->running_queue.push_back(req);
        }

        it = batch_state_->waiting_queue.erase(it);
        if (batch_scheduler_config_.split_fuse_token_num > 0 &&
            step_token_num >= batch_scheduler_config_.split_fuse_token_num) {
          break;
        }
        cache_manager_->UpdateFlexibleCache(req->req_id, req->output_tokens, shared_token_num,
                                            req->flexible_cached_copy_tasks);
        continue;
      }
      KLLM_LOG_ERROR << "Alllocate blocks error, info: " << status.GetMessage();
      KLLM_LOG_DEBUG << "Waiting all pending swapout requests done, and stay in waiting.";
      MergePendingSwapoutRequests(true, false);
    }
    break;
  }
}

Status ContinuousBatchingStrategy::MergePendingSwapinRequests(bool blocking, bool early_stop) {
  KLLM_LOG_DEBUG << "MergePendingSwapinRequests invoked.";
  // Wait all requests done.
  size_t swapin_left_req_num = 0;
  do {
    std::vector<int64_t> swapin_req_ids;
    Status status = cache_manager_->WaitSwapinRequests(swapin_req_ids, swapin_left_req_num, blocking);
    if (!status.OK()) {
      return status;
    }

    KLLM_LOG_DEBUG << "finished swapin request size:" << swapin_req_ids.size();
    for (int64_t req_id : swapin_req_ids) {
      auto it = batch_state_->swapin_pending_requests_.find(req_id);
      if (it == batch_state_->swapin_pending_requests_.end()) {
        KLLM_LOG_ERROR << "The cached swapin req " << req_id << " is not found in pending queue.";
        continue;
      }

      auto &req = it->second;
      status = cache_manager_->MergeSwapinRequest(req->req_id, req->kv_cache_blocks);
      if (!status.OK()) {
        return status;
      }

      RECORD_TRACE_EVENT_END("SI", TraceEventType::SwapIn, std::to_string(req->req_id), TRACE_THREAD_NAME_SWAP);
      KLLM_LOG_DEBUG << "MergePendingSwapinRequests swap in req " << req->req_id
                     << ", current_block_num:" << req->kv_cache_blocks[0].size()
                     << ", current_token_size:" << req->output_tokens.size();
      batch_state_->running_buffer_queue.push_back(req);
      batch_state_->swapin_pending_requests_.erase(it);
    }
  } while (!early_stop && swapin_left_req_num > 0);

  return Status();
}

Status ContinuousBatchingStrategy::MergePendingSwapoutRequests(bool blocking, bool early_stop) {
  KLLM_LOG_DEBUG << "MergePendingSwapoutRequests invoked.";

  // Wait all requests done.
  size_t swapout_left_req_num = 0;
  do {
    std::vector<int64_t> swapout_req_ids;
    Status status = cache_manager_->WaitSwapoutRequests(swapout_req_ids, swapout_left_req_num, blocking);
    if (!status.OK()) {
      return status;
    }

    for (int64_t req_id : swapout_req_ids) {
      auto it = batch_state_->swapout_pending_requests_.find(req_id);
      if (it == batch_state_->swapout_pending_requests_.end()) {
        KLLM_LOG_ERROR << "The cached swapout req " << req_id << " is not found in pending queue.";
        continue;
      }

      auto &req = it->second;
      status = cache_manager_->MergeSwapoutRequest(req->req_id);
      if (!status.OK()) {
        return status;
      }

      RECORD_TRACE_EVENT_END("SO", TraceEventType::SwapOut, std::to_string(req->req_id), TRACE_THREAD_NAME_SWAP);
      KLLM_LOG_DEBUG << "MergePendingSwapinRequests swapout req " << req->req_id
                     << ", current_block_num:" << req->kv_cache_blocks[0].size()
                     << ", current_token_size:" << req->output_tokens.size();
      batch_state_->swapped_queue[req->req_id] = req;
      batch_state_->swapout_pending_requests_.erase(it);
    }
  } while (!early_stop && swapout_left_req_num > 0);

  return Status();
}

void ContinuousBatchingStrategy::Schedule() {
  batch_state_->ResetInfoBeforeSchedule();
  batch_state_->MergeWaitingBufferQueue();
  ProcessRunningQueue();
  ProcessSwappedQueue();
  ProcessWaitingQueue();

  REPORT_COUNTER(batch_scheduler_pending_swapin_size, batch_state_->swapin_pending_requests_.size());
  REPORT_COUNTER(batch_scheduler_pending_swapout_size, batch_state_->swapout_pending_requests_.size());
}

}  // namespace ksana_llm
