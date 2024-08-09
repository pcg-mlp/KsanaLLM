/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/batch_scheduler/strategy/auto_batching.h"

#include <iterator>
#include <vector>

#ifdef ENABLE_ACL
#  include "ksana_llm/utils/ascend/acl_utils.h"
#endif

namespace ksana_llm {

AutoBatchingStrategy::AutoBatchingStrategy(const BatchSchedulerConfig &batch_scheduler_config, int tp_num,
                                           std::shared_ptr<BatchState> batch_state)
    : BaseScheduleStrategy(batch_scheduler_config, tp_num, batch_state) {}

bool AutoBatchingStrategy::CheckBatchFinished() {
  for (auto it = batch_state_->running_queue.begin(); it != batch_state_->running_queue.end(); ++it) {
    auto &req = *it;
    if (req->infer_stage != InferStage::STATE_DECODE) {
      return false;
    }

    if (req->output_tokens.size() <= req->input_tokens.size()) {
      return false;
    }

    std::vector<int> &stop_token_ids = req->sampling_config.stop_token_ids;
    if (std::find(stop_token_ids.begin(), stop_token_ids.end(), req->output_tokens.back()) == stop_token_ids.end() &&
        ((req->sampling_config.max_new_tokens <= 0 ||
          req->output_tokens.size() < req->input_tokens.size() + req->sampling_config.max_new_tokens) &&
         req->output_tokens.size() < batch_scheduler_config_.max_token_len)) {
      return false;
    }
  }
  return true;
}

bool AutoBatchingStrategy::CheckBatchTimeout() {
  for (auto it = batch_state_->running_queue.begin(); it != batch_state_->running_queue.end(); ++it) {
    auto &req = *it;
    if (batch_state_->schedule_time_in_ms < req->timestamp_in_ms + batch_scheduler_config_.waiting_timeout_in_ms) {
      return false;
    }
  }
  return true;
}

void AutoBatchingStrategy::PaddingRequests() {
  if (batch_state_->running_queue.empty()) {
    return;
  }

  size_t max_len = 0;
  for (auto it = batch_state_->running_queue.begin(); it != batch_state_->running_queue.end(); ++it) {
    auto &req = *it;
    size_t size = req->output_tokens.size();
    if (size > max_len) {
      max_len = size;
    }
  }

#ifdef ENABLE_ACL
  std::vector<int> &padded_token_size = GetPaddedTokenSize();
  padded_token_size.clear();
#endif

  for (auto it = batch_state_->running_queue.begin(); it != batch_state_->running_queue.end(); ++it) {
    auto &req = *it;

    size_t padded_num = max_len - req->output_tokens.size();
    if (padded_num > 0) {
      std::vector<int> tmp(padded_num, req->pad_id);
      tmp.insert(tmp.end(), req->output_tokens.begin(), req->output_tokens.end());
      req->padded_size = padded_num;
      req->output_tokens.swap(tmp);
    }

#ifdef ENABLE_ACL
    padded_token_size.push_back(padded_num);
#endif

    // For compatible with model input.
    for (int i = 0; i < tp_num_; ++i) {
      std::vector<int> blocks;
      GetBlockManager()->SetDeviceId(i);
      GetBlockManager()->AllocateBlocks(1, blocks);
      req->kv_cache_blocks[i].insert(req->kv_cache_blocks[i].end(), blocks.begin(), blocks.end());
    }
  }
}

void AutoBatchingStrategy::AdjustInferStage() {
  for (auto it = batch_state_->running_queue.begin(); it != batch_state_->running_queue.end(); ++it) {
    auto &req = *it;
    req->infer_stage = InferStage::STATE_DECODE;

    // Notify streaming iterator if needed.
    req->NotifyStep();
  }
}

void AutoBatchingStrategy::FinishBatchRequests(Status finish_status) {
  for (auto it = batch_state_->running_queue.begin(); it != batch_state_->running_queue.end(); ++it) {
    auto &req = *it;

    KLLM_LOG_DEBUG << "req " << req->req_id << " finished.";
    req->finish_status = finish_status;
    req->finished = true;
    req->Notify();
  }
  batch_state_->running_queue.clear();
}

void AutoBatchingStrategy::FetchBatchRequests() {
  if (batch_state_->waiting_queue.size() <= batch_scheduler_config_.max_batch_size / 2) {
    using namespace std::chrono_literals;
    std::this_thread::sleep_for(100ms);
    batch_state_->MergeWaitingBufferQueue();
  }

  size_t batch_size = std::min(batch_state_->waiting_queue.size(), batch_scheduler_config_.max_batch_size);
  batch_state_->running_queue.insert(batch_state_->running_queue.end(), batch_state_->waiting_queue.begin(),
                                     batch_state_->waiting_queue.begin() + batch_size);
  batch_state_->waiting_queue.erase(batch_state_->waiting_queue.begin(),
                                    batch_state_->waiting_queue.begin() + batch_size);
}

void AutoBatchingStrategy::Schedule() {
  batch_state_->ResetInfoBeforeSchedule();

  if (!batch_state_->running_queue.empty()) {
    AdjustInferStage();

    if (CheckBatchTimeout()) {
      FinishBatchRequests(Status(RET_TIMEOUT, "running timeout."));
    } else if (CheckBatchFinished()) {
      FinishBatchRequests(Status(RET_SUCCESS));
    } else {
      return;
    }
  }

  batch_state_->MergeWaitingBufferQueue();
  if (!batch_state_->waiting_queue.empty()) {
    FetchBatchRequests();
    PaddingRequests();
  }
}

}  // namespace ksana_llm
