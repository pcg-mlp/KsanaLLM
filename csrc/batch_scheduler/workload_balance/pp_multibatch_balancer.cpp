/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "batch_scheduler/workload_balance/pp_multibatch_balancer.h"

#include <algorithm>
#include <limits>

namespace ksana_llm {

PPMultibatchWorkloadBalancer::PPMultibatchWorkloadBalancer(const BatchSchedulerConfig& batch_scheduler_config)
    : pp_multibatch_wb_strategy_(batch_scheduler_config.pp_multibatch_wb_strategy),
      step_token_num_(batch_scheduler_config.max_step_token_num),
      split_fuse_token_num_(batch_scheduler_config.split_fuse_token_num),
      step_token_num_bound_(static_cast<size_t>(step_token_num_ * pp_multibatch_wb_strategy_.threshold_factor)),
      max_draft_token_num_(batch_scheduler_config.mtp_step_num + batch_scheduler_config.ptp_step_num),
      workloads_(batch_scheduler_config.max_pp_batch_num),
      last_workloads_(batch_scheduler_config.max_pp_batch_num) {
  KLLM_CHECK_WITH_INFO(pp_multibatch_wb_strategy_.enable,
                       "PPMultibatchWorkloadBalancer constructed with pp_multibatch_wb_strategy.enable=false");
  KLLM_CHECK_WITH_INFO(batch_scheduler_config.max_pp_batch_num > 1,
                       "PPMultibatchWorkloadBalancer constructed with max_pp_batch_num=1");
  KLLM_CHECK_WITH_INFO(pp_multibatch_wb_strategy_.threshold_factor <= 1.f, "threshold_factor must <= 1.0");
  KLLM_LOG_INFO << "PPMultibatchWorkloadBalancer init with strategy.prefill_hold_iters: "
                << pp_multibatch_wb_strategy_.prefill_hold_iters
                << ", strategy.threshold_factor: " << pp_multibatch_wb_strategy_.threshold_factor
                << ", step_token_num=" << step_token_num_ << ", split_fuse_token_num=" << split_fuse_token_num_
                << ", step_token_num_bound=" << step_token_num_bound_
                << ", max_draft_token_num=" << max_draft_token_num_;
}

void PPMultibatchWorkloadBalancer::BalancePPMultiBatchReqs(size_t multi_batch_id,
                                                           std::vector<std::shared_ptr<InferRequest>>& waiting_reqs,
                                                           std::vector<std::shared_ptr<BatchState>>& batch_states) {
  if (!waiting_reqs.empty() || !held_waiting_reqs_.empty()) {
    DistributeWaitingReqs(multi_batch_id, waiting_reqs, batch_states);
    last_workloads_ = workloads_;
  } else {
    const size_t pp_batch_num = batch_states.size();
    last_workloads_.assign(pp_batch_num, 0);
    for (size_t mb : {multi_batch_id, (multi_batch_id + 1) % pp_batch_num}) {
      for (const auto& req : batch_states[mb]->decoding_queue) {
        last_workloads_[mb] += StepLoadOf(req.get()).first;
      }
    }
  }
}

void PPMultibatchWorkloadBalancer::ShiftChunkedReqs(size_t multi_batch_id,
                                                    std::vector<std::shared_ptr<BatchState>>& batch_states) {
  const size_t next_multi_batch_id = (multi_batch_id + 1) % batch_states.size();
  auto& src_batch = batch_states[multi_batch_id];
  auto& dst_batch = batch_states[next_multi_batch_id];

  // Extract chunked-prefill requests from src waiting_queue.
  std::vector<std::shared_ptr<InferRequest>> chunked_reqs;
  {
    std::lock_guard<std::mutex> src_guard(src_batch->queue_mutex);
    for (auto it = src_batch->waiting_queue.begin(); it != src_batch->waiting_queue.end();) {
      const auto& req = *it;
      KLLM_CHECK_WITH_INFO(!req->HasPlanningTask(), "req must not have any planning task");
      const auto& w = req->GetRemainingWorkload();
      KLLM_CHECK_WITH_INFO(w.prefill_token_num > 0, "req in waiting_queue must have pending prefill");
      const bool eligible = w.prefill_start_offset > 0 && req->kv_cached_token_num > 0;
      if (eligible) {
        KLLM_LOG_SCHEDULER << "Shift req " << req->req_id
                           << " remaining.prefill_start_offset=" << w.prefill_start_offset
                           << " remaining.prefill_token_num=" << w.prefill_token_num
                           << " kv_cached_token_num=" << req->kv_cached_token_num << " from batch " << multi_batch_id
                           << " to batch " << next_multi_batch_id;
        chunked_reqs.emplace_back(std::move(*it));
        it = src_batch->waiting_queue.erase(it);
      } else {
        ++it;
      }
    }
  }

  if (chunked_reqs.empty()) {
    return;
  }

  // Insert at the front of dst waiting_queue, preserving the original relative order
  // among shifted chunked requests so that older chunks still run earlier.
  {
    std::lock_guard<std::mutex> dst_guard(dst_batch->queue_mutex);
    dst_batch->waiting_queue.insert(dst_batch->waiting_queue.begin(), std::make_move_iterator(chunked_reqs.begin()),
                                    std::make_move_iterator(chunked_reqs.end()));
  }

  KLLM_LOG_SCHEDULER << "Shifted " << chunked_reqs.size() << " chunked req(s) from batch " << multi_batch_id
                     << " to batch " << next_multi_batch_id;
}

void PPMultibatchWorkloadBalancer::DistributeWaitingReqs(size_t multi_batch_id,
                                                         std::vector<std::shared_ptr<InferRequest>>& waiting_reqs,
                                                         std::vector<std::shared_ptr<BatchState>>& batch_states) {
  const size_t pp_batch_num = batch_states.size();
  // Candidate states are only cur/next
  const size_t cur = multi_batch_id;
  const size_t next = (multi_batch_id + 1) % pp_batch_num;

  // Preserve FIFO
  // Re-inject reqs that were held in the previous iter back into the global pool.
  if (!held_waiting_reqs_.empty()) {
    waiting_reqs.insert(waiting_reqs.begin(), std::make_move_iterator(held_waiting_reqs_.begin()),
                        std::make_move_iterator(held_waiting_reqs_.end()));
    held_waiting_reqs_.clear();
  }
  if (!batch_states[next]->waiting_queue.empty()) {
    auto& next_waiting_queue = batch_states[next]->waiting_queue;
    std::vector<std::shared_ptr<InferRequest>> non_chunked_reqs;
    for (auto it = next_waiting_queue.begin(); it != next_waiting_queue.end();) {
      const auto& req = *it;
      if (req->GetRemainingWorkload().prefill_start_offset > 0 && req->kv_cached_token_num > 0) {
        ++it;
      } else {
        non_chunked_reqs.emplace_back(std::move(*it));
        it = next_waiting_queue.erase(it);
      }
    }
    waiting_reqs.insert(waiting_reqs.begin(), std::make_move_iterator(non_chunked_reqs.begin()),
                        std::make_move_iterator(non_chunked_reqs.end()));
  }

  // Initialize workloads from decoding/waiting queues.
  workloads_.assign(pp_batch_num, 0);
  for (size_t mb : {cur, next}) {
    for (const auto& req : batch_states[mb]->decoding_queue) {
      workloads_[mb] += StepLoadOf(req.get()).first;
    }
    for (const auto& req : batch_states[mb]->waiting_queue) {
      workloads_[mb] += StepLoadOf(req.get()).first;
    }
  }

  if (workloads_[cur] >= step_token_num_bound_) {
    held_waiting_reqs_.insert(held_waiting_reqs_.end(), std::make_move_iterator(waiting_reqs.begin()),
                              std::make_move_iterator(waiting_reqs.end()));
    waiting_reqs.clear();
    return;
  }

  KLLM_LOG_SCHEDULER << "PP balance start: cur=" << cur << ", next=" << next
                     << ", waiting_reqs.size=" << waiting_reqs.size() << ", workloads[cur]=" << workloads_[cur]
                     << ", workloads[next]=" << workloads_[next];

  // Greedy assignment: distribute to the batch with minimum workload.
  std::vector<std::vector<std::shared_ptr<InferRequest>>> assign(pp_batch_num);
  size_t consumed = 0;
  for (const auto& req : waiting_reqs) {
    int chosen = -1;
    size_t best = std::numeric_limits<size_t>::max();
    for (size_t mb : {cur, next}) {
      if (workloads_[mb] >= step_token_num_) {
        continue;
      }
      if (workloads_[mb] < best) {
        best = workloads_[mb];
        chosen = static_cast<int>(mb);
      }
    }
    if (chosen < 0) {
      // Both candidates already saturated; stop trying further reqs.
      break;
    }

    const auto [cur_step, next_step] = StepLoadOf(req.get());
    workloads_[chosen] += cur_step;
    // Only chunked requests (prefill_start_offset > 0 && kv_cached_token_num > 0)
    // assigned to cur will be shifted to next by ShiftChunkedReqs; their next_step
    // therefore lands on next.
    const auto& w = req->GetRemainingWorkload();
    bool will_shift_to_next =
        (chosen == static_cast<int>(cur)) && (w.prefill_start_offset > 0) && (req->kv_cached_token_num > 0);
    if (will_shift_to_next) {
      workloads_[next] += next_step;
    }
    assign[chosen].push_back(req);
    consumed++;
  }

  // Starvation backstop: hold up to prefill_hold_iters iters, then force-commit current capacity-aware assignment.
  const size_t prev = (multi_batch_id + pp_batch_num - 1) % pp_batch_num;
  bool both_saturated = (workloads_[cur] >= step_token_num_bound_) &&
                        (workloads_[next] >= step_token_num_bound_ || last_workloads_[prev] >= step_token_num_bound_);

  bool force_commit = false;
  if (!both_saturated) {
    hold_iter_++;
    if (hold_iter_ < pp_multibatch_wb_strategy_.prefill_hold_iters) {
      KLLM_LOG_SCHEDULER << "PP balance hold: not both saturated. workloads[cur]=" << workloads_[cur]
                         << ", workloads[next]=" << workloads_[next] << ", hold_iter=" << hold_iter_;
      held_waiting_reqs_.insert(held_waiting_reqs_.end(), std::make_move_iterator(waiting_reqs.begin()),
                                std::make_move_iterator(waiting_reqs.end()));
      waiting_reqs.clear();
      return;
    }
    KLLM_LOG_SCHEDULER << "PP balance fired after " << hold_iter_ << " iters; force-committing current assignment.";
    force_commit = true;
  }
  hold_iter_ = 0;

  // Real assignment.
  if (consumed > 0) {
    for (size_t mb : {cur, next}) {
      if (assign[mb].empty()) {
        continue;
      }
      {
        std::lock_guard<std::mutex> guard(batch_states[mb]->queue_mutex);
        for (auto& req : assign[mb]) {
          batch_states[mb]->waiting_queue.push_back(req);
        }
      }
    }
    KLLM_LOG_SCHEDULER << "PP balance commit" << (force_commit ? " (forced)" : "") << ": consumed " << consumed << "/"
                       << waiting_reqs.size() << " reqs. workloads[cur]=" << workloads_[cur]
                       << ", workloads[next]=" << workloads_[next];
  }

  // Leftover reqs stay in waiting_reqs for next iter.
  waiting_reqs.erase(waiting_reqs.begin(), waiting_reqs.begin() + consumed);
  held_waiting_reqs_.insert(held_waiting_reqs_.end(), std::make_move_iterator(waiting_reqs.begin()),
                            std::make_move_iterator(waiting_reqs.end()));
  waiting_reqs.clear();
}

std::pair<size_t, size_t> PPMultibatchWorkloadBalancer::StepLoadOf(const InferRequest* req) {
  // Load for the *next* iter on this req.
  size_t cap = split_fuse_token_num_ > 0 ? split_fuse_token_num_ : std::numeric_limits<size_t>::max();
  const size_t decode_load = kStepGenerateTokenNum + max_draft_token_num_;
  const auto& w = req->GetRemainingWorkload();
  if (w.prefill_token_num > 0) {
    const size_t cur_step = std::min(w.prefill_token_num, cap);
    const size_t next_step =
        w.prefill_token_num == cur_step ? decode_load : std::min(w.prefill_token_num - cur_step, cap);
    return {cur_step, next_step};
  } else {
    return {decode_load, decode_load};
  }
}

}  // namespace ksana_llm
