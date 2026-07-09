/* Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * Round-robin workload balancer for event-driven scheduler.
 * ==============================================================================*/

#include "basic_workload_balancer.h"

#include <algorithm>
#include <numeric>

#include "utils/logger.h"

namespace ksana_llm {

BasicWorkloadBalancer::BasicWorkloadBalancer(size_t dp_num, size_t max_batch_size, size_t max_decode_tokens_per_req,
                                             size_t max_step_token_num)
    : dp_num_(dp_num),
      max_batch_size_(max_batch_size),
      dp_max_logits_num_(max_batch_size * max_decode_tokens_per_req),
      max_step_token_num_(max_step_token_num),
      inflight_dp_loads_(dp_num, 0) {
  KLLM_CHECK_WITH_INFO(dp_num_ > 0, "BasicWorkloadBalancer requires dp_num > 0.");
  KLLM_CHECK_WITH_INFO(max_batch_size > 0, "BasicWorkloadBalancer requires max_batch_size > 0.");
}

void BasicWorkloadBalancer::AddInferRequests(std::vector<InferRequestPtr>& requests, const ScheduleContext& context) {
  if (requests.empty()) {
    return;
  }

  if (dp_num_ == 1) {
    for (auto& req : requests) {
      if (req != nullptr) {
        req->attn_dp_group_id = 0;
      }
    }
    return;
  }

  std::lock_guard<std::mutex> guard(mu_);
  std::vector<size_t> dp_req_loads(dp_num_, 0);
  // Sum across ALL PP batches: DP assignment happens before PP batch assignment,
  // so we need the global per-DP load picture to balance correctly.
  size_t num_pp_batches = context.GetNumPpBatches();
  for (size_t dp_idx = 0; dp_idx < dp_num_; ++dp_idx) {
    for (size_t bid = 0; bid < num_pp_batches; ++bid) {
      const DpScheduleState& dp_state = context.GetDpScheduleState(bid, dp_idx);
      dp_req_loads[dp_idx] +=
          dp_state.preprocessing_reqs.size() + dp_state.prefilling_reqs.size() + dp_state.decoding_reqs.size();
    }
  }

  const auto get_rr_distance = [this](size_t dp_idx) { return (dp_idx + dp_num_ - next_dp_idx_) % dp_num_; };

  for (auto& req : requests) {
    if (req == nullptr) {
      continue;
    }
    size_t best_dp_idx = 0;
    bool has_best = false;
    for (size_t dp_idx = 0; dp_idx < dp_num_; ++dp_idx) {
      if (!has_best) {
        best_dp_idx = dp_idx;
        has_best = true;
        continue;
      }
      if (dp_req_loads[dp_idx] < dp_req_loads[best_dp_idx]) {
        best_dp_idx = dp_idx;
        continue;
      }
      if (dp_req_loads[dp_idx] == dp_req_loads[best_dp_idx] && get_rr_distance(dp_idx) < get_rr_distance(best_dp_idx)) {
        best_dp_idx = dp_idx;
      }
    }

    req->attn_dp_group_id = static_cast<uint32_t>(best_dp_idx);
    ++dp_req_loads[best_dp_idx];
    next_dp_idx_ = (best_dp_idx + 1) % dp_num_;
  }
}

std::vector<WorkloadQuota> BasicWorkloadBalancer::GetDpSchedulingOrderAndQuota(const PlanningBatch& batch,
                                                                               const ScheduleContext&, InferStage) {
  const size_t dp_num = batch.dp_batches.size();
  if (dp_num == 0) {
    return {};
  }

  if (dp_num == 1) {
    std::vector<WorkloadQuota> single_dp_order;
    single_dp_order.push_back({0, static_cast<int>(max_step_token_num_)});
    return single_dp_order;
  }

  std::lock_guard<std::mutex> guard(mu_);
  const auto get_rr_distance = [this, dp_num](size_t dp_idx) {
    return (dp_idx + dp_num - (next_dp_idx_ % dp_num)) % dp_num;
  };

  std::vector<size_t> sorted_dp_indices(dp_num, 0);
  std::iota(sorted_dp_indices.begin(), sorted_dp_indices.end(), 0);
  std::stable_sort(sorted_dp_indices.begin(), sorted_dp_indices.end(),
                   [&batch, &get_rr_distance](size_t lhs, size_t rhs) {
                     const size_t lhs_load = batch.dp_batches[lhs].running_step_token_num;
                     const size_t rhs_load = batch.dp_batches[rhs].running_step_token_num;
                     if (lhs_load != rhs_load) {
                       return lhs_load < rhs_load;
                     }
                     return get_rr_distance(lhs) < get_rr_distance(rhs);
                   });

  // Round 1: conservative per-dp quota so that head dp does not consume all global step-token budget.
  // Round 2: larger per-dp quota to improve utilization after each dp gets a chance.
  //
  // EDS backend must aggregate has_schedulable_req / has_scheduled_req across all quotas of the same dp
  // before emitting recompute/swapout, otherwise round-1 rejection can skip the round-2 fallback.
  const size_t small_quota = std::max<size_t>(1, max_step_token_num_ / (dp_num * 2));
  const size_t large_quota = std::max<size_t>(1, max_step_token_num_ / dp_num);

  std::vector<WorkloadQuota> dp_scheduling_order;
  dp_scheduling_order.reserve(dp_num * 2);
  for (size_t dp_idx : sorted_dp_indices) {
    dp_scheduling_order.push_back({dp_idx, static_cast<int>(small_quota)});
  }
  for (size_t dp_idx : sorted_dp_indices) {
    dp_scheduling_order.push_back({dp_idx, static_cast<int>(large_quota)});
  }

  next_dp_idx_ = (sorted_dp_indices.front() + 1) % dp_num;
  return dp_scheduling_order;
}

void BasicWorkloadBalancer::RecordInflightBatch(const PlanningBatch& batch, const ScheduleContext&) { return; }

bool BasicWorkloadBalancer::IsRequestAcceptable(const InferRequestPtr& req, const PlanningBatch& batch,
                                                const WorkloadQuota& workload_quota, size_t seq_token_num,
                                                size_t per_dp_step_token_num) const {
  const size_t dp_idx = workload_quota.dp_idx;
  KLLM_CHECK(req != nullptr);
  KLLM_CHECK(dp_idx < batch.dp_batches.size());
  const DpBatch& dp_batch = batch.dp_batches[dp_idx];
  const size_t step_token_limit =
      workload_quota.step_token_limit > 0 ? static_cast<size_t>(workload_quota.step_token_limit) : max_step_token_num_;
  if (batch.GetRunningReqNum() >= max_batch_size_) {
    return false;
  }
  if (dp_batch.running_step_logits_num + req->sampling_token_num > dp_max_logits_num_) {
    return false;
  }
  // per-dp quota 是 round-1/round-2 的公平分配 (max_step_token_num/dp), 用来防止 head dp 在多请求
  // 竞争时吃光全局预算 —— 它约束的是"多个请求挤一个 dp 时各自的份额", 不该约束"单个请求能多大"。
  // 因此仅在该 dp 本步已有请求 (running_step_token_num > 0) 时才用它限制新请求; 当 dp 空时, 这个
  // 请求会单独成批, 只受下面的全局 max_step_token_num (kernel/显存单步上限) 约束。否则 seq >
  // max_step_token_num/dp 的单个大请求会被 per-dp quota 死拒 → backend recompute 活锁
  // (standalone 无逃逸口直接死锁)。DEFAULT(ContinuousBatching) 调度器用全局共享预算 + ticktok
  // 轮转, 单请求本就只受全局上限; 这里让 EDS 在"单请求成批"场景与之对齐。
  if (dp_batch.running_step_token_num > 0 &&
      dp_batch.running_step_token_num + per_dp_step_token_num > step_token_limit) {
    return false;
  }
  // 全局单步 token 上限 (MLA prefix_kv_buffer 等 kernel/显存约束)。prefill 传整条 seq (含命中
  // prefix), decode 传 per-step query。单个请求 ≤ max_token_len ≤ max_step_token_num (见
  // schedule_config_parser 的重置保证), 故 dp 空时单请求成批总能通过这一关。
  if (batch.running_step_token_num + seq_token_num > max_step_token_num_) {
    return false;
  }
  return true;
}

}  // namespace ksana_llm
