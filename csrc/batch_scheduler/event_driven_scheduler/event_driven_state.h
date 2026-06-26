/* Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * 事件驱动调度器：核心状态与规划数据结构，与设计文档 3.1 ~ 3.5 节对应。
 * ==============================================================================*/
#pragma once

#include <cstdint>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "runtime/infer_request.h"
#include "runtime/schedule_output.h"

namespace ksana_llm {

// 当前 DP 分组内计划进行的调度操作，主要是正常forward之外的调度操作。
struct DpScheduleOpsInPlan {
  std::unordered_set<int64_t> stop_req_ids;
  std::unordered_set<int64_t> recompute_req_ids;
};

// 当前 DP 分组内需要通知后续node的调度操作，对齐ScheduleOutput中的内容。
struct DpScheduleOpsForNodes {
  std::vector<int64_t> stopped_req_ids;     // notify all nodes to stop requests
  std::vector<int64_t> recomputed_req_ids;  // notify all nodes to recompute requests

  void Reset() {
    stopped_req_ids.clear();
    recomputed_req_ids.clear();
  }

  std::string ToString() const {
    std::ostringstream oss;
    oss << "DpScheduleOpsForNodes: ";
    if (!stopped_req_ids.empty()) {
      oss << " stopped_req_ids=" << Vector2Str(stopped_req_ids);
    }
    if (!recomputed_req_ids.empty()) {
      oss << ", recomputed_req_ids=" << Vector2Str(recomputed_req_ids);
    }
    return oss.str();
  }
};

// 单个 DP 分组内的请求与运行情况（设计文档 3.1）
struct DpScheduleState {
  std::unordered_map<int64_t, std::shared_ptr<InferRequest>> preprocessing_reqs;
  std::vector<std::shared_ptr<InferRequest>> prefilling_reqs;
  std::vector<std::shared_ptr<InferRequest>> decoding_reqs;

  DpScheduleOpsInPlan in_plan_schedule_ops;
  DpScheduleOpsForNodes schedule_ops_for_nodes;
};

// 调度器全局状态（设计文档 3.2）
struct ScheduleState {
  // 2D matrix: schedule_states_[multi_batch_id][dp_idx].
  // Each multi_batch_id has its own independent set of DpScheduleState entries.
  // When max_pp_batch_num == 1, schedule_states_[0][dp_idx] is equivalent to old schedule_states_[dp_idx].
  std::vector<std::vector<DpScheduleState>> schedule_states;

  // Global request index: req_id -> InferRequest.
  // Used to locate request pointer when only req_id is available (e.g. async hook next_steps).
  std::unordered_map<int64_t, std::shared_ptr<InferRequest>> req_id_to_req_ptr;

  // Map from req_id to PP batch index. Used by event handlers to locate which batch a request belongs to.
  // Invariant: if req_id appears in any schedule_states[*][*] queue, req_to_batch_idx MUST contain it.
  // Immutable after assignment: a request MUST NOT be moved to a different batch once assigned.
  // This prevents state interference between independently-scheduled PP batches.
  std::unordered_map<int64_t, size_t> req_to_batch_idx;
};

// 当前在一个 DP 内已规划的 Batch（设计文档 3.3）
struct DpBatch {
  std::unordered_map<int64_t, std::shared_ptr<InferRequest>> running_reqs;
  size_t running_step_token_num = 0;
  size_t running_step_logits_num = 0;

  // Reset planning state for reuse.
  void Reset() {
    running_reqs.clear();
    running_step_token_num = 0;
    running_step_logits_num = 0;
  }

  // Check whether a request is already in this batch.
  bool IsInBatch(int64_t req_id) const { return running_reqs.find(req_id) != running_reqs.end(); }

 private:
  // Add a request into running_reqs and mark it as planning task.
  // This helper is only called by PlanningBatch to keep two-level counters consistent.
  void AddRunningReq(const std::shared_ptr<InferRequest>& req) {
    KLLM_CHECK(req != nullptr);
    KLLM_CHECK(!IsInBatch(req->req_id));
    running_step_token_num += req->GetPlanningQueryLen();
    running_step_logits_num += req->GetPlanningSamplingTokenNum();
    req->SetPlanningTask();
    running_reqs.emplace(req->req_id, req);
  }

  friend struct PlanningBatch;
};

// 当前已规划的一个 Batch，可能跨多 DP（设计文档 3.4）
struct PlanningBatch {
  int batch_idx = 0;
  // One entry per dp_idx. The vector is indexed directly by dp_idx.
  std::vector<DpBatch> dp_batches;
  size_t running_req_num = 0;
  size_t running_step_token_num = 0;
  size_t running_step_logits_num = 0;

  // Reset planning state for reuse.
  void Reset() {
    running_req_num = 0;
    running_step_token_num = 0;
    running_step_logits_num = 0;
    for (auto& dp_batch : dp_batches) {
      dp_batch.Reset();
    }
  }

  // Add one running request into target dp batch and aggregate workload counters.
  void AddRunningReq(size_t dp_idx, const std::shared_ptr<InferRequest>& req) {
    KLLM_CHECK(dp_idx < dp_batches.size());
    KLLM_CHECK(req != nullptr);
    KLLM_CHECK(!dp_batches[dp_idx].IsInBatch(req->req_id));
    ++running_req_num;
    running_step_token_num += req->GetPlanningQueryLen();
    running_step_logits_num += req->GetPlanningSamplingTokenNum();
    dp_batches[dp_idx].AddRunningReq(req);
  }

  // Remove request from target dp batch if exists and update aggregated counters.
  bool RemoveRunningReq(size_t dp_idx, int64_t req_id) {
    KLLM_CHECK(dp_idx < dp_batches.size());
    auto& dp_batch = dp_batches[dp_idx];
    auto it = dp_batch.running_reqs.find(req_id);
    if (it == dp_batch.running_reqs.end()) {
      return false;
    }
    const auto& req = it->second;
    KLLM_CHECK(req != nullptr);
    KLLM_CHECK(running_req_num > 0);
    --running_req_num;
    // Do not recover running_step_token_num and running_step_logits_num because planning workload is cleared in
    // InferRequest. Will casuse some resource waste. Only happens when request stopped after planned.
    dp_batch.running_reqs.erase(it);
    return true;
  }

  // Get total running request count across all dp batches.
  size_t GetRunningReqNum() const { return running_req_num; }

  // 快速可 launch 判定，不拷贝 running_reqs 到 ScheduleOutput；语义同 ScheduleOutput::IsLaunchable(max_depth)。
  // max_depth=1（慢路径）: 任一 req 有 inflight 即不可 launch；
  // max_depth=2（build-ahead）: 允许 inflight_count_<2 的 req 继续 launch。
  // EventDrivenScheduler 尚未接入 build-ahead，保留默认值 max_depth=1 即可向后兼容。
  bool IsLaunchable(size_t max_depth = 1) const {
    if (running_req_num == 0) {
      return false;
    }
    for (const auto& dp_batch : dp_batches) {
      for (const auto& r : dp_batch.running_reqs) {
        const std::shared_ptr<InferRequest>& req = r.second;
        KLLM_CHECK(req != nullptr);
        if (req->IsStopped()) {
          continue;
        }
        if (!req->HasInflightCapacity(max_depth)) {
          return false;
        }
      }
    }
    return true;
  }
};

}  // namespace ksana_llm
