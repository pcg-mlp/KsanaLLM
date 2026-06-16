/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <unordered_map>
#include <unordered_set>

#include "device/device_types.h"
#include "distributed/raw_packet.h"
#include "runtime/executor_infer_request.h"
#include "runtime/infer_request.h"
#include "utils/blocking_queue.h"
#include "utils/request.h"
#include "utils/status.h"

namespace ksana_llm {

struct SwapReqBlockInfo {
  std::vector<int> dev_block_ids;
  std::vector<int> host_block_ids;
};

// Launch 时刻的 req 字段快照, 用于消除 LaunchPlanningTask 与异步 IPC 序列化之间的竞争.
//
// 背景: max_depth=2 时 schedule N+1 可在 N 的 IPC 发出前 launch, N+1 会修改 req->forwarding_tokens 等字段;
//       若序列化仍读 live req->*, Executor 收到的 step N 数据会被 N+1 污染 (末位 placeholder/长度错位).
// 用法: LaunchScheduleOutput 在持有 schedule_mutex_ 时深拷贝; SerializeScheduleOutput 优先读 snapshot.
// slow path (max_depth=1) 虽无并发 launch, 同样走 snapshot 以保持路径统一且结果等价.
struct InferRequestSerializationSnapshot {
  std::vector<int> forwarding_tokens;
  size_t kv_cached_token_num = 0;
  size_t forwarding_tokens_draft_num = 0;
  size_t sampling_token_num = 0;
  std::vector<int> kv_cache_blocks;
  std::vector<int> swa_kv_cache_blocks;
  std::vector<size_t> swa_kv_cache_block_idx_offsets;
  int prefix_cache_len = 0;
  int step = 0;
  // patch_transfer 由 ModelDriver::SetInferRequestTransferMode 决策; 相邻 schedule_id 会覆写 req->patch_transfer,
  // 故序列化读 snapshot 而非 live 字段, 避免 prefill 首步误走 patch 路径.
  bool patch_transfer = false;
  bool patch_transfer_populated = false;
};

// The scheduler output of every step.
struct ScheduleOutput {
  // Make it empty again, but keep runing reqs, called only on master node.
  void ResetDataForWorkers() {
    finish_req_ids.clear();
    recompute_req_ids.clear();
    merged_swapout_req_ids.clear();
    merged_swapin_req_ids.clear();
    swapout_req_block_ids.clear();
    swapin_req_block_ids.clear();
    structured_generator_configs.clear();
    need_create_generator_req_ids.clear();
    req_serialization_snapshots.clear();
  }

  // 检查 batch 内每个活跃 req 是否还能再 push 一个 inflight task.
  // max_depth=1: 要求 inflight_count_==0 (slow path); max_depth=2: 允许 inflight_count_<=1 (fast path).
  bool IsLaunchable(size_t max_depth = 1) const {
    for (const auto& req : running_reqs) {
      if (req->IsStopped()) {
        continue;
      }
      if (!req->HasInflightCapacity(max_depth)) {
        return false;
      }
    }
    return true;
  }

  void SetPlanningTask() {
    for (auto req : running_reqs) {
      req->SetPlanningTask();
    }
  }

  // 将 planning_task 推入各 req 的 inflight 队列并构造 forwarding_tokens.
  // 若 req 已有在飞 step (HasInflightTask), 末位写 placeholder, 真实 token 由 Executor ReplaceKernel 回填;
  // 否则直接 append generated_tokens (slow path 或 fast path 首步).
  void LaunchScheduleOutput(size_t max_depth = 1) {
    if (!IsLaunchable(max_depth)) {
      return;
    }
    for (const auto& req : running_reqs) {
      if (req->IsStopped()) continue;  // request may be finished in async mode.
      const bool use_placeholder_for_last_token = req->HasInflightTask();
      req->LaunchPlanningTask(use_placeholder_for_last_token);
      // 在 schedule_mutex_ 保护下固化易竞态字段, 供后续异步 IPC 序列化使用 (见 InferRequestSerializationSnapshot).
      auto& snap = req_serialization_snapshots[req->req_id];
      snap.forwarding_tokens = req->forwarding_tokens;
      snap.kv_cached_token_num = req->kv_cached_token_num;
      snap.forwarding_tokens_draft_num = req->forwarding_tokens_draft_num;
      snap.sampling_token_num = req->sampling_token_num;
      snap.kv_cache_blocks = req->kv_cache_blocks;
      snap.swa_kv_cache_blocks = req->swa_kv_cache_blocks;
      snap.swa_kv_cache_block_idx_offsets = req->swa_kv_cache_block_idx_offsets;
      snap.prefix_cache_len = req->prefix_cache_len;
      snap.step = req->step;
    }
  }

  // Make it empty again, called only on worker node.
  void Clear() {
    ResetDataForWorkers();
    running_reqs.clear();
    executor_running_reqs.clear();
  }

  void ClearRunningReqs() {
    running_reqs.clear();
    executor_running_reqs.clear();
  }

  size_t schedule_id = DEFAULT_SCHEDULE_ID;

  // NOTE(karlluo): finished req ids, outer vector is for attention data parallelism.
  std::vector<std::vector<int64_t>> finish_req_ids;

  // recompute requests, should transfer the full struct.
  std::vector<std::vector<int64_t>> recompute_req_ids;

  // NOTE(karlluo): merged requests ids, outer vector is for attention data parallelism.
  std::vector<std::vector<int64_t>> merged_swapout_req_ids;
  std::vector<std::vector<int64_t>> merged_swapin_req_ids;

  // NOTE(karlluo): swapped requests ids, outer vector is for attention data parallelism.
  std::vector<std::unordered_map<int64_t, SwapReqBlockInfo>> swapout_req_block_ids;
  std::vector<std::unordered_map<int64_t, SwapReqBlockInfo>> swapin_req_block_ids;

  // running, for master node.
  std::vector<std::shared_ptr<InferRequest>> running_reqs;

  // running, for executor process.
  std::vector<std::shared_ptr<ExecutorInferRequest>> executor_running_reqs;

  // The incremental patch for running queue.
  std::vector<std::shared_ptr<InferRequestPatch>> running_req_patches;

  // Structured generator configs for creating generators on Executor side
  // Only transmitted on first inference or after preemption recovery
  std::unordered_map<int64_t, StructuredGeneratorConfig> structured_generator_configs;

  // Request IDs that need to create generators on Executor side
  std::unordered_set<int64_t> need_create_generator_req_ids;

  // req_id → Launch 时刻快照 (见 InferRequestSerializationSnapshot).
  // 仅 Engine 主进程填充; Executor 反序列化得到的 ScheduleOutput 中此 map 为空.
  std::unordered_map<int64_t, InferRequestSerializationSnapshot> req_serialization_snapshots;
};

struct ScheduleOutputGroup {
 public:
  size_t schedule_id = DEFAULT_SCHEDULE_ID;
  std::vector<ScheduleOutput*> outputs;

 public:
  explicit ScheduleOutputGroup(size_t dp_num = 1) : schedule_id(DEFAULT_SCHEDULE_ID) {
    outputs.resize(dp_num, nullptr);
  }

  size_t RunningSize() const {
    size_t size = 0;
    for (auto& output : outputs) {
      if (output == nullptr) {
        continue;
      }
      size += output->running_reqs.size();
    }
    return size;
  }
};

// Forward declare.
class ScheduleProcessorInterface;

class ScheduleOutputParser {
 public:
  // We just assume the data memory is large enough, and do not check it.
  static Status SerializeScheduleOutput(const ScheduleOutput* schedule_output, void* ptr, size_t* bytes);
  static Status DeserializeScheduleOutput(const void* ptr, ScheduleOutput* schedule_output);

  // Get the serialized byte of a ScheduleOutput object.
  static size_t GetSerializedSize(const ScheduleOutput* schedule_output);

  // Set schedule processor.
  static void SetScheduleProcessor(const std::shared_ptr<ScheduleProcessorInterface> schedule_processor);

 private:
  static std::shared_ptr<ScheduleProcessorInterface> schedule_processor_;
};

// An object pool for schedule output.
class ScheduleOutputPool {
 public:
  // Get a schedule output object.
  ScheduleOutput* GetFreeScheduleOutput();

  // Free the schedule output to object pool.
  Status FreeScheduleOutput(ScheduleOutput* schedule_output);

  // Put and get send buffer.
  void PutSendScheduleOutput(ScheduleOutput* schedule_output);
  Packet* GetSendSerializedPacket();

  // Put and get serialized packet.
  void PutRecvSerializedPacket(Packet* packet);
  Packet* GetRecvSerializedPacket();

  // All blocked queue will be returned immediately.
  Status Stop();

 private:
  BlockingQueue<ScheduleOutput*> free_queue_;

  // The serialized packet that ready to send.
  BlockingQueue<Packet*> serialized_send_queue_;

  // The serialized packet from master node.
  BlockingQueue<Packet*> serialized_recv_queue_;
};

struct GenerationOutputGroup {
  size_t multi_batch_id = DEFAULT_MULTI_BATCH_ID;

  std::vector<std::vector<std::shared_ptr<InferRequest>>> reqs;

  void Reset() { reqs.clear(); }

  void BuildFromScheduleOutputGroup(const ScheduleOutputGroup& schedule_output_group) {
    reqs.resize(schedule_output_group.outputs.size());
    for (size_t i = 0; i < schedule_output_group.outputs.size(); ++i) {
      auto& output = schedule_output_group.outputs[i];
      if (output == nullptr) {
        continue;
      }
      reqs[i] = output->running_reqs;
    }
  }
};

inline bool RemoveRequestFromQueue(std::vector<std::shared_ptr<InferRequest>>& req_queue,
                                   const std::shared_ptr<InferRequest>& req) {
  auto it = std::find(req_queue.begin(), req_queue.end(), req);
  if (it != req_queue.end()) {
    req_queue.erase(it);
    return true;
  }
  return false;
}

}  // namespace ksana_llm
