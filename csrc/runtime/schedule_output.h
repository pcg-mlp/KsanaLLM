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
  }

  bool IsLaunchable() {
    for (auto req : running_reqs) {
      if (req->IsStopped()) {
        continue;
      }
      if (req->HasInflightTask()) {
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

  // Try to launch schedule output if all running reqs do not have infight task.
  void LaunchScheduleOutput() {
    if (!IsLaunchable()) {
      return;
    }
    for (const auto& req : running_reqs) {
      if (req->IsStopped()) continue;  // request may be finished in async mode.
      req->LaunchPlanningTask();
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
