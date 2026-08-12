/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include "batch_scheduler/batch_scheduler_interface.h"

namespace ksana_llm {

// 调度结果结构体
struct ScheduleResult {
  std::shared_ptr<ScheduleOutput> schedule_output;
  std::unordered_map<int, std::shared_ptr<InferRequest>> running_reqs;
  GenerationOutputGroup generation_output_group;

  std::vector<ScheduleOutput *> outputs;  // keep ref for async info clearing
};

class ScheduleProcessorInterface {
 public:
  virtual ~ScheduleProcessorInterface() = default;

  // 初始化处理器
  virtual void Initialize(std::shared_ptr<BatchSchedulerInterface> batch_scheduler) = 0;

  virtual Status AddInferRequest(std::vector<std::shared_ptr<InferRequest>> &infer_request_group) = 0;

  virtual std::shared_ptr<ScheduleResult> GetNextScheduleResult() = 0;

  virtual void UpdateWithGenerationResult(const GenerationOutputGroup &generation_output) = 0;

  virtual void NotifyNextPPMultiBatch(size_t multi_batch_id, const std::shared_ptr<ScheduleOutput> &schedule_output) {}

  // 外部调用：禁止/恢复 InferRequest 被调度线程修改。
  virtual void Lock() = 0;
  virtual void Unlock() = 0;

  virtual void Start() = 0;
  virtual void Stop() = 0;

 protected:
  std::shared_ptr<BatchSchedulerInterface> batch_scheduler_;
};

}  // namespace ksana_llm
