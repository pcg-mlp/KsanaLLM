/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <future>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ksana_llm/models/base/base_model.h"
#include "ksana_llm/runtime/context.h"
#include "ksana_llm/runtime/forward_request.h"
#include "ksana_llm/runtime/infer_stage.h"
#include "ksana_llm/runtime/sampling_request.h"
#include "ksana_llm/runtime/threadpool.h"
#include "ksana_llm/samplers/sampler.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/tensor.h"

namespace ksana_llm {

// The worker executed on every device.
class Worker {
 public:
  Worker(int rank, std::shared_ptr<ThreadPool> threadpool, std::shared_ptr<Context> context)
      : rank_(rank), threadpool_(threadpool), context_(context) {}

  ~Worker() {}

  // The async forward and sampling.
  std::future<Status> ForwardAsync(std::shared_ptr<BaseModel> model, std::shared_ptr<BaseWeight> weight,
                                   InferStage stage, std::vector<ForwardRequest>& forward_reqs);

  Status Forward(std::shared_ptr<BaseModel> model, std::shared_ptr<BaseWeight> weight, InferStage stage,
                 std::vector<ForwardRequest>& forward_reqs);

  std::future<Status> SamplingAsync(std::shared_ptr<Sampler> sampler, std::vector<SamplingRequest>& sampling_reqs);

  Status Sampling(std::shared_ptr<Sampler> sampler, std::vector<SamplingRequest>& sampling_reqs);

 private:
  // Current worker rank.
  int rank_;

  // Threadpool used to execute async task.
  std::shared_ptr<ThreadPool> threadpool_ = nullptr;

  // GPU related context
  std::shared_ptr<Context> context_ = nullptr;
};

// The worker group that used to manager multiple workers.
class WorkerGroup {
 public:
  WorkerGroup(size_t tensor_parallel_size, size_t pipeline_parallel_size, std::shared_ptr<Context> context);
  ~WorkerGroup();

  // Get worker of specified rank.
  std::shared_ptr<Worker> GetWorker(int rank);

 private:
  // The inner workers.
  std::vector<std::shared_ptr<Worker>> workers_;

  // The parallel size.
  size_t tensor_parallel_size_;
  size_t pipeline_parallel_size_;

  // The thread pool used for workers.
  std::shared_ptr<ThreadPool> threadpool_ = nullptr;
};

}  // namespace ksana_llm
