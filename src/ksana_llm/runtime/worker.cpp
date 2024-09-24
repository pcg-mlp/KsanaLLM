/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/runtime/worker.h"

#include <memory>

#include "ksana_llm/profiler/reporter.h"
#include "ksana_llm/runtime/threadpool.h"
#include "ksana_llm/utils/device_utils.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

Status Worker::Forward(std::shared_ptr<BaseModel> model, std::shared_ptr<BaseWeight> weight, InferStage stage,
                       std::vector<ForwardRequest>& forward_reqs) {
  // TODO(karlluo): confirm redundant usage
  SetDevice(rank_);
  opentelemetry::trace::StartSpanOptions options;

  switch (stage) {
    case InferStage::STAGE_CONTEXT:
      KLLM_LOG_DEBUG << "ContextDecode infer on work_id: " << rank_;
      {
        auto span = REPORT_TRACE(worker_context_decode, options);
        opentelemetry::trace::Scope scope(span);
        model->ContextDecode(weight, forward_reqs);
        span->End();
        for (auto it = forward_reqs.begin(); it != forward_reqs.end(); ++it) {
          auto& forward_req = *it;
          options.parent = forward_req.span_context;
          REPORT_METRIC(time_to_first_token_ms, ProfileTimer::GetCurrentTimeInMs() - forward_req.timestamp_in_ms,
                        forward_req.req_ctx);
        }
      }
      break;
    case InferStage::STATE_DECODE:
      KLLM_LOG_DEBUG << "Decode infer on work_id: " << rank_;
      {
        auto start = ProfileTimer::GetCurrentTimeInMs();
        model->Decode(weight, forward_reqs);
        auto duration = ProfileTimer::GetCurrentTimeInMs() - start;

        REPORT_METRIC(time_per_output_token_ms, duration / static_cast<double>(forward_reqs.size()));

        break;
      }
    default:
      KLLM_THROW(fmt::format("Invalid infer stage: {}. Valid stages include STAGE_CONTEXT and STATE_DECODE", stage));
  }

  return Status();
}

std::future<Status> Worker::ForwardAsync(std::shared_ptr<BaseModel> model, std::shared_ptr<BaseWeight> weight,
                                         InferStage stage, std::vector<ForwardRequest>& forward_reqs) {
  return threadpool_->Submit([=, &forward_reqs]() -> Status { return Forward(model, weight, stage, forward_reqs); });
}

Status Worker::Sampling(std::shared_ptr<Sampler> sampler, std::vector<SamplingRequest>& sampling_reqs) {
  // TODO(karlluo): confirm redundant usage
  SetDevice(rank_);
  sampler->Sampling(sampling_reqs, context_->GetComputeStreams()[rank_]);
  return Status();
}

std::future<Status> Worker::SamplingAsync(std::shared_ptr<Sampler> sampler,
                                          std::vector<SamplingRequest>& sampling_reqs) {
  return threadpool_->Submit([=, &sampling_reqs]() -> Status { return Sampling(sampler, sampling_reqs); });
}

WorkerGroup::WorkerGroup(size_t tensor_parallel_size, size_t pipeline_parallel_size, std::shared_ptr<Context> context)
    : tensor_parallel_size_(tensor_parallel_size), pipeline_parallel_size_(pipeline_parallel_size) {
  threadpool_ = std::make_shared<ThreadPool>(tensor_parallel_size_ * pipeline_parallel_size_);
  threadpool_->Start();

  workers_.resize(tensor_parallel_size_ * pipeline_parallel_size_);
  for (size_t worker_id = 0; worker_id < tensor_parallel_size_; ++worker_id) {
    workers_[worker_id].reset(new Worker(worker_id, threadpool_, context));
  }
}

WorkerGroup::~WorkerGroup() { threadpool_->Stop(); }

std::shared_ptr<Worker> WorkerGroup::GetWorker(int rank) {
  if (rank < 0 || rank >= static_cast<int>(workers_.size())) {
    KLLM_LOG_FATAL << "The worker rank " << rank << " exceed worker size " << workers_.size();
  }
  return workers_[rank];
}

}  // namespace ksana_llm
