/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/runtime/worker.h"

#include <memory>

#include "ksana_llm/runtime/threadpool.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

Status Worker::Forward(std::shared_ptr<BaseModel> model, std::shared_ptr<BaseWeight> weight, InferStage stage,
                       std::vector<ForwardRequest>& forward_reqs) {
  // TODO(karlluo): confirm redundant usage
#ifdef ENABLE_CUDA
  CUDA_CHECK(cudaSetDevice(rank_));
#endif

  switch (stage) {
    case InferStage::STAGE_CONTEXT:
      NLLM_LOG_DEBUG << "ContextDecode infer on work_id: " << rank_;
      model->ContextDecode(weight, forward_reqs);
      break;
    case InferStage::STATE_DECODE:
      NLLM_LOG_DEBUG << "Decode infer on work_id: " << rank_;
      model->Decode(weight, forward_reqs);
      break;
    default:
      throw std::invalid_argument("Invalid infer stage.");
      break;
  }

  return Status();
}

std::future<Status> Worker::ForwardAsync(std::shared_ptr<BaseModel> model, std::shared_ptr<BaseWeight> weight,
                                         InferStage stage, std::vector<ForwardRequest>& forward_reqs) {
  return threadpool_->Submit([=, &forward_reqs]() -> Status { return Forward(model, weight, stage, forward_reqs); });
}

Status Worker::Sampling(std::shared_ptr<Sampler> sampler, std::vector<SamplingRequest>& sampling_reqs) {
  // TODO(karlluo): confirm redundant usage
#ifdef ENABLE_CUDA
  CUDA_CHECK(cudaSetDevice(rank_));

  sampler->Sampling(sampling_reqs, context_->GetComputeStreams()[rank_]);
#endif
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
  for (int worker_id = 0; worker_id < tensor_parallel_size_; ++worker_id) {
    workers_[worker_id].reset(new Worker(worker_id, threadpool_, context));
  }
}

WorkerGroup::~WorkerGroup() { threadpool_->Stop(); }

std::shared_ptr<Worker> WorkerGroup::GetWorker(int rank) {
  if (rank < 0 || rank >= workers_.size()) {
    NLLM_LOG_FATAL << "The worker rank " << rank << " exceed worker size " << workers_.size();
  }
  return workers_[rank];
}

}  // namespace ksana_llm
