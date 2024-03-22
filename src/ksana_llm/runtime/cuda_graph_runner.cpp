/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/runtime/cuda_graph_runner.h"
#include "ksana_llm/utils/nvidia/cuda_utils.h"

namespace ksana_llm {

void CudaGraphBuilder::BeginCapture(cudaStream_t stream) {
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
}

void CudaGraphBuilder::EndCapture(cudaStream_t stream) {
  if (initialized_) {
    CUDA_CHECK(cudaGraphDestroy(graph_));
  }

  CUDA_CHECK(cudaStreamEndCapture(stream, &graph_));
  CUDA_CHECK(cudaGraphInstantiate(&exec_graph_, graph_, nullptr, nullptr, 0));
  initialized_ = true;
}

cudaGraphExec_t CudaGraphBuilder::GetGraphExec() { return exec_graph_; }

bool CudaGraphRunner::CheckGraphAvailable(size_t batch_size) {
  return graph_instances_.find(batch_size) != graph_instances_.end();
}

void CudaGraphRunner::GetGraphBatchSizes(std::vector<size_t>& batch_sizes) {
  batch_sizes.clear();
  for (size_t i = max_batch_size_ / batch_size_step_; i > 0; --i) {
    batch_sizes.push_back(batch_size_step_ * i);
  }

  size_t batch_size = batch_size_step_ / 2;
  while (batch_size > 0) {
    batch_sizes.push_back(batch_size);
    batch_size = batch_size / 2;
  }
}

size_t CudaGraphRunner::GetPaddedBatchSize(size_t batch_size) {
  if (batch_size <= 2) {
    return batch_size;
  } else if (batch_size <= 4) {
    return 4;
  } else {
    return (batch_size + batch_size_step_ - 1) / batch_size_step_ * batch_size_step_;
  }
}

void CudaGraphRunner::LaunchGraph(size_t batch_size, cudaStream_t stream) {
  cudaGraphExec_t graph_exec = graph_instances_[batch_size];
  CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
}

}  // namespace ksana_llm
