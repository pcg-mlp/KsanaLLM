/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "numerous_llm/runtime/context.h"
#include "numerous_llm/runtime/worker.h"
#include "numerous_llm/utils/logger.h"

namespace numerous_llm {

Context::Context(const int tensor_parallel_size, const int pipeline_parallel_size)
    : tensor_parallel_size_(tensor_parallel_size), pipeline_parallel_size_(pipeline_parallel_size) {
  if (pipeline_parallel_size_ != 1) {
    throw std::runtime_error("Only support pipeline_parallel_size == 1");
  }

  device_num_ = GetDeviceNumber();

  if (device_num_ > tensor_parallel_size_ * pipeline_parallel_size_) {
    throw std::runtime_error(fmt::format("{} tensor_parallel_size should not bigger than devices num: {}",
                                         tensor_parallel_size_, device_num_));
  }

  for (int worker_id = 0; worker_id < tensor_parallel_size_; ++worker_id) {
    NLLM_LOG_INFO << "Init nvidia gpu relate handler on device " << worker_id;

    CUDA_CHECK(cudaSetDevice(worker_id));

    NLLM_LOG_INFO << "Init nvidia compute_stream on device " << worker_id;
    cudaStream_t compute_stream;
    CUDA_CHECK(cudaStreamCreate(&compute_stream));
    compute_streams_.emplace_back(std::move(compute_stream));

    NLLM_LOG_INFO << "Init nvidia h2d_stream on device " << worker_id;
    cudaStream_t h2d_stream;
    CUDA_CHECK(cudaStreamCreate(&h2d_stream));
    h2d_streams_.emplace_back(std::move(h2d_stream));

    NLLM_LOG_INFO << "Init nvidia d2h_stream on device " << worker_id;
    cudaStream_t d2h_stream;
    CUDA_CHECK(cudaStreamCreate(&d2h_stream));
    d2h_streams_.emplace_back(std::move(d2h_stream));

    NLLM_LOG_INFO << "Init nvidia nccl_stream on device " << worker_id;
    cudaStream_t nccl_stream;
    CUDA_CHECK(cudaStreamCreate(&nccl_stream));
    nccl_streams_.emplace_back(std::move(nccl_stream));
  }

  nccl_uid_ = GenerateNCCLUniqueID();
  nccl_params_.resize(tensor_parallel_size_);
  NCCL_CHECK(ncclGroupStart());
  // TODO(karlluo): for single machine multiple xpus, device_num is the world_size
  // for multiple machine, world size should change in future, and the same situation of rank_id
  for (int worker_id = 0; worker_id < tensor_parallel_size_; ++worker_id) {
    CUDA_CHECK(cudaSetDevice(worker_id));
    NCCL_CHECK(ncclCommInitRank(/*comm=*/&(nccl_params_[worker_id].nccl_comm),
                                /*nranks=*/tensor_parallel_size_,
                                /*commId=*/nccl_uid_, /*rank=*/worker_id));
  }
  NCCL_CHECK(ncclGroupEnd());
}

Context::~Context() {
  for (int worker_id = 0; worker_id < tensor_parallel_size_; ++worker_id) {
    CUDA_CHECK(cudaSetDevice(worker_id));

    CUDA_CHECK(cudaStreamDestroy(compute_streams_[worker_id]));
    CUDA_CHECK(cudaStreamDestroy(h2d_streams_[worker_id]));
    CUDA_CHECK(cudaStreamDestroy(d2h_streams_[worker_id]));
    CUDA_CHECK(cudaStreamDestroy(nccl_streams_[worker_id]));

    NCCL_CHECK(DestroyNCCLParam(nccl_params_[worker_id]));
  }

  compute_streams_.clear();
  h2d_streams_.clear();
  d2h_streams_.clear();
  nccl_streams_.clear();
  nccl_params_.clear();
}

}  // namespace numerous_llm
