/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "numerous_llm/runtime/context.h"
#include "numerous_llm/utils/logger.h"

namespace numerous_llm {

Context::Context(const int tensor_parallel_size, const int pipeline_parallel_size)
    : tensor_parallel_size_(tensor_parallel_size), pipeline_parallel_size_(pipeline_parallel_size) {
  if (pipeline_parallel_size_ != 1) {
    throw std::runtime_error("Only support pipeline_parallel_size == 1");
  }

  device_num_ = GetDeviceNumber();

  if (device_num_ > tensor_parallel_size_) {
    throw std::runtime_error(fmt::format("{} tensor_parallel_size should not bigger than devices num: {}",
                                         tensor_parallel_size_, device_num_));
  }

  nccl_uid_ = GenerateNCCLUniqueID();

  for (int device_id = 0; device_id < device_num_; ++device_id) {
    NLLM_LOG_INFO << "Init nvidia gpu relate handler on device " << device_id;

    CUDA_CHECK(cudaSetDevice(device_id));
    cudaStream_t compute_stream;
    CUDA_CHECK(cudaStreamCreate(&compute_stream));
    compute_streams_.emplace_back(std::move(compute_stream));

    cudaStream_t h2d_stream;
    CUDA_CHECK(cudaStreamCreate(&h2d_stream));
    h2d_streams_.emplace_back(std::move(h2d_stream));

    cudaStream_t d2h_stream;
    CUDA_CHECK(cudaStreamCreate(&d2h_stream));
    d2h_streams_.emplace_back(std::move(d2h_stream));

    cudaStream_t nccl_stream;
    CUDA_CHECK(cudaStreamCreate(&nccl_stream));
    nccl_streams_.emplace_back(std::move(nccl_stream));

    NCCLParam nccl_param;
    // TODO(karlluo): for single machine multiple xpus, device_num is the world_size
    // for multiple machine, world size should change in future, and the same situation of rank_id
    NCCL_CHECK(InitNCCLParam(nccl_param, device_num_, device_id, nccl_uid_));
    nccl_params_.emplace_back(std::move(nccl_param));
  }
}

Context::~Context() {
  for (int device_id = 0; device_id < device_num_; ++device_id) {
    CUDA_CHECK(cudaSetDevice(device_id));

    CUDA_CHECK(cudaStreamDestroy(compute_streams_[device_id]));
    CUDA_CHECK(cudaStreamDestroy(h2d_streams_[device_id]));
    CUDA_CHECK(cudaStreamDestroy(d2h_streams_[device_id]));
    CUDA_CHECK(cudaStreamDestroy(nccl_streams_[device_id]));

    NCCL_CHECK(DestroyNCCLParam(nccl_params_[device_id]));
  }

  compute_streams_.clear();
  h2d_streams_.clear();
  d2h_streams_.clear();
  nccl_streams_.clear();
  nccl_params_.clear();
}

}  // namespace numerous_llm
