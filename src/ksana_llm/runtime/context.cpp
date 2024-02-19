/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/runtime/context.h"
#include "ksana_llm/runtime/worker.h"
#include "ksana_llm/utils/logger.h"

namespace ksana_llm {

constexpr int CUDA_MEMPOOL_MIN_DRIVER_VERSION = 11030;

Context::Context(const int tensor_parallel_size, const int pipeline_parallel_size)
    : tensor_parallel_size_(tensor_parallel_size), pipeline_parallel_size_(pipeline_parallel_size) {
  if (pipeline_parallel_size_ != 1) {
    throw std::runtime_error("Only support pipeline_parallel_size == 1");
  }

  device_num_ = GetDeviceNumber();

  if (device_num_ < tensor_parallel_size_ * pipeline_parallel_size_) {
    throw std::runtime_error(fmt::format("{} tensor_parallel_size should not bigger than devices num: {}",
                                         tensor_parallel_size_, device_num_));
  }

  CUDA_CHECK(cudaDriverGetVersion(&driver_version_));

  for (int worker_id = 0; worker_id < tensor_parallel_size_; ++worker_id) {
    NLLM_LOG_DEBUG << "Init nvidia gpu relate handler on worker " << worker_id;

    CUDA_CHECK(cudaSetDevice(worker_id));

    InitGpuMemoryPool(worker_id);

    InitCudaStreams(worker_id);

    InitCublasHandle(worker_id);
  }

  InitNcclParam();

  // reset device id
  CUDA_CHECK(cudaSetDevice(defalt_device_num_));
}

Context::~Context() {
  for (int worker_id = 0; worker_id < tensor_parallel_size_; ++worker_id) {
    CUDA_CHECK(cudaSetDevice(worker_id));

    CUDA_CHECK(cudaStreamDestroy(compute_streams_[worker_id]));
    CUDA_CHECK(cudaStreamDestroy(h2d_streams_[worker_id]));
    CUDA_CHECK(cudaStreamDestroy(d2h_streams_[worker_id]));
    CUDA_CHECK(cudaStreamDestroy(d2d_streams_[worker_id]));
    CUDA_CHECK(cudaStreamDestroy(nccl_streams_[worker_id]));

    CUDA_CHECK(cublasDestroy(cublas_handles_[worker_id]));
    CUDA_CHECK(cublasLtDestroy(cublaslt_handles_[worker_id]));

    NCCL_CHECK(DestroyNCCLParam(nccl_params_[worker_id]));
  }

  memory_manage_streams_.clear();
  compute_streams_.clear();
  h2d_streams_.clear();
  d2h_streams_.clear();
  d2d_streams_.clear();
  nccl_streams_.clear();
  nccl_params_.clear();
}

void Context::InitGpuMemoryPool(const int worker_id) {
  NLLM_LOG_DEBUG << "Init nvidia memroy pool on worker " << worker_id;
  if (driver_version_ >= CUDA_MEMPOOL_MIN_DRIVER_VERSION) {
    int device_supports_memory_pools = 0;
    int pool_supported_handle_types = 0;
    cudaMemPool_t mempool;
    CUDA_CHECK(cudaDeviceGetAttribute(&device_supports_memory_pools, cudaDevAttrMemoryPoolsSupported, worker_id));
    CUDA_CHECK(
        cudaDeviceGetAttribute(&pool_supported_handle_types, cudaDevAttrMemoryPoolSupportedHandleTypes, worker_id));
    CUDA_CHECK(cudaDeviceGetDefaultMemPool(&mempool, worker_id));
    memory_pool_.emplace_back(std::move(mempool));
  }
}

void Context::InitCudaStreams(const int worker_id) {
  NLLM_LOG_DEBUG << "Init nvidia memroy_manage_stream on worker " << worker_id;
  cudaStream_t memory_manage_stream;
  CUDA_CHECK(cudaStreamCreate(&memory_manage_stream));
  memory_manage_streams_.emplace_back(std::move(memory_manage_stream));

  NLLM_LOG_DEBUG << "Init nvidia compute_stream on worker " << worker_id;
  cudaStream_t compute_stream;
  CUDA_CHECK(cudaStreamCreate(&compute_stream));
  compute_streams_.emplace_back(std::move(compute_stream));

  NLLM_LOG_DEBUG << "Init nvidia h2d_stream on worker " << worker_id;
  cudaStream_t h2d_stream;
  CUDA_CHECK(cudaStreamCreate(&h2d_stream));
  h2d_streams_.emplace_back(std::move(h2d_stream));

  NLLM_LOG_DEBUG << "Init nvidia d2h_stream on worker " << worker_id;
  cudaStream_t d2h_stream;
  CUDA_CHECK(cudaStreamCreate(&d2h_stream));
  d2h_streams_.emplace_back(std::move(d2h_stream));

  NLLM_LOG_DEBUG << "Init nvidia d2d_stream on worker " << worker_id;
  cudaStream_t d2d_stream;
  CUDA_CHECK(cudaStreamCreate(&d2d_stream));
  d2d_streams_.emplace_back(std::move(d2d_stream));

  NLLM_LOG_DEBUG << "Init nvidia nccl_stream on worker " << worker_id;
  cudaStream_t nccl_stream;
  CUDA_CHECK(cudaStreamCreate(&nccl_stream));
  nccl_streams_.emplace_back(std::move(nccl_stream));
}

void Context::InitCublasHandle(const int worker_id) {
  NLLM_LOG_DEBUG << "Init nvidia cublas/cublasLt on worker " << worker_id;
  cublasHandle_t cublas_handle;
  cublasLtHandle_t cublaslt_handle;
  CUDA_CHECK(cublasCreate(&cublas_handle));
  CUDA_CHECK(cublasLtCreate(&cublaslt_handle));
  cublas_handles_.emplace_back(cublas_handle);
  cublaslt_handles_.emplace_back(cublaslt_handle);

  // binding compute stream to cublas
  CUDA_CHECK(cublasSetStream(cublas_handles_[worker_id], compute_streams_[worker_id]));
}

void Context::InitNcclParam() {
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

}  // namespace ksana_llm
