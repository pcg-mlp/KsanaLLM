/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <vector>

#include "ksana_llm/block_manager/memory_block.h"
#include "ksana_llm/utils/nvidia/cuda_utils.h"
#include "ksana_llm/utils/nvidia/nccl_utils.h"
#include "ksana_llm/utils/ret_code.h"

namespace ksana_llm {

// The global context, like cuda stream, nccl handler.
class Context {
 public:
  Context(const int tensor_parallel_size, const int pipeline_parallel_size);
  ~Context();

  int GetTensorParallelSize() { return tensor_parallel_size_; }

  int GetPipeLineParallelSize() { return pipeline_parallel_size_; }

  inline bool IsRunContextDecodeAndDecodeSerially() { return is_contextdecode_and_decode_run_serially_; }

  std::vector<cudaMemPool_t>& GetMemoryPools() { return memory_pool_; }

  std::vector<cudaStream_t>& GetMemoryManageStreams() { return memory_manage_streams_; }

  std::vector<cudaStream_t>& GetComputeStreams() { return compute_streams_; }

  std::vector<cudaStream_t>& GetH2DStreams() { return h2d_streams_; }

  std::vector<cudaStream_t>& GetD2HStreams() { return d2h_streams_; }

  std::vector<cudaStream_t>& GetD2DStreams() { return d2d_streams_; }

  std::vector<cudaStream_t>& GetNCCLStreams() { return nccl_streams_; }

  ncclUniqueId& GetNCCLUniqueID() { return nccl_uid_; }

  std::vector<NCCLParam>& GetNCCLParam() { return nccl_params_; }

  std::vector<cublasHandle_t>& GetCublasHandles() { return cublas_handles_; }

  std::vector<cublasLtHandle_t>& GetCublasLtHandles() { return cublaslt_handles_; }

  // Get the device type.
  MemoryDevice GetDevice() {
    // Support GPU only for now.
    return MemoryDevice::MEMORY_GPU;
  }

 private:
  int device_num_{0};
  int tensor_parallel_size_{0};
  int pipeline_parallel_size_{0};
  const int defalt_device_num_{0};
  int driver_version_;
  // if true, only one thread execute context_decode/decode and context_decode decode run in sync
  // TODO(karlluo): load from environment
  bool is_contextdecode_and_decode_run_serially_{true};

  // Nvidia GPU memory pool
  std::vector<cudaMemPoolProps> memory_pool_props_;
  std::vector<cudaMemPool_t> memory_pool_;

  // cuda streams
  std::vector<cudaStream_t> memory_manage_streams_;
  std::vector<cudaStream_t> compute_streams_;
  std::vector<cudaStream_t> h2d_streams_;
  std::vector<cudaStream_t> d2h_streams_;
  std::vector<cudaStream_t> d2d_streams_;
  std::vector<cudaStream_t> nccl_streams_;
  // nccl comms
  ncclUniqueId nccl_uid_;
  std::vector<NCCLParam> nccl_params_;
  // cublas handles
  std::vector<cublasHandle_t> cublas_handles_;
  std::vector<cublasLtHandle_t> cublaslt_handles_;

 private:
  // init gpu memory pool
  void InitGpuMemoryPool(const int worker_id);

  // init cuda streams
  void InitCudaStreams(const int worker_id);

  // init cublas handle
  void InitCublasHandle(const int worker_id);

  // init nccl handle
  void InitNcclParam();
};

}  // namespace ksana_llm
