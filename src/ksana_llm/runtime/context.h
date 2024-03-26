/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <vector>

#include "ksana_llm/block_manager/memory_block.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/ret_code.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/stream.h"

#ifdef ENABLE_CUDA
#  include "ksana_llm/utils/nvidia/cuda_utils.h"
#  include "ksana_llm/utils/nvidia/nccl_utils.h"
#endif

namespace ksana_llm {

// The global context, like cuda stream, nccl handler.
class Context {
 public:
  Context(const int tensor_parallel_size, const int pipeline_parallel_size, const MemoryDevice device_type);
  ~Context();

  int GetTensorParallelSize() { return tensor_parallel_size_; }

  int GetPipeLineParallelSize() { return pipeline_parallel_size_; }

  inline bool IsRunContextDecodeAndDecodeSerially() { return is_contextdecode_and_decode_run_serially_; }

  std::vector<Stream>& GetMemoryManageStreams() { return memory_manage_streams_; }

  std::vector<Stream>& GetComputeStreams() { return compute_streams_; }

  std::vector<Stream>& GetH2DStreams() { return h2d_streams_; }

  std::vector<Stream>& GetD2HStreams() { return d2h_streams_; }

  std::vector<Stream>& GetD2DStreams() { return d2d_streams_; }

  std::vector<Stream>& GetNCCLStreams() { return nccl_streams_; }

#ifdef ENABLE_CUDA
  std::vector<cudaMemPool_t>& GetMemoryPools() { return memory_pool_; }

  ncclUniqueId& GetNCCLUniqueID() { return nccl_uid_; }

  std::vector<NCCLParam>& GetNCCLParam() { return nccl_params_; }

  std::vector<cublasHandle_t>& GetCublasHandles() { return cublas_handles_; }

  std::vector<cublasLtHandle_t>& GetCublasLtHandles() { return cublaslt_handles_; }
#endif

  void** GetCustomAllReduceBuffers() { return static_cast<void**>(reduce_buffers_.data()); }

  void** GetCustomAllReduceMetas() { return static_cast<void**>(reduce_metas_.data()); }

  void** GetCustomAllReduceInputs(int input_index) { return static_cast<void**>(reduce_inputs_[input_index].data()); }

  // Get the device type.
  MemoryDevice GetDevice() { return device_type_; }

 private:
  int device_num_{0};
  int tensor_parallel_size_{0};
  int pipeline_parallel_size_{0};
  const int defalt_device_num_{0};
  int driver_version_;
  // if true, only one thread execute context_decode/decode and context_decode decode run in sync
  // TODO(karlluo): load from environment
  bool is_contextdecode_and_decode_run_serially_{true};
  MemoryDevice device_type_{MemoryDevice::MEMORY_GPU};

  // streams
  std::vector<Stream> memory_manage_streams_;
  std::vector<Stream> compute_streams_;
  std::vector<Stream> h2d_streams_;
  std::vector<Stream> d2h_streams_;
  std::vector<Stream> d2d_streams_;
  std::vector<Stream> nccl_streams_;

#ifdef ENABLE_CUDA
  // Nvidia GPU memory pool
  std::vector<cudaMemPoolProps> memory_pool_props_;
  std::vector<cudaMemPool_t> memory_pool_;
  // nccl comms
  ncclUniqueId nccl_uid_;
  std::vector<NCCLParam> nccl_params_;
  // cublas handles
  std::vector<cublasHandle_t> cublas_handles_;
  std::vector<cublasLtHandle_t> cublaslt_handles_;
#endif

  std::vector<void*> reduce_buffers_;
  std::vector<void*> reduce_metas_;
  int max_reduce_inputs_num_{2};
  std::vector<std::vector<void*>> reduce_inputs_;

 private:
  // init cuda streams
  void InitStreams(const int worker_id);

#ifdef ENABLE_CUDA
  // init gpu memory pool
  void InitGpuMemoryPool(const int worker_id);

  // init cublas handle
  void InitCublasHandle(const int worker_id);

  // init nccl handle
  void InitNcclParam();
#endif
};

}  // namespace ksana_llm
