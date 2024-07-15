/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <vector>

#include "ksana_llm/utils/common_context.h"

#include "ksana_llm/utils/nvidia/cuda_utils.h"
#include "ksana_llm/utils/nvidia/nccl_utils.h"

namespace ksana_llm {

// The class used for nvidia extension.
template <int T>
class NvidiaContextExtension {
 public:
  explicit NvidiaContextExtension(ContextT<T>* base_ptr) { base_ptr_ = base_ptr; }

  std::vector<cudaMemPool_t>& GetMemoryPools() { return memory_pool_; }

  ncclUniqueId& GetNCCLUniqueID() { return nccl_uid_; }

  std::vector<NCCLParam>& GetNCCLParam() { return nccl_params_; }

  std::vector<cublasHandle_t>& GetCublasHandles() { return cublas_handles_; }

  std::vector<cublasLtHandle_t>& GetCublasLtHandles() { return cublaslt_handles_; }

  void** GetCustomAllReduceBuffers() { return static_cast<void**>(reduce_buffers_.data()); }

  void** GetCustomAllReduceMetas() { return static_cast<void**>(reduce_metas_.data()); }

  void** GetCustomAllReduceInputs(int input_index) { return static_cast<void**>(reduce_inputs_[input_index].data()); }

  // Initialize and destroy extension.
  void Initialize();
  void Destroy();

 private:
  // init gpu memory pool
  void InitGpuMemoryPool(const int worker_id);

  // init cublas handle
  void InitCublasHandle(const int worker_id);

  // init nccl handle
  void InitNcclParam();

 private:
  ContextT<T>* base_ptr_ = nullptr;

  // The cuda driver version.
  int cuda_driver_version_;

  // Nvidia GPU memory pool
  std::vector<cudaMemPoolProps> memory_pool_props_;
  std::vector<cudaMemPool_t> memory_pool_;

  // nccl comms
  ncclUniqueId nccl_uid_;
  std::vector<NCCLParam> nccl_params_;

  // cublas handles
  std::vector<cublasHandle_t> cublas_handles_;
  std::vector<cublasLtHandle_t> cublaslt_handles_;

  // The custom reduce buffers and metas.
  std::vector<void*> reduce_buffers_;
  std::vector<void*> reduce_metas_;

  // The max reduce inputs num for custom reduce..
  int max_reduce_inputs_num_{2};

  // The reduce inputs used for nccl.
  std::vector<std::vector<void*>> reduce_inputs_;
};

template <>
struct ExtensionTypeTraits<DEVICE_TYPE_NVIDIA> {
  typedef NvidiaContextExtension<DEVICE_TYPE_NVIDIA> value_type;
};

// 构造扩展类对象
template <>
void ContextT<DEVICE_TYPE_NVIDIA>::InitializeExtension();

// 销毁扩展类对象
template <>
void ContextT<DEVICE_TYPE_NVIDIA>::DestroyExtension();

}  // namespace ksana_llm
