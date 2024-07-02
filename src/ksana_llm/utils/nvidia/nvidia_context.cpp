/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/utils/nvidia/nvidia_context.h"

namespace ksana_llm {

// The minimum cuda version that support mempool.
constexpr int CUDA_MEMPOOL_MIN_DRIVER_VERSION = 11030;

template <int T>
void NvidiaContextExtension<T>::InitGpuMemoryPool(const int worker_id) {
  NLLM_LOG_DEBUG << "Init nvidia memroy pool on worker " << worker_id;
  CUDA_CHECK(cudaDriverGetVersion(&cuda_driver_version_));
  if (cuda_driver_version_ >= CUDA_MEMPOOL_MIN_DRIVER_VERSION) {
    int device_supports_memory_pools = 0;
    int pool_supported_handle_types = 0;
    cudaMemPool_t mempool;
    CUDA_CHECK(cudaDeviceGetAttribute(&device_supports_memory_pools, cudaDevAttrMemoryPoolsSupported, worker_id));
    CUDA_CHECK(
        cudaDeviceGetAttribute(&pool_supported_handle_types, cudaDevAttrMemoryPoolSupportedHandleTypes, worker_id));
    CUDA_CHECK(cudaDeviceGetDefaultMemPool(&mempool, worker_id));
    // Set access_id's accessing to the worker_id's mempool.
    for (int access_id = 0; access_id < base_ptr_->tensor_parallel_size_; ++access_id) {
      if (access_id != worker_id) {
        cudaMemAccessDesc desc = {};
        desc.location.type = cudaMemLocationTypeDevice;
        desc.location.id = access_id;
        desc.flags = cudaMemAccessFlagsProtReadWrite;
        int can_access = 0;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, access_id, worker_id));
        if (can_access == 0) {
          NLLM_LOG_ERROR << "GPU " << access_id << " is not capable of directly accessing memory of peer GPU "
                         << worker_id;
          exit(-1);
        }
        CUDA_CHECK(cudaMemPoolSetAccess(mempool, &desc, 1));
      }
    }
    memory_pool_.emplace_back(std::move(mempool));
  }
}

template <int T>
void NvidiaContextExtension<T>::InitCublasHandle(const int worker_id) {
  NLLM_LOG_DEBUG << "Init nvidia cublas/cublasLt on worker " << worker_id;
  cublasHandle_t cublas_handle;
  cublasLtHandle_t cublaslt_handle;
  CUDA_CHECK(cublasCreate(&cublas_handle));
  CUDA_CHECK(cublasLtCreate(&cublaslt_handle));
  cublas_handles_.emplace_back(cublas_handle);
  cublaslt_handles_.emplace_back(cublaslt_handle);

  // binding compute stream to cublas
  CUDA_CHECK(cublasSetStream(cublas_handles_[worker_id], base_ptr_->compute_streams_[worker_id].Get()));
}

template <int T>
void NvidiaContextExtension<T>::InitNcclParam() {
  NLLM_LOG_DEBUG << "Init nvidia nccl param.";
  reduce_metas_.resize(max_reduce_inputs_num_);
  reduce_buffers_.resize(base_ptr_->tensor_parallel_size_);
  reduce_inputs_.resize(max_reduce_inputs_num_);
  for (int i = 0; i < max_reduce_inputs_num_; ++i) {
    reduce_inputs_[i].resize(base_ptr_->tensor_parallel_size_);
  }

  nccl_uid_ = GenerateNCCLUniqueID();
  nccl_params_.resize(base_ptr_->tensor_parallel_size_);
  if (base_ptr_->tensor_parallel_size_ == 1) return;
  NCCL_CHECK(ncclGroupStart());
  // TODO(karlluo): for single machine multiple xpus, device_num is the world_size
  // for multiple machine, world size should change in future, and the same situation of rank_id
  for (int worker_id = 0; worker_id < base_ptr_->tensor_parallel_size_; ++worker_id) {
    CUDA_CHECK(cudaSetDevice(worker_id));
    NCCL_CHECK(
        ncclCommInitRank(&(nccl_params_[worker_id].nccl_comm), base_ptr_->tensor_parallel_size_, nccl_uid_, worker_id));
  }
  NCCL_CHECK(ncclGroupEnd());
}

template <int T>
void NvidiaContextExtension<T>::Initialize() {
  CUDA_CHECK(cudaDriverGetVersion(&base_ptr_->driver_version_));

  for (int worker_id = 0; worker_id < base_ptr_->tensor_parallel_size_; ++worker_id) {
    NLLM_LOG_DEBUG << "Init nvidia gpu relate handler on worker " << worker_id;

    CUDA_CHECK(cudaSetDevice(worker_id));

    InitGpuMemoryPool(worker_id);

    InitCublasHandle(worker_id);
  }

  InitNcclParam();

  // reset device id
  CUDA_CHECK(cudaSetDevice(base_ptr_->defalt_device_num_));
}

template <int T>
void NvidiaContextExtension<T>::Destroy() {
  for (int worker_id = 0; worker_id < base_ptr_->tensor_parallel_size_; ++worker_id) {
    CUDA_CHECK(cudaSetDevice(worker_id));
    CUDA_CHECK(cublasDestroy(cublas_handles_[worker_id]));
    CUDA_CHECK(cublasLtDestroy(cublaslt_handles_[worker_id]));
    NCCL_CHECK(DestroyNCCLParam(nccl_params_[worker_id]));
  }
}

template <>
void ContextT<DEVICE_TYPE_NVIDIA>::InitializeExtension() {
  ext = new NvidiaContextExtension<DEVICE_TYPE_NVIDIA>(this);
  ext->Initialize();
}

template <>
void ContextT<DEVICE_TYPE_NVIDIA>::DestroyExtension() {
  ext->Destroy();
  delete ext;
}

}  // namespace ksana_llm
