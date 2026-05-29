/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "device/nvidia/nvidia_device_context_impl.h"

#include <cuda_profiler_api.h>
#include <csignal>

#include "3rdparty/LLM_kernels/csrc/utils/nvidia/cuda_utils.h"
#include "device/nvidia/deepseek_deepgemm_bridge.h"

namespace ksana_llm {

// The minimum cuda version that support mempool.
constexpr int CUDA_MEMPOOL_MIN_DRIVER_VERSION = 11030;
constexpr int CUDA_GEMM_SUPPORT_FP8_MIN_CUBLASLT_VERSION = 120103;

template <int T>
void NvidiaDeviceContextExtension<T>::InitGpuMemoryPool() {
  // TODO(yancyliu): to optimize memory with multiple memory pool description
  constexpr size_t kMemDescCount = 1;
  int device_supports_memory_pools = 0;
  CUDA_CHECK(
      cudaDeviceGetAttribute(&device_supports_memory_pools, cudaDevAttrMemoryPoolsSupported, base_ptr_->device_id_));

  // NOTE: 1 if the device supports using the cudaMallocAsync and cudaMemPool family of APIs, and 0 otherwise
  if (device_supports_memory_pools == 0) {
    KLLM_LOG_WARNING << fmt::format("GPU {} is not support GPU mempool, skip init.", base_ptr_->device_id_);
    return;
  }

  KLLM_LOG_DEBUG << "Init nvidia memroy pool on GPU " << base_ptr_->device_id_;
  CUDA_CHECK(cudaDriverGetVersion(&cuda_driver_version_));
  if (cuda_driver_version_ >= CUDA_MEMPOOL_MIN_DRIVER_VERSION) {
    int pool_supported_handle_types = 0;
    cudaMemPool_t mempool;
    CUDA_CHECK(cudaDeviceGetAttribute(&pool_supported_handle_types, cudaDevAttrMemoryPoolSupportedHandleTypes,
                                      base_ptr_->device_id_));
    CUDA_CHECK(cudaDeviceGetDefaultMemPool(&mempool, base_ptr_->device_id_));
    uint64_t threshold = UINT64_MAX;
    CUDA_CHECK(cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold));

    int enable = 1;
    CUDA_CHECK(cudaMemPoolSetAttribute(mempool, cudaMemPoolReuseFollowEventDependencies, &enable));
    CUDA_CHECK(cudaMemPoolSetAttribute(mempool, cudaMemPoolReuseAllowOpportunistic, &enable));
    CUDA_CHECK(cudaMemPoolSetAttribute(mempool, cudaMemPoolReuseAllowInternalDependencies, &enable));
  }
}

template <int T>
void NvidiaDeviceContextExtension<T>::InitCublasHandle() {
  KLLM_LOG_DEBUG << "Init nvidia cublas/cublasLt on worker " << base_ptr_->device_id_;
  CUDA_CHECK(cublasCreate(&cublas_handle_));
  CUDA_CHECK(cublasLtCreate(&cublaslt_handle_));

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, base_ptr_->device_id_));
  // FP8 GEMM is supported when arch >= 89 and cublaslt >= 12.1.3.
  base_ptr_->is_gemm_fp8_supported_ = (((prop.major >= 8 && prop.minor >= 9) || prop.major >= 9) &&
                                       (cublasLtGetVersion() >= CUDA_GEMM_SUPPORT_FP8_MIN_CUBLASLT_VERSION));

  // binding compute stream to cublas
  CUDA_CHECK(cublasSetStream(cublas_handle_, base_ptr_->primary_compute_stream_->Get()));
}

template <int T>
void NvidiaDeviceContextExtension<T>::InitNcclParam(ncclUniqueId nccl_uid) {
  KLLM_LOG_DEBUG << "Init nvidia nccl param.";
  nccl_uid_ = nccl_uid;

  // Init nccl
  const int node_device_num = base_ptr_->node_device_num_;
  if (node_device_num > 1) {
    int device_rank = base_ptr_->device_id_;
    KLLM_LOG_DEBUG << "Invoke ncclCommInitRank, node_device_num:" << node_device_num << ", device_rank:" << device_rank;
    NCCL_CHECK(ncclCommInitRank(&(nccl_param_.nccl_comm), node_device_num, nccl_uid_, device_rank));
  }

  init_done_.store(true, std::memory_order_relaxed);
  KLLM_LOG_DEBUG << "Init nvidia nccl param done.";
}

template <int T>
bool NvidiaDeviceContextExtension<T>::CheckGpuP2PAccess() {
  int src_id = base_ptr_->device_id_;
  for (int dst_id = 0; dst_id < base_ptr_->node_device_num_; ++dst_id) {
    if (src_id == dst_id) {
      continue;
    }

    // NOTE: returns in *canAccessPeer a value of 1 if device is capable of directly accessing memory
    // from peerDevice and 0 otherwise.
    int can_access_peer = 0;

    // Refer https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__PEER.html canAccessPeer Returned
    // access capability.
    CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, static_cast<int>(src_id), static_cast<int>(dst_id)));
    CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, static_cast<int>(dst_id), static_cast<int>(src_id)));
    if (can_access_peer == 0) {
      return false;
    }

    // cudaDeviceEnablePeerAccess is not required in multi-process mode
  }
  return true;
}

template <int T>
void NvidiaDeviceContextExtension<T>::Initialize() {
  int device_count;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count <= 0) {
    KLLM_THROW("There is not GPUs detected on your machine.");
  }
  CUDA_CHECK(cudaDriverGetVersion(&base_ptr_->driver_version_));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  sm_ = prop.major * 10 + prop.minor;
  int cuda_ver_tmp;
  CUDA_CHECK(cudaRuntimeGetVersion(&cuda_ver_tmp));
  cuda_ver_ = static_cast<uint32_t>(cuda_ver_tmp);
  KLLM_LOG_INFO << fmt::format("Get SM: {}, cuda version: {}", sm_, cuda_ver_);

  // Resize shared pointers for custom all reduce
  const size_t node_device_num = base_ptr_->node_device_num_;
  reduce_signals_.resize(node_device_num);
  reduce_inputs_.resize(node_device_num);

  // Used to store data buffer, flag buffer, and lamport buffer for each rank respectively
  trt_reduce_buffers_.resize(3 * node_device_num);
  trt_reduce_flags_.resize(node_device_num);
  trt_reduce_workspaces_.resize(node_device_num);

  KLLM_LOG_DEBUG << "Init nvidia gpu relate handler on worker " << base_ptr_->device_id_;
  InitGpuMemoryPool();
  InitCublasHandle();
  is_p2p_supported_ = CheckGpuP2PAccess();
  if (llm_kernels::utils::GetNvLinkVersion(0, base_ptr_->device_id_) == 0) {
    is_full_nvlink_ = false;
  }
  // For performance reasons, enable multicast only when using more than 4 GPUs
  is_multicast_supported_ = (node_device_num > 4);
  for (size_t worker_id = 0; worker_id < node_device_num && is_multicast_supported_; ++worker_id) {
    int multicast_supported = 0;
    CU_CHECK(cuDeviceGetAttribute(&multicast_supported, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, worker_id));
    is_multicast_supported_ &= (multicast_supported != 0);
  }
  if (is_multicast_supported_) {
    // Init nvls mcast memory if required
    NvlsMcastMemory::GetInstance()->Initialize(node_device_num);
  }
  KLLM_LOG_INFO << fmt::format("is_p2p_supported: {}, is_full_nvlink: {}, is_multicast_supported: {}",
                               is_p2p_supported_, is_full_nvlink_, is_multicast_supported_);
}

template <int T>
void NvidiaDeviceContextExtension<T>::Destroy() {
  // wait nccl async init finish
  GetNCCLParam();

  // Finalize DeepGEMM runtime before destroying other resources
  deepseek::FinalizeDeepGemm();

  CUDA_CHECK(cublasDestroy(cublas_handle_));
  CUDA_CHECK(cublasLtDestroy(cublaslt_handle_));
  NCCL_CHECK(DestroyNCCLParam(nccl_param_));
}

template <>
void DeviceContextT<DEVICE_TYPE_NVIDIA>::InitializeExtension() {
  ext = new NvidiaDeviceContextExtension<DEVICE_TYPE_NVIDIA>(this);
  ext->Initialize();
}

template <>
void DeviceContextT<DEVICE_TYPE_NVIDIA>::DestroyExtension() {
  ext->Destroy();
  delete ext;
}

// Explicit initialize.
template void NvidiaDeviceContextExtension<DEVICE_TYPE_NVIDIA>::InitNcclParam(ncclUniqueId);

}  // namespace ksana_llm
