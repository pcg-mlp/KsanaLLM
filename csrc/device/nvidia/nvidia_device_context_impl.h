/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <vector>

#include "device/device_context_impl.h"

#include "device/nvidia/cuda_utils.h"
#include "device/nvidia/nccl_utils.h"

#include "3rdparty/LLM_kernels/csrc/kernels/nvidia/gemm_wrapper/gemm_algo_map.h"

namespace ksana_llm {

// The class used for nvidia extension.
template <int T>
class NvidiaDeviceContextExtension {
 public:
  explicit NvidiaDeviceContextExtension(DeviceContextT<T>* base_ptr) { base_ptr_ = base_ptr; }

  cudaMemPool_t& GetMemoryPool() { return memory_pool_; }

  ncclUniqueId& GetNCCLUniqueID() { return nccl_uid_; }

  NCCLParam& GetNCCLParam() {
    while (!init_done_.load(std::memory_order_relaxed)) {
      std::this_thread::yield();
    }
    return nccl_param_;
  }

  cublasHandle_t& GetCublasHandle() { return cublas_handle_; }

  cublasLtHandle_t& GetCublasLtHandle() { return cublaslt_handle_; }

  void** GetCustomAllReduceSignals() { return reduce_signals_.data(); }
  void** GetCustomAllReduceInputs() { return reduce_inputs_.data(); }

  std::vector<void*>& GetTrtAllReduceBuffers() { return trt_reduce_buffers_; }
  std::vector<void*>& GetTrtAllReduceFlags() { return trt_reduce_flags_; }
  std::vector<void*>& GetTrtAllReduceWorkspaces() { return trt_reduce_workspaces_; }

  const bool IsFullNvLink() const { return is_full_nvlink_; }

  const bool IsP2PAccessSupported() const { return is_p2p_supported_; }
  const bool IsMulticastSupported() const { return is_multicast_supported_; }

  uint32_t GetComputeCapacity() { return sm_; }
  uint32_t GetCudaVersion() { return cuda_ver_; }

  llm_kernels::nvidia::GPUGemmAlgoHelper& GetGPUGemmAlgoHelper() { return gpu_gemm_algo_helper_; }

  // Initialize and destroy extension.
  void Initialize();
  void Destroy();

  // init nccl handle
  void InitNcclParam(ncclUniqueId nccl_uid);

 private:
  // init gpu memory pool
  void InitGpuMemoryPool();

  // init cublas handle
  void InitCublasHandle();

  // return true if Gpus support P2P access and false otherwise
  bool CheckGpuP2PAccess();

 private:
  DeviceContextT<T>* base_ptr_ = nullptr;

  bool is_full_nvlink_ = true;

  // The cuda driver version.
  int cuda_driver_version_;

  // Nvidia GPU memory pool
  cudaMemPool_t memory_pool_;

  // nccl comms
  ncclUniqueId nccl_uid_;
  NCCLParam nccl_param_;

  // cublas handles
  cublasHandle_t cublas_handle_;
  cublasLtHandle_t cublaslt_handle_;

  // Stores the intermediate results for each gpu of custom all reduce.
  std::vector<void*> reduce_signals_;

  // Maintain the input tensor required by each gpu of custom all reduce.
  std::vector<void*> reduce_inputs_;

  // Maintain shared device pointers for trt allreduce
  std::vector<void*> trt_reduce_buffers_;
  std::vector<void*> trt_reduce_flags_;
  std::vector<void*> trt_reduce_workspaces_;

  uint32_t sm_{0};
  uint32_t cuda_ver_{0};
  // The helper for providing the best performance gemm algo
  llm_kernels::nvidia::GPUGemmAlgoHelper gpu_gemm_algo_helper_;

  std::atomic_bool init_done_{false};

  bool is_p2p_supported_{false};
  // NVLink4 provides multicast support
  // Refer to http://docs.nvidia.com/networking/software/accelerator-software/index.html#nvidia-sharp
  bool is_multicast_supported_{false};
};

template <>
struct DeviceExtensionTypeTraits<DEVICE_TYPE_NVIDIA> {
  typedef NvidiaDeviceContextExtension<DEVICE_TYPE_NVIDIA> value_type;
};

// Initialize extension.
template <>
void DeviceContextT<DEVICE_TYPE_NVIDIA>::InitializeExtension();

// Destroy extension.
template <>
void DeviceContextT<DEVICE_TYPE_NVIDIA>::DestroyExtension();

}  // namespace ksana_llm
