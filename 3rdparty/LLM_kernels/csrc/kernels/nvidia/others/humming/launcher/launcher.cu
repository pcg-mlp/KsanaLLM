/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 * Kernel launch implementation for humming MoE GEMM.
 */

#include "csrc/kernels/nvidia/others/humming/launcher/launcher.h"
#include "csrc/kernels/nvidia/others/humming/launcher/tma.h"
#include "csrc/utils/nvidia/assert.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstring>

namespace llm_kernels {
namespace nvidia {
namespace humming {

namespace {

/// Get the number of SMs on the current device.
int32_t GetSmCount() {
  int device = 0;
  cudaGetDevice(&device);
  int sm_count = 0;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
  return sm_count;
}

/// Helper: identity cast to void* for arg packing.
inline void* ToVoidPtr(void* ptr) { return ptr; }

/// Build TMA descriptor for weight B: [num_experts, N, K_packed]
/// K_packed = K / 8 for FP4 stored as uint32
/// TMA sees 2D slices: [K_packed, N] per expert → 3D descriptor
void BuildTmaB(CUtensorMap* tmap,
               const void* b_ptr,
               int64_t num_experts, int64_t size_n, int64_t size_k,
               int32_t block_n, int32_t block_k) {
  // FP4 packs 8 elements into one uint32 → K_packed = K/8, element_size = 4 bytes
  const int64_t k_packed = size_k / 8;
  const uint32_t element_size = 4;  // uint32

  // Logical dimensions (outermost first): [num_experts, N, K_packed]
  uint64_t gmem_dims[3] = {
      static_cast<uint64_t>(num_experts),
      static_cast<uint64_t>(size_n),
      static_cast<uint64_t>(k_packed)
  };

  // Strides in bytes (outermost first, ndim-1 = 2 entries)
  // stride[0] = num_experts stride = N * K_packed * element_size
  // stride[1] = N stride = K_packed * element_size
  uint64_t gmem_strides[2] = {
      static_cast<uint64_t>(size_n * k_packed * element_size),
      static_cast<uint64_t>(k_packed * element_size)
  };

  // Shared memory box: [1, block_n, block_k/8]
  uint32_t smem_box[3] = {
      1,
      static_cast<uint32_t>(block_n),
      static_cast<uint32_t>(block_k / 8)
  };

  BuildTmaDescriptor(tmap, b_ptr,
                     CU_TENSOR_MAP_DATA_TYPE_UINT32,
                     3, gmem_dims, gmem_strides, smem_box, 128);
}

/// Build TMA descriptor for weight scale BS: [num_experts, num_groups, N]
/// where num_groups = K / group_size, stored as uint8 (E8M0)
void BuildTmaBs(CUtensorMap* tmap,
                const void* bs_ptr,
                int64_t num_experts, int64_t size_n, int64_t size_k,
                int32_t block_n, int32_t block_k,
                int32_t weight_scale_group_size) {
  const int64_t num_groups = size_k / weight_scale_group_size;
  const uint32_t element_size = 1;  // uint8

  // Logical: [num_experts, num_groups, N]
  uint64_t gmem_dims[3] = {
      static_cast<uint64_t>(num_experts),
      static_cast<uint64_t>(num_groups),
      static_cast<uint64_t>(size_n)
  };

  // Strides in bytes (outermost first)
  uint64_t gmem_strides[2] = {
      static_cast<uint64_t>(num_groups * size_n * element_size),
      static_cast<uint64_t>(size_n * element_size)
  };

  // Shared memory box: [1, block_k/group_size, block_n]
  uint32_t num_groups_per_block = static_cast<uint32_t>(block_k / weight_scale_group_size);
  uint32_t smem_box[3] = {
      1,
      num_groups_per_block,
      static_cast<uint32_t>(block_n)
  };

  BuildTmaDescriptor(tmap, bs_ptr,
                     CU_TENSOR_MAP_DATA_TYPE_UINT8,
                     3, gmem_dims, gmem_strides, smem_box, 0);
}

/// Build TMA descriptor for activation A: [M, K]
/// Used only for GROUPED_CONTIGUOUS gemm type.
void BuildTmaA(CUtensorMap* tmap,
               const void* a_ptr,
               int64_t size_m, int64_t size_k,
               int32_t block_m, int32_t block_k,
               ComputeDtype compute_dtype) {
  // FP8 = 1 byte, BF16 = 2 bytes
  const uint32_t element_size = (compute_dtype == ComputeDtype::FP8) ? 1 : 2;
  CUtensorMapDataType dtype = (compute_dtype == ComputeDtype::FP8)
      ? CU_TENSOR_MAP_DATA_TYPE_UINT8
      : CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;

  // Logical: [M, K]
  uint64_t gmem_dims[2] = {
      static_cast<uint64_t>(size_m),
      static_cast<uint64_t>(size_k)
  };

  // Strides in bytes: stride[0] = M stride = K * element_size
  uint64_t gmem_strides[1] = {
      static_cast<uint64_t>(size_k * element_size)
  };

  // Shared memory box: [block_m, block_k]
  uint32_t smem_box[2] = {
      static_cast<uint32_t>(block_m),
      static_cast<uint32_t>(block_k)
  };

  BuildTmaDescriptor(tmap, a_ptr, dtype, 2, gmem_dims, gmem_strides, smem_box, 128);
}

/// Build TMA descriptor for output C: [M, N]
/// Used only for GROUPED_CONTIGUOUS gemm type.
void BuildTmaC(CUtensorMap* tmap,
               void* c_ptr,
               int64_t size_m, int64_t size_n,
               int32_t block_m, int32_t block_n) {
  const uint32_t element_size = 2;  // BF16

  // Logical: [M, N]
  uint64_t gmem_dims[2] = {
      static_cast<uint64_t>(size_m),
      static_cast<uint64_t>(size_n)
  };

  // Strides: stride[0] = M stride = N * element_size
  uint64_t gmem_strides[1] = {
      static_cast<uint64_t>(size_n * element_size)
  };

  // Shared memory box: [block_m, block_n]
  uint32_t smem_box[2] = {
      static_cast<uint32_t>(block_m),
      static_cast<uint32_t>(block_n)
  };

  BuildTmaDescriptor(tmap, c_ptr,
                     CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                     2, gmem_dims, gmem_strides, smem_box, 0);
}

}  // anonymous namespace

void launch_kernel(const KernelMeta* meta,
                   const void* a, const void* b, void* c,
                   const void* a_scale, const void* b_scale,
                   const void* global_scale, const void* b_zero_point,
                   const void* bias,
                   const int32_t* sorted_ids, const int32_t* expert_ids,
                   const int32_t* num_tokens_padded, const void* expert_layout,
                   int32_t* locks,
                   void* workspace,
                   int64_t size_m, int64_t size_n, int64_t size_k,
                   int64_t num_experts, int64_t top_k,
                   bool use_int64_expert_layout,
                   cudaStream_t stream) {
  KLLM_KERNEL_CHECK_WITH_INFO(meta != nullptr, "KernelMeta must not be null");
  KLLM_KERNEL_CHECK_WITH_INFO(meta->func != nullptr, "Kernel CUfunction is null");

  // ------- Grid configuration (stream-K) -------
  const int32_t sm_count = GetSmCount();
  const uint32_t num_ctas = static_cast<uint32_t>(meta->num_ctas_per_sm * sm_count);

  // ------- Build TMA descriptors -------
  // B (weight) always uses TMA
  CUtensorMap tensor_map_b = {};
  if (meta->use_tma_b && b != nullptr) {
    // Weight scale group size: use block_k as approximation for group_size
    // (the actual group_size is encoded in kernel metadata)
    BuildTmaB(&tensor_map_b, b, num_experts, size_n, size_k,
              meta->block_n, meta->block_k);
  }

  // BS (weight scale) uses TMA when available
  CUtensorMap tensor_map_bs = {};
  if (meta->use_tma_bs && b_scale != nullptr) {
    // For E8M0 fused scale, group_size is block_k
    // For regular group quantization, we use block_k as group_size
    int32_t weight_scale_group_size = meta->block_k;
    BuildTmaBs(&tensor_map_bs, b_scale, num_experts, size_n, size_k,
               meta->block_n, meta->block_k, weight_scale_group_size);
  }

  // A (activation) uses TMA only for GROUPED_CONTIGUOUS
  CUtensorMap tensor_map_a = {};
  if (meta->use_tma_a && a != nullptr) {
    BuildTmaA(&tensor_map_a, a, size_m, size_k,
              meta->block_m, meta->block_k, meta->compute_dtype);
  }

  // C (output) uses TMA only for GROUPED_CONTIGUOUS
  CUtensorMap tensor_map_c = {};
  if (meta->use_tma_c && c != nullptr) {
    BuildTmaC(&tensor_map_c, c, size_m, size_n,
              meta->block_m, meta->block_n);
  }

  // BZP (zero point) — currently not building TMA, pass pointer directly
  CUtensorMap tensor_map_bzp = {};
  // Bias — currently not building TMA, pass pointer directly
  CUtensorMap tensor_map_bias = {};

  // ------- Prepare raw pointers for arg packing -------
  void* a_ptr = const_cast<void*>(a);
  void* b_ptr = const_cast<void*>(b);
  void* c_ptr = c;
  void* as_ptr = const_cast<void*>(a_scale);
  void* bs_ptr = const_cast<void*>(b_scale);
  void* bzp_ptr = const_cast<void*>(b_zero_point);
  void* bias_ptr = const_cast<void*>(bias);
  void* gs_ptr = const_cast<void*>(global_scale);
  void* sorted_ids_ptr = const_cast<void*>(static_cast<const void*>(sorted_ids));
  void* expert_ids_ptr = const_cast<void*>(static_cast<const void*>(expert_ids));
  void* num_tokens_padded_ptr = const_cast<void*>(static_cast<const void*>(num_tokens_padded));
  void* expert_layout_ptr = const_cast<void*>(expert_layout);
  void* tensor_map_buffer_ptr = workspace;  // device-side TMA storage
  void* locks_void_ptr = static_cast<void*>(locks);

  // ------- Locks for stream-K -------
  // NOTE: Locks buffer is zeroed by the caller (HummingFp4MoeLayer) before each GEMM.
  // The layer-level memset covers the full locks buffer, which is always >= num_ctas * sizeof(int32_t).
  // Do NOT add a memset here — it would be redundant with the caller's zero.

  // ------- Shape scalars -------
  uint32_t shape_m_u32 = static_cast<uint32_t>(size_m);
  uint32_t top_k_u32 = static_cast<uint32_t>(top_k);
  KLLM_KERNEL_CHECK_WITH_INFO(
      static_cast<int64_t>(shape_m_u32) == size_m,
      "size_m=%ld overflows uint32_t", static_cast<long>(size_m));
  KLLM_KERNEL_CHECK_WITH_INFO(
      static_cast<int64_t>(top_k_u32) == top_k,
      "top_k=%ld overflows uint32_t", static_cast<long>(top_k));

  // ------- Pack 17 kernel arguments (matching humming reference) -------
  void* kernel_args[] = {
      meta->use_tma_a  ? ToVoidPtr(&tensor_map_a)    : ToVoidPtr(&a_ptr),          // 0: A
      meta->use_tma_b  ? ToVoidPtr(&tensor_map_b)    : ToVoidPtr(&b_ptr),          // 1: B
      meta->use_tma_c  ? ToVoidPtr(&tensor_map_c)    : ToVoidPtr(&c_ptr),          // 2: C
      &as_ptr,                                                                      // 3: AS (activation scale)
      meta->use_tma_bs ? ToVoidPtr(&tensor_map_bs)   : ToVoidPtr(&bs_ptr),         // 4: BS
      ToVoidPtr(&bzp_ptr),                                                          // 5: BZP
      ToVoidPtr(&bias_ptr),                                                         // 6: Bias
      &gs_ptr,                                                                      // 7: GS (global scale)
      &sorted_ids_ptr,                                                              // 8: sorted_ids
      &expert_ids_ptr,                                                              // 9: expert_ids
      &num_tokens_padded_ptr,                                                       // 10: num_tokens_padded
      &expert_layout_ptr,                                                           // 11: expert_layout
      &tensor_map_buffer_ptr,                                                       // 12: tensor_map_buffer
      &locks_void_ptr,                                                              // 13: locks
      &shape_m_u32,                                                                 // 14: shape_m (uint32_t)
      &top_k_u32,                                                                   // 15: top_k (uint32_t)
      &use_int64_expert_layout                                                      // 16: bool
  };

  // ------- Launch configuration -------
  CUlaunchConfig config = {};
  config.gridDimX = num_ctas;
  config.gridDimY = 1;
  config.gridDimZ = 1;
  config.blockDimX = static_cast<unsigned int>(meta->num_threads);
  config.blockDimY = 1;
  config.blockDimZ = 1;
  config.sharedMemBytes = static_cast<unsigned int>(meta->smem_size);
  config.hStream = stream;

  // Cluster launch attributes (for multi-cast / multi-CTA cooperation)
  CUlaunchAttribute attrs[1];
  config.attrs = nullptr;
  config.numAttrs = 0;
  // Note: multi_cast is not exposed in current KernelMeta. If needed, add cluster config here.
  (void)attrs;

  // ------- Set shared memory size and launch -------
  CUfunction func = meta->func;
  CUresult res;

  res = cuFuncSetAttribute(func,
                           CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                           meta->smem_size);
  if (res != CUDA_SUCCESS) {
    const char* err_str = nullptr;
    cuGetErrorString(res, &err_str);
    KLLM_KERNEL_CHECK_WITH_INFO(false,
        "cuFuncSetAttribute(smem=%d) failed: %s",
        meta->smem_size, err_str ? err_str : "unknown");
  }

  res = cuLaunchKernelEx(&config, func, kernel_args, nullptr);
  if (res != CUDA_SUCCESS) {
    const char* err_str = nullptr;
    cuGetErrorString(res, &err_str);
    KLLM_KERNEL_CHECK_WITH_INFO(false,
        "cuLaunchKernelEx failed: %s (grid=%u, threads=%d, smem=%d)",
        err_str ? err_str : "unknown",
        num_ctas, meta->num_threads, meta->smem_size);
  }
}

}  // namespace humming
}  // namespace nvidia
}  // namespace llm_kernels
