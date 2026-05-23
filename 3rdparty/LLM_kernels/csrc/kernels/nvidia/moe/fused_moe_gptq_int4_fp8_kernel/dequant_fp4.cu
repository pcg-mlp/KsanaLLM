/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

#include "dequant_fp4.h"

#include "csrc/utils/nvidia/cuda_utils.h"
#include "csrc/utils/nvidia/string_utils.h"

namespace llm_kernels {
namespace nvidia {
namespace dequant {

// ============================================================================
// FP4 E2M1 -> FP8 E4M3 Lookup Table (with exponent shift pre-applied)
//
// Index: shift * 16 + nibble (total 128 entries, shifts 0..7)
// shift = max_exp - sub_group_exp
// nibble = 4-bit fp4 value (E2M1 format)
//
// FP4 E2M1 values: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0} (positive)
// After shift by N: multiply by 2^(-N), result is exact in FP8 E4M3
// All products are exactly representable for shift 0..7:
//   max fp4 = 6.0, shift=7 -> 6.0 * 2^-7 = 0.046875 (fp8 normal)
//   min nonzero fp4 = 0.5, shift=7 -> 0.5 * 2^-7 = 0.00390625 (fp8 subnormal m=2)
// ============================================================================
// clang-format off
__constant__ uint8_t kFp4Fp8Lut[128] = {
    // shift=0: fp4 values as-is
    0x00, 0x30, 0x38, 0x3C, 0x40, 0x44, 0x48, 0x4C,
    0x00, 0xB0, 0xB8, 0xBC, 0xC0, 0xC4, 0xC8, 0xCC,
    // shift=1: fp4 * 0.5
    0x00, 0x28, 0x30, 0x34, 0x38, 0x3C, 0x40, 0x44,
    0x00, 0xA8, 0xB0, 0xB4, 0xB8, 0xBC, 0xC0, 0xC4,
    // shift=2: fp4 * 0.25
    0x00, 0x20, 0x28, 0x2C, 0x30, 0x34, 0x38, 0x3C,
    0x00, 0xA0, 0xA8, 0xAC, 0xB0, 0xB4, 0xB8, 0xBC,
    // shift=3: fp4 * 0.125
    0x00, 0x18, 0x20, 0x24, 0x28, 0x2C, 0x30, 0x34,
    0x00, 0x98, 0xA0, 0xA4, 0xA8, 0xAC, 0xB0, 0xB4,
    // shift=4: fp4 * 0.0625
    0x00, 0x10, 0x18, 0x1C, 0x20, 0x24, 0x28, 0x2C,
    0x00, 0x90, 0x98, 0x9C, 0xA0, 0xA4, 0xA8, 0xAC,
    // shift=5: fp4 * 0.03125
    0x00, 0x08, 0x10, 0x14, 0x18, 0x1C, 0x20, 0x24,
    0x00, 0x88, 0x90, 0x94, 0x98, 0x9C, 0xA0, 0xA4,
    // shift=6: fp4 * 0.015625
    0x00, 0x04, 0x08, 0x0C, 0x10, 0x14, 0x18, 0x1C,
    0x00, 0x84, 0x88, 0x8C, 0x90, 0x94, 0x98, 0x9C,
    // shift=7: fp4 * 0.0078125
    0x00, 0x02, 0x04, 0x06, 0x08, 0x0C, 0x10, 0x14,
    0x00, 0x82, 0x84, 0x86, 0x88, 0x8C, 0x90, 0x94,
};
// clang-format on

// ============================================================================
// FP4 -> FP8 dequant kernel with scale merging
//
// High-throughput LUT-based implementation with 4× loop unroll for ILP.
// Achieves ~2720 GB/s on H20 (68% of theoretical 4000 GB/s peak).
//
// Key design choices (validated by ncu profiling):
//   1. uint16 reads → 4 nibbles at once (100% load coalescing)
//   2. uint32 writes → 4 fp8 bytes at once (100% store coalescing)
//   3. 128-byte LUT in shared memory: minimal bank conflicts (3.4%)
//   4. Scale shifts precomputed in shared memory, amortized over all data
//   5. Bit manipulation for bf16 output (no exp2f or __float2bfloat16)
//   6. 4× manual loop unroll: -7.8% total instructions vs non-unrolled
//   7. __ldg for global reads: texture cache path
//   8. programmaticStreamSerialization for dependent kernel overlap
//
// Memory layout:
//   Input weight: contiguous packed fp4 (2 per byte), read as uint16 (4 nibbles)
//   Input scale: E8M0 bytes, 1 per 32 fp4 elements
//   Output fp8: contiguous fp8 bytes, written as uint32 (4 at a time)
//   Output merged_scale: 1 bf16 per 128 fp4 elements
// ============================================================================
template <size_t THREAD, size_t LEN>
__global__ void dequant_fp4_fp8_kernel(const uint16_t* __restrict__ input_weight,
                                       const uint8_t* __restrict__ input_scale,
                                       __nv_fp8x4_e4m3* __restrict__ output_fp8,
                                       __nv_bfloat16* __restrict__ output_scale,
                                       const uint8_t* __restrict__ active_expert_mask, uint32_t blocks_per_expert) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  // 稀疏模式：跳过未命中专家。注意必须在 return 前发 launch_dependents，
  // 否则 PDL 优化失效，依赖 grid 要等到本 grid 全部 block 自然结束才启动。
  if (active_expert_mask != nullptr) {
    const uint32_t expert_id = blockIdx.x / blocks_per_expert;
    if (active_expert_mask[expert_id] == 0) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
      asm volatile("griddepcontrol.launch_dependents;");
#endif
      return;
    }
  }

  constexpr size_t kFp4PerUint16 = 4;
  constexpr size_t kMergedGroupSize = 128;
  constexpr size_t kSubGroupSize = 32;
  constexpr size_t kSubGroupsPerMerged = kMergedGroupSize / kSubGroupSize;
  constexpr size_t kTotalFp4 = LEN * kFp4PerUint16;
  constexpr size_t kNumMergedGroups = kTotalFp4 / kMergedGroupSize;
  constexpr size_t kNumSubGroups = kTotalFp4 / kSubGroupSize;
  constexpr size_t kUint16PerSubGroup = kSubGroupSize / kFp4PerUint16;

  __shared__ uint8_t smem_lut[128];
  __shared__ uint8_t smem_shifts[kNumSubGroups];

  // Load LUT into shared memory (128 bytes = 32 uint32)
  if (threadIdx.x < 32) {
    reinterpret_cast<uint32_t*>(smem_lut)[threadIdx.x] = reinterpret_cast<const uint32_t*>(kFp4Fp8Lut)[threadIdx.x];
  }

  // Phase 1: Load scales and compute shifts
  const size_t scale_base = static_cast<size_t>(blockIdx.x) * kNumSubGroups;
  for (size_t s = threadIdx.x; s < kNumMergedGroups; s += THREAD) {
    const uint32_t scales_packed =
        __ldg(reinterpret_cast<const uint32_t*>(&input_scale[scale_base + s * kSubGroupsPerMerged]));
    const uint8_t e0 = scales_packed & 0xFF;
    const uint8_t e1 = (scales_packed >> 8) & 0xFF;
    const uint8_t e2 = (scales_packed >> 16) & 0xFF;
    const uint8_t e3 = (scales_packed >> 24) & 0xFF;

    const uint8_t max_exp = max(max(e0, e1), max(e2, e3));

    const size_t shift_base = s * kSubGroupsPerMerged;
    smem_shifts[shift_base + 0] = max_exp - e0;
    smem_shifts[shift_base + 1] = max_exp - e1;
    smem_shifts[shift_base + 2] = max_exp - e2;
    smem_shifts[shift_base + 3] = max_exp - e3;

    // Write merged scale: 2^(max_exp - 127) as bf16 via bit manipulation
    const uint16_t bf16_bits = static_cast<uint16_t>(max_exp) << 7;
    reinterpret_cast<uint16_t*>(output_scale)[static_cast<size_t>(blockIdx.x) * kNumMergedGroups + s] = bf16_bits;
  }

  __syncthreads();

  // Phase 2: 4× unrolled dequant with __ldg
  const size_t data_base = static_cast<size_t>(blockIdx.x) * LEN;
  const size_t out_base = static_cast<size_t>(blockIdx.x) * LEN;
  constexpr size_t kStride = THREAD;
  constexpr size_t kUnroll = 4;

  for (size_t base_i = threadIdx.x; base_i < LEN; base_i += kStride * kUnroll) {
    // Issue 4 global loads upfront for ILP
    const uint16_t d0 = __ldg(&input_weight[data_base + base_i]);
    const uint16_t d1 = __ldg(&input_weight[data_base + base_i + kStride]);
    const uint16_t d2 = __ldg(&input_weight[data_base + base_i + kStride * 2]);
    const uint16_t d3 = __ldg(&input_weight[data_base + base_i + kStride * 3]);

    auto process_uint16 = [&](uint16_t data, size_t idx) {
      const uint8_t shift = smem_shifts[idx / kUint16PerSubGroup];
      const uint32_t lb = static_cast<uint32_t>(shift) << 4;
      const uint32_t fp8x4 = static_cast<uint32_t>(smem_lut[lb + (data & 0xF)]) |
                             (static_cast<uint32_t>(smem_lut[lb + ((data >> 4) & 0xF)]) << 8) |
                             (static_cast<uint32_t>(smem_lut[lb + ((data >> 8) & 0xF)]) << 16) |
                             (static_cast<uint32_t>(smem_lut[lb + ((data >> 12) & 0xF)]) << 24);
      reinterpret_cast<uint32_t*>(output_fp8)[out_base + idx] = fp8x4;
    };

    process_uint16(d0, base_i);
    process_uint16(d1, base_i + kStride);
    process_uint16(d2, base_i + kStride * 2);
    process_uint16(d3, base_i + kStride * 3);
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

void dequant_fp4_fp8_launcher(cudaStream_t stream, void* output_fp8, void* output_scale, const void* input_weight,
                              const void* input_scale, size_t num_groups_128, const void* active_expert_mask,
                              size_t num_experts) {
#if defined(ENABLE_FUSED_MOE_GPTQ_INT4_FP8)
  constexpr size_t kThreadsPerBlock = 64;
  constexpr size_t kLenPerBlock = 1024;  // uint16 units: 1024 × 4 = 4096 fp4 = 32 merged-groups

  const size_t total_uint16 = num_groups_128 * 32;  // 128 fp4 / 4 per uint16 = 32 uint16 per group
  KLLM_KERNEL_CHECK(total_uint16 % kLenPerBlock == 0);
  const size_t grid_size = total_uint16 / kLenPerBlock;

  // 稀疏模式：每个 block 负责 4096 个 fp4 元素，需要把 grid_size 均匀分配到 num_experts。
  // 必须满足 grid_size % num_experts == 0，等价于 (N * K) % 4096 == 0。
  // 常见 N、K 都是 128 的倍数 → N*K 至少含 16384 因子 → 该约束自然满足。
  uint32_t blocks_per_expert = 0;
  if (active_expert_mask != nullptr) {
    KLLM_KERNEL_CHECK(num_experts > 0);
    KLLM_KERNEL_CHECK(grid_size % num_experts == 0);
    blocks_per_expert = static_cast<uint32_t>(grid_size / num_experts);
  }

  cudaLaunchConfig_t config;
  config.gridDim = grid_size;
  config.blockDim = kThreadsPerBlock;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = true;
  config.numAttrs = 1;
  config.attrs = attrs;

  auto fn = &dequant_fp4_fp8_kernel<kThreadsPerBlock, kLenPerBlock>;
  cudaLaunchKernelEx(&config, fn, reinterpret_cast<const uint16_t*>(input_weight),
                     reinterpret_cast<const uint8_t*>(input_scale), reinterpret_cast<__nv_fp8x4_e4m3*>(output_fp8),
                     reinterpret_cast<__nv_bfloat16*>(output_scale),
                     reinterpret_cast<const uint8_t*>(active_expert_mask), blocks_per_expert);

#else
  KLLM_KERNEL_THROW("SM version is lower than 90. skipping dequant_fp4 kernel.");
#endif
}

}  // namespace dequant
}  // namespace nvidia
}  // namespace llm_kernels
