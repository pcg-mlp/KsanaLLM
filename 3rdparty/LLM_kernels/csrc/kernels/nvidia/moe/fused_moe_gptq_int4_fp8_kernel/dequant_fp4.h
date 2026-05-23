/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace llm_kernels {
namespace nvidia {
namespace dequant {

/**
 * @brief Dequantize FP4 (E2M1) weights to FP8 (E4M3) with scale merging.
 *
 * Converts packed FP4 weights into FP8 values and merges every 4 consecutive
 * E8M0 group-32 scales into one BF16 group-128 scale.
 *
 * The merged scale is max(s0, s1, s2, s3), and each FP4 value is rescaled by
 * (original_scale / merged_scale) during conversion. This rescaling is exact
 * because all involved values are exact in FP8 E4M3.
 *
 * Sparse expert mode (active_expert_mask != nullptr):
 *   - 仅命中专家（mask byte == 1）的权重会被反量化；
 *   - 未命中专家在 output_fp8 / output_scale 中对应的内存区域**不会被写入**，
 *     保留原有内容（脏数据）；
 *   - 下游 GEMM kernel 通过 expert_ids 索引访问 output，且 expert_ids 只含命中专家，
 *     因此脏数据永远不会被读取，正确性不受影响；
 *   - 时间近似线性下降到 (num_active_experts / num_experts) 比例。
 *
 * @param stream       CUDA stream
 * @param output_fp8   Output FP8 weights, shape (total_elements,), fp8_e4m3
 * @param output_scale Output merged scale, shape (total_elements / 128,), bf16
 * @param input_weight Input packed FP4 weights, shape (total_elements / 2,), uint8
 *                     Each byte contains 2 FP4 E2M1 values (low nibble first)
 * @param input_scale  Input E8M0 scales, shape (total_elements / 32,), uint8
 *                     Each byte is a biased exponent: value = 2^(byte - 127)
 * @param num_groups_128 Number of 128-element groups to process
 *                       (= total_elements / 128 = num_experts * N * K / 128)
 * @param active_expert_mask Optional (default nullptr) device pointer to
 *                           uint8_t[num_experts]. Only experts with mask byte == 1
 *                           are dequantized. Pass nullptr to dequant all experts
 *                           (legacy behavior).
 * @param num_experts        Required when active_expert_mask != nullptr; ignored otherwise.
 *                           Used to compute blocks_per_expert = grid_size / num_experts.
 *                           Note: num_groups_128 must be divisible by num_experts
 *                           (i.e., (N*K) % 128 == 0; satisfied by all common N/K).
 */
void dequant_fp4_fp8_launcher(cudaStream_t stream, void* output_fp8, void* output_scale, const void* input_weight,
                              const void* input_scale, size_t num_groups_128,
                              const void* active_expert_mask = nullptr, size_t num_experts = 0);

}  // namespace dequant
}  // namespace nvidia
}  // namespace llm_kernels
