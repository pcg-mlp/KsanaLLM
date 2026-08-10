/* Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * hpc-ops FP8 fused-MoE kernel interface (blockwise and per-tensor quantization).
 * Source: https://github.com/Tencent/hpc-ops (submodule pinned to 8d300e1)
 * SM90+ (Hopper) only. Compiled as static lib with isolated cutlass 4.4.2.
 *
 * Forwards a Qwen3-MoE expert compute to the hpc-ops kernels
 * (hpc::fuse_moe::fuse_moe_blockwise_fp8_async / fuse_moe_pertensor_fp8_async). Gated at
 * the layer level by the `setting.moe_backend.moe_impl` switch; when it does not resolve
 * to hpc the default KsanaLLM MoE path is used unchanged.
 *
 * Expert-parallel: this is a pure single-GPU kernel with no cross-rank comm; the
 * caller (MoeLayer) still owns dispatch/combine. `num_experts_local` are the experts
 * resident on this rank, `num_experts_total` the global pool, `rank_ep` this rank's
 * expert-parallel index. For single-card pass num_experts_local == num_experts_total
 * and rank_ep == 0. Following hpc's own entry, num_experts_total drives only the
 * per-group token average / padding; everything else keys off num_experts_local.
 *
 * Layout contract (all device pointers, verified against the Qwen3-30B-A3B-FP8
 * weight loader + hpc-ops sources):
 *   x_q                 : [num_tokens, hidden]              fp8_e4m3 (per-token blockwise-quantized hidden_states)
 *   x_scale             : [num_tokens, hidden/128]          fp32, row-major
 *   gate_up_weight      : [E, 2*moe_inter, hidden]          fp8_e4m3 (gate concat up)
 *   gate_up_weight_scale: [E, 2*moe_inter/128, hidden/128]  fp32 (last dim already a multiple of 4)
 *   down_weight         : [E, hidden, moe_inter]            fp8_e4m3
 *   down_weight_scale   : [E, hidden/128, moe_inter/128]    fp32 (last dim repadded to a multiple of 4 internally)
 *   topk_ids            : [num_tokens, topk]                int32 (arbitrary order; not required to be sorted)
 *   topk_scale          : [num_tokens, topk]                fp32 (final routing weights)
 *   output              : [num_tokens, hidden]              bf16
 * ==============================================================================*/
#pragma once

#include <cuda_runtime.h>

namespace llm_kernels {
namespace nvidia {

void InvokeHpcBlockwiseFp8Moe(const void* x_q, const void* x_scale, const void* gate_up_weight,
                              const void* gate_up_weight_scale, const void* down_weight,
                              const void* down_weight_scale, const void* topk_ids,
                              const void* topk_scale, void* output, int num_tokens, int hidden_size,
                              int moe_inter_size, int num_experts_local, int num_experts_total,
                              int rank_ep, int topk, float routed_scaling_factor, cudaStream_t stream);

/* Per-tensor (static activation scale) FP8 variant of the above. Same gather / group-GEMM /
 * reduce pipeline, but the GEMM scales are one float per expert instead of a 128x128 block grid,
 * so there is no x_scale and no scale repadding.
 *
 * Layout contract (all device pointers, matching what QuantWeight::BindFp8E4m3ScaleOfMoeWeight
 * already produces for a per-tensor FP8 MoE checkpoint):
 *   x_q               : [num_tokens, hidden]      fp8_e4m3 (hidden_states / up_gate.input_scale)
 *   gate_up_weight    : [E, 2*moe_inter, hidden]  fp8_e4m3 (gate concat up)
 *   gate_up_scale     : [E]                       fp32, up_gate.input_scale * up_gate.weight_scale[e]
 *                                                 (the dequant scale of the gate_up GEMM output)
 *   act_and_mul_scale : [1]                       fp32, 1 / down.input_scale — silu(gate)*up is
 *                                                 multiplied by it to requantize into down's fp8 input
 *   down_weight       : [E, hidden, moe_inter]    fp8_e4m3
 *   down_scale        : [E]                       fp32, down.input_scale * down.weight_scale[e]
 *   topk_ids          : [num_tokens, topk]        int32 (arbitrary order; not required to be sorted)
 *   topk_scale        : [num_tokens, topk]        fp32 (final routing weights)
 *   output            : [num_tokens, hidden]      bf16
 *
 * use_bf16_mul rounds silu(gate) to bf16 before multiplying by up (cheaper, slightly lossier);
 * pass false to keep the multiply in fp32.
 */
void InvokeHpcPerTensorFp8Moe(const void* x_q, const void* gate_up_weight, const void* gate_up_scale,
                              const void* act_and_mul_scale, const void* down_weight,
                              const void* down_scale, const void* topk_ids, const void* topk_scale,
                              void* output, int num_tokens, int hidden_size, int moe_inter_size,
                              int num_experts_local, int num_experts_total, int rank_ep, int topk,
                              bool use_bf16_mul, cudaStream_t stream);

}  // namespace nvidia
}  // namespace llm_kernels
