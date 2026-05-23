/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */
#include "csrc/kernels/nvidia/others/sglang/main/topk_softplus_sqrt/topk_softplus_sqrt_sglang.h"

#include "csrc/kernels/nvidia/others/sglang/main/topk_softplus_sqrt/hash_topk.h"
#include "csrc/kernels/nvidia/others/sglang/main/topk_softplus_sqrt/moe_fused_gate.h"

#include <stdexcept>
#include <string>

namespace llm_kernels::nvidia::sglang_gate {

void InvokeTopkSoftplusSqrtSglang(const void* gating_output, GateInputDtype input_dtype, void* topk_weights,
                                  void* topk_indices, int num_tokens, int num_experts, int topk, bool renormalize,
                                  float routed_scaling_factor, const float* correction_bias, const void* input_ids,
                                  const void* tid2eid, cudaStream_t stream) {
  const bool use_hash = (input_ids != nullptr) && (tid2eid != nullptr);

  const uint32_t e = static_cast<uint32_t>(num_experts);
  const uint32_t k = static_cast<uint32_t>(topk);
  if (kSglangSupportedExpertTopk.count({e, k}) == 0) {
    throw std::runtime_error(
        "sglang topk_softplus_sqrt: unsupported (num_experts, topk) = (" + std::to_string(num_experts) + ", " +
        std::to_string(topk) + "); supported narrow set is {(256,6),(256,8),(384,6),(384,8)}");
  }

  constexpr bool kUsePDL = true;
  if (use_hash) {
    InvokeHashTopK(gating_output, input_dtype, static_cast<const int32_t*>(input_ids),
                   static_cast<const int32_t*>(tid2eid), static_cast<float*>(topk_weights),
                   static_cast<int32_t*>(topk_indices), static_cast<uint32_t>(num_tokens), e, k,
                   routed_scaling_factor, kUsePDL, stream);
  } else {
    if (correction_bias == nullptr) {
      throw std::runtime_error("sglang topk_softplus_sqrt score path requires non-null correction_bias");
    }
    const bool ok = InvokeScoreTopK(gating_output, input_dtype, correction_bias, static_cast<float*>(topk_weights),
                                    static_cast<int32_t*>(topk_indices), static_cast<uint32_t>(num_tokens), e, k,
                                    renormalize, routed_scaling_factor, kUsePDL, stream);
    if (!ok) {
      throw std::runtime_error("sglang score kernel dispatch failed (should be caught by narrow-set check)");
    }
  }
}

}  // namespace llm_kernels::nvidia::sglang_gate
