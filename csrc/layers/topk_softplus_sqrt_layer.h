/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "layers/base_layer.h"

namespace ksana_llm {

struct TopKSoftplusSqrtLayerParameters : public BaseLayerParameters {
  uint32_t topk{0};           // Experts selected per token (e.g. 6 for DS-V4).
  uint32_t num_experts{0};    // Total routed experts (e.g. 256, 384).
  uint32_t max_token_num{0};  // Used to size token_expert_indices workspace (non-hash mode).
  bool renormalize{false};    // Divide per-token weights by their sum before scaling.
  float routed_scaling_factor{1.0f};
  bool use_hash{false};                 // true: DS-V4 hash path; false: score-based top-k.
  bool use_correction_bias{false};      // Non-hash path only: adds bias when selecting top-k.
  DataType topk_ids_dtype{TYPE_INT32};  // Dtype of topk_ids output. TYPE_INT32 or TYPE_INT64.

  std::string ToString() const {
    return fmt::format(
        "topk:{}, num_experts:{}, max_token_num:{}, renormalize:{}, "
        "routed_scaling_factor:{}, use_hash:{}, use_correction_bias:{}, topk_ids_dtype:{}",
        topk, num_experts, max_token_num, renormalize, routed_scaling_factor, use_hash, use_correction_bias,
        topk_ids_dtype);
  }
};

enum class SqrtSoftplusBackend { kVllm, kSglang, kHybrid };

#ifdef ENABLE_CUDA
class TopKSoftplusSqrtLayer : public BaseLayer {
 public:
  Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
              std::shared_ptr<Context> context, int rank) override;

  size_t GetWorkspaceSize() override;

  Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors,
                         const LayerExecContext& exec_ctx = {}) override;

 private:
  TopKSoftplusSqrtLayerParameters params_;
  SqrtSoftplusBackend backend_{SqrtSoftplusBackend::kHybrid};
};
#endif

}  // namespace ksana_llm
