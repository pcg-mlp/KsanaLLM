/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "models/base/common_model_weight_loader.h"
#include "utils/status.h"
#include "utils/tensor.h"

namespace ksana_llm {

// RTN (Round-To-Nearest) quantization method for MoE expert weights.
// Handles INT4 weights with blockwise weight_scale and per-expert input_scale,
// targeting CUTLASS MoE kernels.
class RtnMethod {
 public:
  RtnMethod(std::shared_ptr<CommonModelWeightLoader> common_weight_loader, int tp)
      : common_weight_loader_(common_weight_loader), tp_(tp) {}

  Status load_experts_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                           const std::string& weight_name, const Tensor& weight_tensor, int dev_rank);

  Status process_experts_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                              const std::string& weight_prefix_name, int dev_rank);

 private:
  std::shared_ptr<CommonModelWeightLoader> common_weight_loader_;
  int tp_;

  // Verify all weight names exist in device_model_weights.
  // Returns error Status with details if any are missing.
  Status VerifyWeightsExist(const std::unordered_map<std::string, Tensor>& device_model_weights,
                            const std::vector<std::string>& weight_names, const std::string& label);

  // Apply CUTLASS MoE interleave transformation to blockwise scale tensor.
  // Reshapes [E, N, K_groups] -> permute {0,2,1,3} -> [E, K_groups/interleave, N*interleave]
  void ApplyScaleInterleave(Tensor& scales_tensor, size_t interleave, int dev_rank);

  // Create act_scale (bf16, 1/max_input_scale) and alpha (fp32, [num_experts,1]) tensors
  // for CUTLASS MoE kernel, and store them in device_model_weights.
  void CreateActScaleAndAlpha(std::unordered_map<std::string, Tensor>& device_model_weights,
                              const std::string& proj_prefix, float max_input_scale,
                              size_t local_num_experts, int dev_rank);

  // Compute max input_scale across all local experts for given projection types,
  // erasing the per-expert input_scale entries from device_model_weights.
  // proj_suffixes: e.g. {"gate_proj", "up_proj"} for up_gate, or {"down_proj"} for down.
  float ComputeMaxInputScale(std::unordered_map<std::string, Tensor>& device_model_weights,
                             const std::string& weight_prefix_name,
                             const std::vector<int>& local_expert_indices,
                             const std::vector<std::string>& proj_suffixes);

  // Process a single projection type (up_gate_proj or down_proj):
  // - Merge expert weight tensors and reshape to 3D
  // - Merge expert weight_scale tensors, reshape to 3D, apply interleave
  // - Compute max input_scale and create act_scale/alpha if needed
  // proj_name: e.g. "up_gate_proj" or "down_proj"
  // sub_proj_suffixes: e.g. {"gate_proj", "up_proj"} or {"down_proj"}
  //   For multiple sub-projs, weights are merged in [sub0, sub1, ...] order per expert.
  // reshape_dims: target 3D shape [local_num_experts, dim0, dim1_packed]
  // interleave_k_dim: K dimension used to compute CUTLASS interleave factor
  Status ProcessExpertProj(std::unordered_map<std::string, Tensor>& device_model_weights,
                           const std::string& weight_prefix_name,
                           const std::vector<int>& local_expert_indices,
                           const std::string& proj_name,
                           const std::vector<std::string>& sub_proj_suffixes,
                           const std::vector<size_t>& reshape_dims,
                           size_t interleave_k_dim,
                           bool has_input_scale,
                           int dev_rank);

  // Bind weight_scale, act_scale, and alpha pointers to a merged weight tensor.
  // Must be called AFTER all map insertions are done to avoid rehash invalidation.
  void BindProjPointers(std::unordered_map<std::string, Tensor>& device_model_weights,
                        const std::string& weight_prefix_name,
                        const std::string& proj_name,
                        bool has_input_scale);
};

}  // namespace ksana_llm
