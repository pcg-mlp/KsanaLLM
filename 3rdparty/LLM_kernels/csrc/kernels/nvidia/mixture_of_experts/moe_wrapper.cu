/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "csrc/kernels/nvidia/mixture_of_experts/moe_kernels.h"
#include "csrc/kernels/nvidia/mixture_of_experts/moe_wrapper.h"

namespace llm_kernels {
namespace nvidia {

template <typename T>
void MoeGemmWrapper<T>::GetWorkspaceSize(size_t token_num, size_t expert_num, size_t expert_hidden_size,
                                         size_t expert_inter_size, size_t expert_topk, int tp_size, int rank,
                                         bool use_lora, size_t& ws_bytes) {
  llm_kernels::nvidia::MOEParallelismConfig parallelism_config(tp_size, rank, 1, 0);
  auto moe_gemm = std::make_shared<llm_kernels::nvidia::CutlassMoeFCRunner<T, T>>();
  size_t moe_workspace_size =
      moe_gemm->getWorkspaceSize(token_num, expert_hidden_size, expert_inter_size, expert_num, expert_topk,
                                 llm_kernels::nvidia::ActivationType::Swiglu, parallelism_config, use_lora);
  // Output of post-softmax routing probabilities
  size_t scale_probabilities_size = token_num * expert_num * sizeof(float);
  // Permutation map
  size_t src_to_dest_map_size = expert_topk * token_num * sizeof(int);
  // Selected expert map
  size_t selected_expert_size = expert_topk * token_num * sizeof(int);
  size_t lora_workspace_size = 0;
  if (use_lora) {
    // TODO(winminkong): add lora workspace size
  }
  ws_bytes =
      moe_workspace_size + scale_probabilities_size + src_to_dest_map_size + selected_expert_size + lora_workspace_size;
}

template <typename T>
size_t MoeGemmWrapper<T>::GetBestConfigIndex() {  // TODO(winminkong): add best config index
  auto moe_gemm = std::make_shared<llm_kernels::nvidia::CutlassMoeFCRunner<T, T>>();
  std::vector<cutlass_extensions::CutlassGemmConfig> configs;
  configs = moe_gemm->getTactics();
  return 0;
}

template <typename T>
void MoeGemmWrapper<T>::Gemm(void const* input_activations_void, void const* gating_output,
                             void const* fc1_expert_weights_void, void const* fc2_expert_weights_void,
                             int64_t const num_rows, int64_t const hidden_size, int64_t const inter_size,
                             int const num_experts, int const topk, char* workspace_ptr, void* final_output_void,
                             void* token_topk_final_scales_void, int* expanded_source_row_to_expanded_dest_row,
                             int* expert_for_source_row, int tp_size, int rank, bool use_lora, size_t best_config_index,
                             MOEExpertScaleNormalizationMode moe_norm_mode, cudaStream_t stream) {
  int64_t const num_not_finished = num_rows;
  llm_kernels::nvidia::MOEParallelismConfig parallelism_config(tp_size, rank, 1, 0);
  QuantParams quant_params{};  // TODO(winminkong): support quant moe and lora moe
  LoraParams lora_params{};

  auto moe_gemm = std::make_shared<llm_kernels::nvidia::CutlassMoeFCRunner<T, T>>();

  std::vector<cutlass_extensions::CutlassGemmConfig> configs;
  configs = moe_gemm->getTactics();
  moe_gemm->setTactic(configs[best_config_index + 1], configs[best_config_index + 2]);
  // mixtral : MOEExpertScaleNormalizationMode::RENORMALIZE
  // qwen2_moe : MOEExpertScaleNormalizationMode::NONE
  moe_gemm->runMoe(input_activations_void, static_cast<float const*>(gating_output), fc1_expert_weights_void, nullptr,
                   llm_kernels::nvidia::ActivationType::Swiglu, fc2_expert_weights_void, nullptr, quant_params,
                   num_rows, hidden_size, inter_size, num_experts, topk, workspace_ptr, final_output_void, nullptr,
                   num_not_finished, token_topk_final_scales_void, expanded_source_row_to_expanded_dest_row,
                   expert_for_source_row, parallelism_config, moe_norm_mode, use_lora, lora_params, stream);
}

template class MoeGemmWrapper<float>;
template class MoeGemmWrapper<half>;
#ifdef ENABLE_BF16
template class MoeGemmWrapper<__nv_bfloat16>;
#endif

}  // namespace nvidia
}  // namespace llm_kernels