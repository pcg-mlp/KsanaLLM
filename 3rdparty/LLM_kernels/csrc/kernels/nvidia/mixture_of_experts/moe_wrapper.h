/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "csrc/kernels/nvidia/mixture_of_experts/moe_norm_config.h"
#include "csrc/utils/nvidia/cuda_bf16_wrapper.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace llm_kernels {
namespace nvidia {

template <typename T>
class MoeGemmWrapper {
 public:
  void GetWorkspaceSize(size_t token_num, size_t expert_num, size_t expert_hidden_size, size_t expert_inter_size,
                        size_t expert_topk, int tp_size, int rank, bool use_lora, size_t& ws_bytes);

  size_t GetBestConfigIndex();

  void Gemm(void const* input_activations_void, void const* gating_output, void const* fc1_expert_weights_void,
            void const* fc2_expert_weights_void, int64_t const num_rows, int64_t const hidden_size,
            int64_t const inter_size, int const num_experts, int const topk, char* workspace_ptr,
            void* final_output_void, void* token_topk_final_scales_void, int* expanded_source_row_to_expanded_dest_row,
            int* expert_for_source_row, int tp_size, int rank, bool use_lora, size_t best_config_index,
            MOEExpertScaleNormalizationMode moe_norm_mode, cudaStream_t stream);
};

}  // namespace nvidia
}  // namespace llm_kernels