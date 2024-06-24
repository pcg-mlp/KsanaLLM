/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/attention_layer.h"

#include "csrc/kernels/ascend/rotary_embedding/rotary_embedding.h"

namespace ksana_llm {

template <typename T>
std::shared_ptr<llm_kernels::ascend::PagedAttention<T>> AttentionLayer<T>::ascend_paged_attn_ = nullptr;

template <typename T>
Status AttentionLayer<T>::Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, context, rank);
  int parameter_index = 0;
  layer_index_ = std::any_cast<const int>(parameters[parameter_index++]);
  layer_num_ = std::any_cast<const int>(parameters[parameter_index++]);
  int max_position_embeddings = std::any_cast<const int>(parameters[parameter_index++]);
  num_heads_ = std::any_cast<const int>(parameters[parameter_index++]);
  num_kv_heads_ = std::any_cast<const int>(parameters[parameter_index++]);
  head_size_ = std::any_cast<const int>(parameters[parameter_index++]);
  stride_size_ = std::any_cast<const int>(parameters[parameter_index++]);
  tensor_para_size_ = std::any_cast<const int>(parameters[parameter_index++]);
  kv_cache_dtype_ = std::any_cast<DataType>(parameters[parameter_index++]);
  uint32_t rotary_dim = std::any_cast<const int>(parameters[parameter_index++]);
  float base = std::any_cast<const float>(parameters[parameter_index++]);
  bool is_neox = std::any_cast<const bool>(parameters[parameter_index++]);
  bool is_alibi = std::any_cast<const bool>(parameters[parameter_index++]);
  void* cos_sin_cache_ptr = std::any_cast<void*>(parameters[parameter_index++]);

  block_size_ = GetBlockManager()->GetBlockSize();
  block_token_num_ = GetBlockManager()->GetBlockTokenNum();

  float scaling_factor = 1.0f;
  llm_kernels::ascend::RotaryEmbeddingType scaling_type = llm_kernels::ascend::RotaryEmbeddingType::DEFAULT;

  if (ascend_paged_attn_ == nullptr) {
    // setting scaling factor and mode
    RoPEScalingFactor rope_scaling_factor_config =
        std::any_cast<const RoPEScalingFactor>(parameters[parameter_index++]);
    if (rope_scaling_factor_config.type == "dynamic") {
      scaling_type = llm_kernels::ascend::RotaryEmbeddingType::DYNAMIC_NTK_SCALING;
      scaling_factor = rope_scaling_factor_config.factor;
    } else if ("liner") {
      scaling_type = llm_kernels::ascend::RotaryEmbeddingType::LINEAR_SCALING;
      scaling_factor = rope_scaling_factor_config.factor;
    }

    ascend_paged_attn_ = std::make_shared<llm_kernels::ascend::PagedAttention<T>>();
    ascend_paged_attn_->Initialize(num_heads_, num_kv_heads_, head_size_, layer_num_, layer_index_, block_token_num_,
                                   context->GetComputeStreams()[rank].Get(), scaling_type, scaling_factor);
  }

  return Status();
}

template class AttentionLayer<float>;
template class AttentionLayer<float16>;

template <typename T>
void AttentionLayer<T>::PrepareWorkspaceBuffer(const size_t workspace_needed, void* workspace_buf_ptr) {
  // NOTE(karlluo): allocate the workspace for float32
  if (workspace_block_id_ == -1 || workspace_size_ == 0) {
    workspace_size_ = workspace_needed;
    GetBlockManager()->AllocateContiguous(workspace_size_, workspace_block_id_);
  }
  // NOTE(karlluo): not enough, reallocate
  if (workspace_size_ < workspace_needed) {
    GetBlockManager()->FreeContiguous(workspace_block_id_);
    GetBlockManager()->AllocateContiguous(workspace_needed, workspace_block_id_);
    workspace_size_ = workspace_needed;
  }

  GetBlockManager()->GetContiguousPtr(workspace_block_id_, workspace_buf_ptr);
}

}  // namespace ksana_llm
