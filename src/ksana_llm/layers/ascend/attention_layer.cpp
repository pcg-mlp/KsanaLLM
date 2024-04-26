/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/attention_layer.h"

#include "csrc/kernels/ascend/rotary_embedding/rotary_embedding.h"

namespace ksana_llm {
template <typename T>
Status AttentionLayer<T>::Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, context, rank);
  int parameter_index = 0;
  layer_index_ = std::any_cast<const int>(parameters[parameter_index++]);
  int max_position_embeddings = std::any_cast<const int>(parameters[parameter_index++]);
  num_heads_ = std::any_cast<const int>(parameters[parameter_index++]);
  num_kv_heads_ = std::any_cast<const int>(parameters[parameter_index++]);
  head_size_ = std::any_cast<const int>(parameters[parameter_index++]);
  stride_size_ = std::any_cast<const int>(parameters[parameter_index++]);
  tensor_para_size_ = std::any_cast<const int>(parameters[parameter_index++]);
  uint32_t rotary_dim = std::any_cast<const int>(parameters[parameter_index++]);
  float base = std::any_cast<const float>(parameters[parameter_index++]);
  bool is_neox = std::any_cast<const bool>(parameters[parameter_index++]);
  bool is_alibi = std::any_cast<const bool>(parameters[parameter_index++]);
  void* cos_sin_cache_ptr = std::any_cast<void*>(parameters[parameter_index++]);

  block_size_ = GetBlockManager()->GetBlockSize();
  block_token_num_ = GetBlockManager()->GetBlockTokenNum();

  if (ascend_flash_attn_ == nullptr) {
    // setting scaling factor and mode
    RoPEScalingFactor rope_scaling_factor_config =
      std::any_cast<const RoPEScalingFactor>(parameters[parameter_index++]);
    float scaling_factor = 1.0f;
    if (rope_scaling_factor_config.type == "dynamic") {
      scaling_factor = rope_scaling_factor_config.factor;
    }

    aclDataType dtype = aclDataType::ACL_FLOAT16;

    const float rope_theta = 10000.0;
    ascend_flash_attn_ = std::make_shared<llm_kernels::ascend::FlashAttentionACL>();

    ascend_flash_attn_->Init(max_position_embeddings, head_size_, num_heads_, num_heads_, rope_theta, scaling_factor,
                             dtype, context->GetComputeStreams()[rank].Get(), GetWorkSpaceFunc());
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
