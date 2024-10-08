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
  layer_num_ = std::any_cast<const int>(parameters[parameter_index++]);
  max_position_embeddings_ = std::any_cast<const int>(parameters[parameter_index++]);
  num_heads_ = std::any_cast<const int>(parameters[parameter_index++]);
  num_kv_heads_ = std::any_cast<const int>(parameters[parameter_index++]);
  head_size_ = std::any_cast<const int>(parameters[parameter_index++]);
  stride_size_ = std::any_cast<const int>(parameters[parameter_index++]);
  tensor_para_size_ = std::any_cast<const int>(parameters[parameter_index++]);
  kv_cache_dtype_ = std::any_cast<DataType>(parameters[parameter_index++]);
  k_scale_ = std::any_cast<const float>(parameters[parameter_index++]);
  v_scale_ = std::any_cast<const float>(parameters[parameter_index++]);
  uint32_t rotary_dim = std::any_cast<const int>(parameters[parameter_index++]);
  base_ = std::any_cast<const float>(parameters[parameter_index++]);
  bool is_neox = std::any_cast<const bool>(parameters[parameter_index++]);
  PositionEncoding position_encoding = std::any_cast<const PositionEncoding>(parameters[parameter_index++]);
  void* cos_sin_cache_ptr = std::any_cast<void*>(parameters[parameter_index++]);
  RoPEScalingFactor rope_scaling_factor_config = std::any_cast<const RoPEScalingFactor>(parameters[parameter_index++]);
  max_batch_size_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  is_context_stage_ = std::any_cast<const bool>(parameters[parameter_index++]);

  // TODO(zhongzhicao): The cast should be removed after implementing ROPE.
  // Cast the unused variables to void to suppress the -Wunused-value warnings.
  (void)rotary_dim;
  (void)is_neox;
  (void)position_encoding;
  (void)cos_sin_cache_ptr;

  block_size_ = GetBlockManager()->GetBlockSize();
  block_token_num_ = GetBlockManager()->GetBlockTokenNum();
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
