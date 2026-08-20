/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "layers/attention_layer.h"

namespace ksana_llm {

struct PagedAttentionInputs {
#ifdef ENABLE_CUDA
  const Tensor* query;
  const Tensor* context_lens;
  const Tensor* kv_list;
  const Tensor* cache_offset;
  const Tensor* rotary_embedding_pos;
  const Tensor* rotary_embedding_mask;
  const Tensor* workspace;
  const Tensor* forward_shape;
  const Tensor* qkv_workspace;
  const Tensor* query_layernorm_weight;
  const Tensor* key_layernorm_weight;
  const Tensor* layer_kv_cache_ptr;
  const Tensor* block_table;
#elif defined(ENABLE_ACL)
  const Tensor* qkv_tensor;
  const Tensor* rotary_embedding_pos;
  const Tensor* layers_slot_mapping;
  const Tensor* layers_block_table;
  const Tensor* k_cache_blocks_base;
  const Tensor* v_cache_blocks_base;
  const Tensor* seq_len_host;
  const Tensor* forward_shape;
  const Tensor* atb_attention_attr;
#elif defined(ENABLE_KUNLUN)
  // Same tensor protocol as Ascend/ATB, plus optional QK-Norm weights.
  const Tensor* qkv_tensor;
  const Tensor* rotary_embedding_pos;
  const Tensor* layers_slot_mapping;
  const Tensor* layers_block_table;
  const Tensor* k_cache_blocks_base;
  const Tensor* v_cache_blocks_base;
  const Tensor* seq_len_host;
  const Tensor* forward_shape;
  const Tensor* atb_attention_attr;
  const Tensor* query_layernorm_weight = nullptr;
  const Tensor* key_layernorm_weight = nullptr;
#endif
};

class PagedAttentionLayer : public AttentionLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;

  Status Forward(const PagedAttentionInputs& inputs, std::vector<Tensor>& output_tensors);

#ifdef ENABLE_CUDA
  // Override SetWorkspaceBuffer to initialize FlashInfer workspaces.
  virtual Status SetWorkspaceBuffer(const std::shared_ptr<Tensor>& workspace_buffer) override;

  // Initialize shared FlashInfer workspaces for the given rank.
  // This function should be called after Init() if using FlashInfer backend.
  void SetFlashInferWorkspace(int num_heads, int num_kv_heads, int head_dim, int rank);
#endif

 private:
  template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
  Status ForwardT(const PagedAttentionInputs& inputs, std::vector<Tensor>& output_tensors);

 private:
#ifdef ENABLE_ACL
  // ATB attention implement for ascend device, for ATB run with graph, each layer need's one kernel implement instance
  std::shared_ptr<llm_kernels::ascend::ATBAttention> atb_paged_attn_;
#endif
  bool enable_blocked_multi_token_forwarding_kv_;
#ifdef ENABLE_CUDA
  bool use_flashinfer_for_decode_;
  bool use_hpc_attention_ = false;
  // Shared FlashInfer workspaces across all layers on the same rank.
  void* shared_pinned_host_workspace_ = nullptr;
  std::shared_ptr<Tensor> shared_device_workspace_;
  // Shared FlashInfer prefill helper across all layers on the same rank.
  std::shared_ptr<void> flashinfer_prefill_helper_;
#endif
};

}  // namespace ksana_llm
