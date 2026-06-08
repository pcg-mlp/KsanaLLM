/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include <memory>
#include <vector>

#include "layers/flash_dsv4_attention_layer.h"
#include "layers/paged_dsv4_attention_layer.h"
#include "models/base/buffer_manager.h"
#include "models/base/forwarding_context.h"
#include "models/base/layer_creation_context.h"
#include "models/deepseek_v4/attention_meta/deepseek_v4_attention_meta.h"
#include "modules/attention/c4_indexer.h"
#include "modules/attention/compressor.h"
#include "modules/basic/bmm.h"
#include "modules/basic/layernorm.h"
#include "modules/basic/linear.h"
#include "utils/status.h"

namespace ksana_llm {

// Buffers used in DSV4 MQA attention.
// Pre-allocated once during model creation and shared across all DSV4 MQA layers.
struct Dsv4AttnBuffer {
  TensorBuffer* kv_compressed_output_buffer;  // [max_token_num, head_dim]
  TensorBuffer* q_indexer_buffer;             // [max_token_num, index_n_heads, index_head_dim]
};

/*
 * Dsv4Mqa: Multi-Query Attention module for DeepSeek V4.
 *
 * Computation graph:
 *   hidden --+--> wqkv_a --> split --+--> q_norm --> wq_b --> q_post --> RoPE --+
 *            |                       |                                          |
 *            |                       +--> kv_norm --> RoPE ---------------------+
 *            |                                                                  |
 *            +--> [compress_ratio!=0] compressor --> kv_compressed              |
 *            +--> [compress_ratio==4] c4_indexer --> topk_indices               |
 *                                                                               |
 *                                                                               v
 *                                                                           MQA attn
 *                                                                               |
 *                                                                               v
 *                                                                           inv-RoPE
 *                                                                               |
 *                                                                               v
 *                                                                      reshape [T, G, D/G]
 *                                                                               |
 *                                                                               v
 *                                                                       wo_a (group Bmm)
 *                                                                               |
 *                                                                               v
 *                                                                             wo_b
 */
class Dsv4Mqa {
 public:
  Dsv4Mqa(int layer_idx, bool is_neox, bool add_qkv_bias, bool use_qk_norm, LayerCreationContext& creation_context,
          ModelCreationConfig& model_creation_config, Dsv4AttnBuffer& dsv4_attn_buffer);
  ~Dsv4Mqa() = default;

  Status Forward(std::vector<Tensor>& hidden_buffer_tensors_0, std::vector<Tensor>& reduce_buffer_tensors,
                 ForwardingContext& forwarding_context);

  // Pre-allocate dsv4_attn_buffer. Called once during model creation.
  static Status CreateBuffers(BufferManager* buffer_mgr, const AttentionCreationConfig& attn_config,
                              const RuntimeConfig& runtime_config, Dsv4AttnBuffer& dsv4_attn_buffer);

 private:
  template <typename LayerT>
  Status RunDsv4Attention(LayerT* attention_layer, StandardAttentionMeta& model_input_info,
                          MqaAttentionMeta& mqa_input_info, Tensor& q_tensor, Tensor& kv_tensor, Tensor& attn_output,
                          Tensor& topk_indices_tensor, Tensor& layer_kv_cache_ptr,
                          Tensor& layer_swa_kv_cache_ptr_tensor, size_t token_count, size_t token_offset,
                          size_t q_stride, size_t kv_stride);

  // ── Q path ──────────────────────────────────────────────────────────────────
  // wqkv_a: [hidden_size → q_lora_rank + head_dim], replicated
  std::unique_ptr<Linear> wqkv_a_proj_;
  // Parallel q_norm + kv_norm: strided RMSNorm on q_lora_rank and head_dim
  std::unique_ptr<Layernorm> qkv_a_norm_;
  // wq_b: [q_lora_rank → n_heads * head_dim], TP column-parallel
  std::unique_ptr<Linear> wq_b_proj_;

  // ── KV path ─────────────────────────────────────────────────────────────────
  // (kv_norm is fused into qkv_a_norm_ above)

  // ── Attention ────────────────────────────────────────────────────────────────
  // Flash DSV4 attention (prefill/context phase)
  std::unique_ptr<FlashDsv4AttentionLayer> flash_dsv4_attention_layer_;
  // Paged DSV4 attention (decode phase)
  std::unique_ptr<PagedDsv4AttentionLayer> paged_dsv4_attention_layer_;
  // Attention sink weight: [n_local_heads, head_dim]
  Tensor attn_sink_weight_;

  // ── Output path ─────────────────────────────────────────────────────────────
  // wo_a: group-wise Bmm, einsum "tgd,grd->tgr"
  // weight shape [n_local_groups, o_lora_rank, head_dim * n_local_heads / n_local_groups]
  std::unique_ptr<Bmm> wo_a_bmm_;
  // wo_b: [o_groups * o_lora_rank → hidden_size], TP row-parallel
  std::unique_ptr<Linear> wo_b_proj_;

  // ── Flags & dimensions ───────────────────────────────────────────────────────
  int layer_idx_;
  int local_layer_idx_;  // = attn_config.idx (0-based within this pipeline node); used to look up
                         //   per-layer pool offset for SWA buffer pre/post copy.
  // 0 = no compression, 4 = C4 sparse, 128 = C128 compression
  uint32_t compress_ratio_;

  // Dimension fields derived from model config at construction time
  uint32_t head_dim_;      // qk_nope_head_dim + qk_rope_head_dim
  uint32_t q_lora_rank_;   // LoRA rank for Q projection
  uint32_t o_lora_rank_;   // LoRA rank for output projection
  size_t n_local_heads_;   // head_num / attn_tensor_parallel_size
  size_t o_local_groups_;  // o_groups / attn_tensor_parallel_size

  // ── Shared pre-allocated buffers ────────────────────────────────────────────
  Dsv4AttnBuffer& dsv4_attn_buffer_;

  // ── Compressor / C4Indexer ──────────────────────────────────────────────────
  // Non-null when compress_ratio_ != 0.
  std::unique_ptr<Compressor> compressor_;
  // Non-null when compress_ratio_ == 4.
  std::unique_ptr<C4Indexer> c4_indexer_;
};

}  // namespace ksana_llm
