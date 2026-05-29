/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "modules/attention/compressor.h"

#include <fmt/core.h>

#include "layers/cast_layer.h"
#include "layers/compressor_layer.h"
#include "layers/nvidia/dsv4_tail_copy.h"
#include "profiler/profile_event.h"
#include "utils/logger.h"

namespace ksana_llm {

namespace {
// Compute the local layer index for a given global layer index, taking pipeline
// parallelism (lower/upper_layer_idx) and MTP layers into account. This mirrors
// dsv4_mqa.cpp's attn_config.idx assignment so that per-layer tensors (which are
// populated by local index) can be indexed correctly inside Compressor.
int ComputeLocalLayerIdx(int global_layer_idx, const PipelineConfig& pipeline_config, int num_layer) {
  int local_layer_idx = global_layer_idx - pipeline_config.lower_layer_idx;
  if (global_layer_idx >= num_layer) {
    // MTP / next-N layer: place after the main layers on this node.
    local_layer_idx =
        pipeline_config.upper_layer_idx - pipeline_config.lower_layer_idx + global_layer_idx - num_layer + 1;
  }
  return local_layer_idx;
}
}  // namespace

Compressor::Compressor(int layer_idx, bool is_in_indexer, const LayerCreationContext& creation_context,
                       const ModelCreationConfig& model_creation_config)
    : layer_idx_(layer_idx),
      local_layer_idx_(
          ComputeLocalLayerIdx(layer_idx, creation_context.pipeline_config,
                               static_cast<int>(model_creation_config.attn_config.model_config.num_layer))),
      is_in_indexer_(is_in_indexer) {
  const auto& attn_config = model_creation_config.attn_config;
  const auto& model_config = attn_config.model_config;
  const auto& runtime_config = creation_context.runtime_config;

  KLLM_CHECK_WITH_INFO(
      layer_idx >= 0 && layer_idx < static_cast<int>(model_config.dsv4_attn_config.compress_ratios.size()),
      "layer_idx out of bounds for compress_ratios array");
  compress_ratio_ = model_config.dsv4_attn_config.compress_ratios[layer_idx];
  coff_ = 1 + (compress_ratio_ == 4);
  KLLM_CHECK_WITH_INFO(compress_ratio_ > 0, "Compressor requires compress_ratio > 0");
  KLLM_CHECK_WITH_INFO(!is_in_indexer || compress_ratio_ == 4, "Indexer requires compress_ratio == 4");
  const int n_heads = is_in_indexer ? model_config.dsa_config.index_n_heads
                                    : attn_config.head_num_per_tp;  // Required to be passed but not actually used
  head_dim_ = is_in_indexer ? model_config.dsa_config.index_head_dim : attn_config.size_per_head;

  const std::string layer_prefix = is_in_indexer
                                       ? fmt::format("model.layers.{}.self_attn.indexer.compressor", layer_idx)
                                       : fmt::format("model.layers.{}.self_attn.compressor", layer_idx);
  const auto& linear_compute_backend = model_config.quant_config.backend;
  const float layernorm_eps = model_creation_config.layernorm_config.layernorm_eps;

  // ── Weights ──────────────────────────────────────────────────────────────────
  // wkv_gate: [hidden_size → 2 * coff * head_dim]
  wkv_gate_proj_ =
      std::make_unique<Linear>(layer_prefix + ".wkv_gate.weight", creation_context, linear_compute_backend);

  // ape: [ratio, coff * head_dim]
  ape_weight_ = creation_context.base_weight->GetModelWeights(layer_prefix + ".ape");

  // norm weight tensor (passed directly to CompressorLayer::Forward as input)
  norm_weight_ = creation_context.base_weight->GetModelWeights(layer_prefix + ".norm.weight");

  // ── CastLayer initialisation (bf16 → fp32 for kv_score_input) ──────────
  cast_layer_ = std::make_unique<CastLayer>();
  cast_layer_->Init({}, runtime_config, creation_context.context, creation_context.rank);
  // CastLayer expects input_tensors[1] as a shape tensor: {output_offset_bytes, vocab_size, vocab_size_pad}.
  // For a simple element-wise cast with no offset or stride, use {0, 1, 1}.
  cast_shape_tensor_.shape = {0, 1, 1};

  // ── CompressorLayer initialisation ───────────────────────────────────────────
  compressor_layer_ = std::make_unique<CompressorLayer>();

  std::vector<std::any> params;
  params.push_back(compress_ratio_);                                                       // compress_ratio
  params.push_back(n_heads);                                                               // n_heads
  params.push_back(static_cast<int>(head_dim_));                                           // head_dim
  params.push_back(static_cast<int>(model_config.dsv4_attn_config.qk_rope_head_dim));      // rope_head_dim
  params.push_back(layernorm_eps);                                                         // rms_norm_eps
  params.push_back(attn_config.idx);                                                       // local_layer_idx
  params.push_back(attn_config.position_encoding);                                         // position_encoding
  params.push_back(attn_config.compress_rope_theta);                                       // compress_rope_theta
  params.push_back(attn_config.compress_cos_sin_cache_ptr);                                // freqs_cis
  params.push_back(model_config.rope_scaling_factor_config);                               // rope_scaling
  params.push_back(attn_config.max_position_embeddings);                                   // max_position_embeddings
  params.push_back(static_cast<int>(runtime_config.attn_backend_config.block_token_num));  // block_token_num
  params.push_back(static_cast<int64_t>(runtime_config.attn_backend_config.block_size));   // page_stride
  params.push_back(attn_config.kv_cache_dtype);                                            // kv_cache_dtype (extra)

  compressor_layer_->Init(params, runtime_config, creation_context.context, creation_context.rank);
}

// ─────────────────────────────────────────────────────────────────────────────
// Forward: linear projection + compress
// ─────────────────────────────────────────────────────────────────────────────

Status Compressor::Forward(const Tensor& x, const Tensor& workspace, Tensor& kv_compressed_output,
                           ForwardingContext& forwarding_context) {
  const int rank = forwarding_context.GetCurrentRank();
  PROFILE_EVENT_SCOPE(compressor_forward, "compressor_forward", rank);

  // x.shape = [total_tokens, hidden_dim]
  const size_t total_tokens = x.shape[0];
  // Used to locate unused workspace buffers
  // Assume the given workspace is sufficient since n_heads and head_dim in compressor is small
  size_t workspace_offset = 0;

  // Step 1: kv_score = wkv_gate_proj_(x)
  // Input: x [T, hidden_dim], Output: kv_score [T, 2 * coff * head_dim]
  Tensor kv_score_input = workspace.GetView({total_tokens, 2 * coff_ * head_dim_}, workspace_offset);
  workspace_offset += kv_score_input.GetElementNumber();
  std::vector<Tensor> wkv_gate_proj_outputs = {kv_score_input};
  STATUS_CHECK_RETURN(wkv_gate_proj_->Forward(x, wkv_gate_proj_outputs));

  // Step 2: Convert kv_score from bf16 to fp32
  KLLM_CHECK(workspace.dtype == wkv_gate_proj_outputs[0].dtype);
  std::vector<Tensor> fp32_cast_outputs = {
      workspace.GetView({kv_score_input.GetElementNumber() * 2 /*for bf16 to fp32*/}, workspace_offset)};
  STATUS_CHECK_RETURN(cast_layer_->Forward({wkv_gate_proj_outputs[0], cast_shape_tensor_}, fp32_cast_outputs));
  kv_score_input = fp32_cast_outputs[0];

  // Step 3: kv_compressed_output = compress_layer_(kv_score_input)
  // Input: kv_score_input [T, 2 * coff * head_dim] FP32, Output: kv_compressed_output [T, head_dim]
  const auto& compressor_meta = forwarding_context.GetModelInput()->compressor_meta_;
  const auto& model_input = forwarding_context.GetModelInput();
  const size_t dp_context_tokens = compressor_meta->dp_context_tokens;
  const size_t dp_decode_tokens = compressor_meta->dp_decode_tokens;

  // Prefix cache tail read-copy: 把 prefix 末 block 的 tail kv_score 复制到 state buffer。
  if (compress_ratio_ == 4) {
    if (const int num_req = static_cast<int>(compressor_meta->tail_read_block_tensor.GetElementNumber()); num_req > 0) {
      const auto& dsv4_mqa_meta = forwarding_context.GetModelInput()->dsv4_mqa_meta_;
      Tensor* tail_layer_base_tensor = is_in_indexer_ ? &dsv4_mqa_meta->layer_tail_indexer_cache_ptr_tensor
                                                      : &dsv4_mqa_meta->layer_tail_attn_cache_ptr_tensor;
      void* const tail_layer_base = tail_layer_base_tensor->GetPtr<void*>()[1 + local_layer_idx_];
      Tensor& state_layer_ptr_tensor = is_in_indexer_ ? compressor_meta->layer_c4_indexer_state_ptr_tensor
                                                      : compressor_meta->layer_c4_state_ptr_tensor;
      void* const state_layer_base = state_layer_ptr_tensor.GetPtr<void*>()[local_layer_idx_];
      const int per_token_bytes = static_cast<int>(2 * coff_ * head_dim_ * sizeof(float));
      const int64_t block_cache_size = static_cast<int64_t>(dsv4_mqa_meta->GetSwaBlockSize());
      constexpr int kTailTokens = 4;
      KLLM_LOG_DEBUG << fmt::format(
          "tail read-copy: layer={} is_in_indexer={} num_req={} per_token_bytes={} block_cache_size={}", layer_idx_,
          is_in_indexer_, num_req, per_token_bytes, block_cache_size);
      CopyTailToState(state_layer_base, tail_layer_base, compressor_meta->tail_read_block_tensor.GetPtr<int32_t>(),
                      compressor_meta->tail_target_indices_tensor.GetPtr<int32_t>(), num_req, block_cache_size,
                      per_token_bytes, kTailTokens,
                      forwarding_context.GetContext()->device->GetPrimaryComputeStream().Get());
    }
  }

  if (dp_context_tokens > 0) {
    Tensor context_kv_score_input = kv_score_input.GetView({dp_context_tokens, kv_score_input.shape[1]});
    std::vector<Tensor> context_outputs = {kv_compressed_output.GetView({dp_context_tokens, head_dim_})};
    if (compress_ratio_ == 4) {
      // ForwardPrefillC4
      compressor_layer_->Forward({is_in_indexer_ ? compressor_meta->layer_c4_indexer_state_ptr_tensor
                                                 : compressor_meta->layer_c4_state_ptr_tensor,  // kv_score_buffer
                                  context_kv_score_input,                                       // kv_score_input
                                  ape_weight_,                                                  // score_bias
                                  compressor_meta->flash_input.c4_indices_tensor,               // indices
                                  compressor_meta->flash_input.c4_indices_bundle_tensor,        // extra
                                  compressor_meta->c4_compress_plan_tensor,                     // compress_plan
                                  compressor_meta->c4_write_plan_tensor,                        // write_plan
                                  compressor_meta->flash_input.seq_lens_tensor,                 // seq_lens
                                  compressor_meta->flash_input.extend_lens_tensor,              // extend_lens
                                  norm_weight_,                                                 // norm_weight
                                  is_in_indexer_ ? model_input->layer_indexer_kv_cache_ptr
                                                 : model_input->layer_kv_cache_ptr,  // layer_kv_cache_ptr
                                  compressor_meta->flash_input.c4_out_loc_tensor},   // compressed_out_loc
                                 context_outputs);
      compressor_meta->c4_compress_plan_tensor.shape[2] = 1;  // Mark meta as initialized
      compressor_meta->c4_write_plan_tensor.shape[2] = 1;     // Mark meta as initialized
    } else {
      // ForwardPrefillC128 (is_in_indexer_ never true for C128)
      compressor_layer_->Forward({compressor_meta->layer_c128_state_ptr_tensor,           // kv_score_buffer
                                  context_kv_score_input,                                 // kv_score_input
                                  ape_weight_,                                            // score_bias
                                  compressor_meta->flash_input.c128_indices_tensor,       // indices
                                  compressor_meta->flash_input.c128_load_indices_tensor,  // load_indices
                                  compressor_meta->c128_compress_plan_tensor,             // compress_plan
                                  compressor_meta->c128_write_plan_tensor,                // write_plan
                                  compressor_meta->flash_input.seq_lens_tensor,           // seq_lens
                                  compressor_meta->flash_input.extend_lens_tensor,        // extend_lens
                                  norm_weight_,                                           // norm_weight
                                  model_input->layer_kv_cache_ptr,                        // layer_kv_cache_ptr
                                  compressor_meta->flash_input.c128_out_loc_tensor},      // compressed_out_loc
                                 context_outputs);
      compressor_meta->c128_compress_plan_tensor.shape[2] = 1;  // Mark meta as initialized
      compressor_meta->c128_write_plan_tensor.shape[2] = 1;     // Mark meta as initialized
    }
  }

  if (dp_decode_tokens > 0) {
    Tensor decode_kv_score_input = kv_score_input.GetView({dp_decode_tokens, kv_score_input.shape[1]},
                                                          dp_context_tokens * kv_score_input.shape[1]);
    std::vector<Tensor> decode_outputs = {
        kv_compressed_output.GetView({dp_decode_tokens, head_dim_}, dp_context_tokens * head_dim_)};
    if (compress_ratio_ == 4) {
      // ForwardDecodeC4
      compressor_layer_->Forward({is_in_indexer_ ? compressor_meta->layer_c4_indexer_state_ptr_tensor
                                                 : compressor_meta->layer_c4_state_ptr_tensor,  // kv_score_buffer
                                  decode_kv_score_input,                                        // kv_score_input
                                  ape_weight_,                                                  // score_bias
                                  compressor_meta->page_input.c4_indices_tensor,                // indices
                                  compressor_meta->page_input.seq_lens_tensor,                  // seq_lens
                                  compressor_meta->page_input.c4_indices_prev_tensor,           // extra (prev_indices)
                                  norm_weight_,                                                 // norm_weight
                                  is_in_indexer_ ? model_input->layer_indexer_kv_cache_ptr
                                                 : model_input->layer_kv_cache_ptr,  // layer_kv_cache_ptr
                                  compressor_meta->page_input.c4_out_loc_tensor},    // compressed_out_loc
                                 decode_outputs);
    } else {
      // ForwardDecodeC128 (is_in_indexer_ never true for C128)
      compressor_layer_->Forward({compressor_meta->layer_c128_state_ptr_tensor,      // kv_score_buffer
                                  decode_kv_score_input,                             // kv_score_input
                                  ape_weight_,                                       // score_bias
                                  compressor_meta->page_input.c128_indices_tensor,   // indices
                                  compressor_meta->page_input.seq_lens_tensor,       // seq_lens
                                  norm_weight_,                                      // norm_weight
                                  model_input->layer_kv_cache_ptr,                   // layer_kv_cache_ptr
                                  compressor_meta->page_input.c128_out_loc_tensor},  // compressed_out_loc
                                 decode_outputs);
    }
  }

  // Prefix cache write-copy: 把 c4 层当前 step 的 kv_score 持久化到 tail 区，供下个请求复用。
  if (compress_ratio_ == 4) {
    if (const size_t num_tokens = compressor_meta->tail_write_indices_tensor.GetElementNumber(); num_tokens > 0) {
      KLLM_CHECK(num_tokens == total_tokens);
      const auto& dsv4_mqa_meta = forwarding_context.GetModelInput()->dsv4_mqa_meta_;
      Tensor* tail_layer_base_tensor = is_in_indexer_ ? &dsv4_mqa_meta->layer_tail_indexer_cache_ptr_tensor
                                                      : &dsv4_mqa_meta->layer_tail_attn_cache_ptr_tensor;
      // layer_tail_*_cache_ptr_tensor layout: [block_num(int64) | layer_0_base | layer_1_base | ...]
      // Slots 1..N correspond to local layer 0..N-1, so local_layer_idx_ must be used here.
      void* const tail_layer_base = tail_layer_base_tensor->GetPtr<void*>()[1 + local_layer_idx_];
      // Per-token bytes match a single row of kv_score_input.
      const int per_token_bytes = static_cast<int>(2 * coff_ * head_dim_ * sizeof(float));
      // tail 写入的目标位于 SWA pool, stride 使用 SWA pool 的 block_size。
      const int64_t block_cache_size = static_cast<int64_t>(dsv4_mqa_meta->GetSwaBlockSize());
      KLLM_LOG_DEBUG << fmt::format(
          "tail write-copy: layer={} is_in_indexer={} num_tokens={} per_token_bytes={} block_cache_size={}", layer_idx_,
          is_in_indexer_, num_tokens, per_token_bytes, block_cache_size);
      CopyKvScoreToTail(tail_layer_base,
                        kv_score_input.GetPtr<void>(),  // FP32 [total_tokens, 2*coff*head_dim]
                        compressor_meta->tail_write_indices_tensor.GetPtr<int32_t>(), static_cast<int>(num_tokens),
                        block_cache_size, per_token_bytes,
                        forwarding_context.GetContext()->device->GetPrimaryComputeStream().Get());
    }
  }

  // Correctly set the output shape
  kv_compressed_output.shape = {total_tokens, head_dim_};  // Uncompressed tokens remain in place

  KLLM_LOG_DEBUG << fmt::format("Compressor Forward completed: layer={}", layer_idx_);

  return Status();
}

}  // namespace ksana_llm
