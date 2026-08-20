/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/flash_attention_layer.h"

#include <algorithm>
#include <cstring>
#include <type_traits>
#include <vector>

#include "device/device_utils.h"
#include "device/kunlun/kunlun_device_context_impl.h"
#include "kernels/kunlun/attention_kernel_wrapper.h"
#include "layers/attention_layer.h"  // DISPATCH_BY_DTYPE_AND_KVTYPE

namespace ksana_llm {

Status FlashAttentionLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                 std::shared_ptr<Context> context, int rank) {
  AttentionLayer::Init(parameters, runtime_config, context, rank);
  return Status();
}

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
Status FlashAttentionLayer::ForwardT(const KunlunForwardParams& p, Tensor* output) {
  using T = SCALAR_T;
  // KunlunForwardParams fields (mirrors AclForwardParams + QK Norm weights + qlod):
  //   p.qkv                  : qkv_tensor shape [total_token_num, (num_heads + 2*num_kv_heads) * head_size]
  //   p.rotary_embedding_pos : shape [total_token_num], type int64_t
  //   p.layers_slot_mapping   : shape [layer_num * slot_mapping_dim_1], type int32_t
  //   p.k_cache_blocks_base   : k_cache shape [max_block_num, block_token_num, num_kv_heads, head_size]
  //   p.v_cache_blocks_base   : v_cache shape [max_block_num, block_token_num, num_kv_heads, head_size]
  //   p.seq_len_host          : shape [batch_size], type int32_t
  //   p.batch_size / p.max_tokens / p.layer_block_num: scalars from forward_shape
  //     batch_size corresponds to forward_shape[7] = dp_multi_token_request_num
  //     (CUDA 13D + Kunlun [13] layout, see forwarding_context.cpp UpdateBeforeForward)
  //   p.atb_attention_attr    : shape [2], content: 0: layers_slot_mapping_dim_1; 1: max_num_blocks_per_query
  //   p.query_layernorm_weight / p.key_layernorm_weight: optional QK norm weights (may be nullptr)
  //   p.qlod                  : HOST int32 cumulative q offsets [batch_size+1] (avoids per-layer D2H cumsum)
  // output:
  //   output[0]: flash_attention_output

  int32_t total_token_num = static_cast<int32_t>(p.qkv->shape[0]);
  int64_t batch_size = static_cast<int64_t>(p.batch_size);

  output->shape[0] = total_token_num;
  output->shape[1] = this->num_heads_ * this->head_size_;
  output->dtype = p.qkv->dtype;

  namespace xpukernel = xpytorch::xpu::api;
  void* stream = context_->device->GetPrimaryComputeStream().Get();
  xdnn::Context* xctx = static_cast<xdnn::Context*>(context_->device->ext->GetXdnnContext(stream));
  auto* xpuctx = static_cast<xpukernel::Context*>(context_->device->ext->GetXpuKernelContext(stream));

  T* qkv_ptr = p.qkv->GetPtr<T>();
  int64_t* positions = p.rotary_embedding_pos->GetPtr<int64_t>();
  T* k_cache = p.k_cache_blocks_base->GetPtr<T>();
  T* v_cache = p.v_cache_blocks_base->GetPtr<T>();
  int32_t* seq_len_host = p.seq_len_host->GetPtr<int32_t>();
  T* output_ptr = output->GetPtr<T>();

  uint64_t* atb_attention_attr_ptr = p.atb_attention_attr->GetPtr<uint64_t>();
  uint64_t slot_mapping_dim_1 = atb_attention_attr_ptr[0];
  int32_t* slot_mapping = p.layers_slot_mapping->GetPtr<int32_t>() + this->layer_index_ * slot_mapping_dim_1;

  int q_dim = num_heads_ * head_size_;
  int kv_dim = num_kv_heads_ * head_size_;
  int qkv_stride = q_dim + 2 * kv_dim;

  size_t q_bytes = static_cast<size_t>(total_token_num) * q_dim * sizeof(T);
  size_t kv_bytes = static_cast<size_t>(total_token_num) * kv_dim * sizeof(T);

  T* workspace = workspace_buffer_->GetPtr<T>();
  T* q_cont = workspace;
  T* k_cont = reinterpret_cast<T*>(reinterpret_cast<char*>(workspace) + q_bytes);
  T* v_cont = reinterpret_cast<T*>(reinterpret_cast<char*>(workspace) + q_bytes + kv_bytes);

  // Fused path: split(QKV) + RMSNorm(Q,K) + RoPE(Q,K) in one kernel when qk_norm enabled.
  // Norm weights are kept in original dtype (T) — no fp32 cast needed.
  bool use_fused_split_norm_rope = false;
  if constexpr (std::is_same_v<T, float16> || std::is_same_v<T, bfloat16>) {
    if ((use_qk_norm_ || enable_qk_pre_norm_before_rotary_pos_) && cos_sin_cache_neox_ptr_ != nullptr && !no_rope_ &&
        p.query_layernorm_weight != nullptr && p.key_layernorm_weight != nullptr) {
      const T* q_norm_weight = p.query_layernorm_weight->GetPtr<T>(/*check_nullptr=*/false);
      const T* k_norm_weight = p.key_layernorm_weight->GetPtr<T>(/*check_nullptr=*/false);
      if (q_norm_weight != nullptr && k_norm_weight != nullptr) {
        int fused_ret = XpuSplitNormRopeNeox<T>(
            xctx, qkv_ptr, reinterpret_cast<const T*>(cos_sin_cache_neox_ptr_), q_norm_weight, k_norm_weight, positions,
            q_cont, k_cont, v_cont, static_cast<int64_t>(total_token_num),
            static_cast<int64_t>(max_position_embeddings_), static_cast<int64_t>(num_heads_),
            static_cast<int64_t>(num_kv_heads_), static_cast<int64_t>(head_size_), static_cast<int64_t>(head_size_),
            layernorm_eps_);
        if (fused_ret == 0) {
          use_fused_split_norm_rope = true;
        } else {
          KLLM_LOG_WARNING << "FlashAttentionLayer: fused split_norm_rope_neox failed (ret=" << fused_ret
                           << "), falling back to separate kernels";
        }
      }
    }

    // Models WITHOUT QK-Norm: still fuse split(QKV)+RoPE into a single kernel.
    // Replaces 3x copy2d + 1x rotary_embedding_neox with one launch.
    if (!use_fused_split_norm_rope && !split_rope_fusion_failed_ && IsSplitRopeFusionEnabled() && !use_qk_norm_ &&
        !enable_qk_pre_norm_before_rotary_pos_ && !no_rope_ && cos_sin_cache_ptr_ != nullptr) {
      int fused_ret = XpuSplitRopeNeox<T>(xctx, qkv_ptr, reinterpret_cast<const T*>(cos_sin_cache_ptr_), positions,
                                          q_cont, k_cont, v_cont, static_cast<int64_t>(total_token_num),
                                          static_cast<int64_t>(max_position_embeddings_),
                                          static_cast<int64_t>(num_heads_), static_cast<int64_t>(num_kv_heads_),
                                          static_cast<int64_t>(head_size_), static_cast<int64_t>(head_size_));
      if (fused_ret == 0) {
        use_fused_split_norm_rope = true;
      } else {
        split_rope_fusion_failed_ = true;
        KLLM_LOG_WARNING << "FlashAttentionLayer: fused split_rotary_embedding failed (ret=" << fused_ret
                         << "), permanently falling back to separate kernels for this layer";
      }
    }
  }

  if (!use_fused_split_norm_rope) {
    // 1. QKV split: split the packed [total_token_num, qkv_stride] tensor into
    //    contiguous Q/K/V outputs along axis=1 (one launch, equivalent to the
    //    3x XpuCopy2d above). XpuSplit is the canonical Kunlun split used in
    //    SplitLayer; split_list sums to qkv_stride (= q_dim + 2*kv_dim).
    if (XpuSplit<T>(xpuctx, qkv_ptr, {q_cont, k_cont, v_cont},
                    {static_cast<int64_t>(total_token_num), static_cast<int64_t>(qkv_stride)},
                    {static_cast<int64_t>(q_dim), static_cast<int64_t>(kv_dim), static_cast<int64_t>(kv_dim)},
                    /*axis=*/1) != 0) {
      return Status(RET_RUNTIME_FAILED, "split QKV failed");
    }

    // 1.5 QK Norm: apply RMSNorm per-head to Q and K before RoPE.
    // Norm weights are kept in original dtype (T) — no fp32 cast on load (load_attn_norm just
    // MoveToDevice), so read them as T and use the TW=T kernel instantiation. This matches the
    // fused split_norm_rope path above and the decode path in paged_attention_layer.cpp; the
    // XpuRmsLayerNorm kernel only provides TW==T instantiations (no fp32-weight fallback).
    if ((use_qk_norm_ || enable_qk_pre_norm_before_rotary_pos_) && p.query_layernorm_weight != nullptr &&
        p.key_layernorm_weight != nullptr) {
      const T* q_norm = p.query_layernorm_weight->GetPtr<T>(/*check_nullptr=*/false);
      const T* k_norm = p.key_layernorm_weight->GetPtr<T>(/*check_nullptr=*/false);
      if (q_norm != nullptr && k_norm != nullptr) {
        int64_t q_rows = static_cast<int64_t>(total_token_num) * num_heads_;
        int64_t k_rows = static_cast<int64_t>(total_token_num) * num_kv_heads_;
        int64_t cols = head_size_;
        XpuRmsLayerNorm<T, T, T>(xctx, q_cont, q_cont, q_rows, cols, layernorm_eps_, q_norm, /*bias=*/nullptr,
                                 /*var=*/nullptr);
        XpuRmsLayerNorm<T, T, T>(xctx, k_cont, k_cont, k_rows, cols, layernorm_eps_, k_norm, /*bias=*/nullptr,
                                 /*var=*/nullptr);
      } else {
        KLLM_LOG_WARNING << "FlashAttentionLayer: QK norm enabled but weights are null";
      }
    }

    // 2. RoPE: infer_ops::rotary_embedding_neox (fp16+bf16 device-side).
    //    cos_sin_cache stores [max_pos, rot_dim] with [cos_half | sin_half] in T.
    if constexpr (std::is_same_v<T, float16> || std::is_same_v<T, bfloat16>) {
      if (!no_rope_ && cos_sin_cache_ptr_ != nullptr) {
        const int rot_dim = head_size_;
        int rope_ret =
            XpuRotaryEmbeddingNeox<T>(xctx, positions, q_cont, k_cont, reinterpret_cast<const T*>(cos_sin_cache_ptr_),
                                      rot_dim, num_heads_, num_kv_heads_, head_size_, total_token_num);
        KLLM_CHECK_WITH_INFO(rope_ret == 0,
            fmt::format("FlashAttentionLayer: XpuRotaryEmbeddingNeox failed, ret={}", rope_ret));
      }
    }
  }

  // 3. Flash Attention
  StreamSynchronize(context_->device->GetPrimaryComputeStream());
  // Reuse the xdnn Context already obtained at function top (xctx): GetXdnnContext on the
  // primary compute stream returns the same handle, so the previous separate xctx_attn was
  // a redundant re-fetch.

  // Use pre-computed qlod host tensor (p.qlod), same pattern as seq_len_host (HOST Tensor).
  int32_t* qlod_ptr = p.qlod->GetPtr<int32_t>();
  int qlod_len = static_cast<int>(p.qlod->shape[0]);
  int32_t* kvlod_ptr = qlod_ptr;
  int kvlod_len = qlod_len;

  int32_t max_seq_len = *std::max_element(seq_len_host, seq_len_host + batch_size);
  // alpha = 1.0 because the xfa kernel internally applies 1/sqrt(head_dim).
  // See history note in xsglang reference: passing 1/sqrt(d) here would double-scale.
  float softmax_scale = 1.0f;

  int ret = XpuFlashAttentionContext<T, T, int32_t>(
      xctx, q_cont, k_cont, v_cont,
      /*k_cache=*/nullptr, /*v_cache=*/nullptr, output_ptr, batch_size, max_seq_len, max_seq_len, /*max_seq_cache=*/0,
      num_heads_, num_kv_heads_, head_size_, is_causal_, softmax_scale, qlod_ptr, qlod_len, kvlod_ptr, kvlod_len,
      /*cachelod=*/nullptr, /*cachelod_len=*/0,
      /*block_table=*/nullptr, /*block_size=*/-1, /*max_blocks_per_seq=*/-1);
  if (ret != 0) {
    KLLM_LOG_ERROR << "FlashAttentionLayer: XpuFlashAttentionContext failed, ret=" << ret;
    return Status(RET_RUNTIME_FAILED, "XpuFlashAttentionContext failed");
  }

  // 4. KV cache scatter
  {
    int64_t num_blocks = p.k_cache_blocks_base->shape[0];
    int kv_ret = XpuReshapeCachedKV<T>(
        xctx, k_cont, v_cont, k_cache, v_cache, slot_mapping, static_cast<int64_t>(total_token_num), num_blocks,
        static_cast<int64_t>(num_kv_heads_), static_cast<int64_t>(head_size_), static_cast<int64_t>(block_token_num_));
    if (kv_ret != 0) {
      KLLM_LOG_ERROR << "FlashAttentionLayer: reshape_cached_kv failed, ret=" << kv_ret;
    }
  }
  return Status();
}

Status FlashAttentionLayer::Forward(const KunlunForwardParams& p, Tensor* output, const LayerExecContext& exec_ctx) {
  DISPATCH_BY_DTYPE_AND_KVTYPE(inter_data_type_, kv_cache_dtype_, ForwardT, p, output);
}

// Explicit instantiations for fp16 and bf16. KV_DTYPE=kAuto is the only path on Kunlun.
template Status FlashAttentionLayer::ForwardT<float16, float16, llm_kernels::utils::KVCacheType::kAuto>(
    const KunlunForwardParams&, Tensor*);
template Status FlashAttentionLayer::ForwardT<bfloat16, bfloat16, llm_kernels::utils::KVCacheType::kAuto>(
    const KunlunForwardParams&, Tensor*);

}  // namespace ksana_llm
