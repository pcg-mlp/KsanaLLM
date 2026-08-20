/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/paged_attention_layer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <type_traits>
#include <vector>

#include "device/device_utils.h"
#include "device/kunlun/kunlun_device_context_impl.h"
#include "kernels/kunlun/attention_kernel_wrapper.h"
#include "layers/attention_layer.h"  // DISPATCH_BY_DTYPE_AND_KVTYPE

namespace ksana_llm {

Status PagedAttentionLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                 std::shared_ptr<Context> context, int rank) {
  AttentionLayer::Init(parameters, runtime_config, context, rank);
  return Status();
}

Status PagedAttentionLayer::Forward(const PagedAttentionInputs& inputs, std::vector<Tensor>& output_tensors) {
  DISPATCH_BY_DTYPE_AND_KVTYPE(inter_data_type_, kv_cache_dtype_, ForwardT, inputs, output_tensors);
}

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
Status PagedAttentionLayer::ForwardT(const PagedAttentionInputs& inputs, std::vector<Tensor>& output_tensors) {
  using T = SCALAR_T;
  // Rebuild the Ascend/ATB-compatible tensor vector from the typed struct so the body below is
  // unchanged. The two QK-Norm weights are appended only when present (size() > 10 gates their use).
  std::vector<Tensor> input_tensors = {
      *inputs.qkv_tensor,        *inputs.rotary_embedding_pos, *inputs.layers_slot_mapping,
      *inputs.layers_block_table, *inputs.k_cache_blocks_base,  *inputs.v_cache_blocks_base,
      *inputs.seq_len_host,       *inputs.forward_shape,        *inputs.atb_attention_attr};
  if (inputs.query_layernorm_weight != nullptr && inputs.key_layernorm_weight != nullptr) {
    input_tensors.push_back(*inputs.query_layernorm_weight);
    input_tensors.push_back(*inputs.key_layernorm_weight);
  }
  // input_tensors (与 Ascend 后端保持一致):
  //   0: qkv_tensor shape [batch_size, (num_heads + 2*num_kv_heads) * head_size]
  //   1: rotary_embedding_pos shape [batch_size], type int64_t
  //   2: slot_mapping shape [layer_num * slot_mapping_dim_1], type int32_t
  //   3: blocks_table shape [layer_num * batch_size * max_num_blocks_per_query], type int32_t
  //   4: k_cache shape [max_block_num, block_token_num, num_kv_heads, head_size]
  //   5: v_cache shape [max_block_num, block_token_num, num_kv_heads, head_size]
  //   6: seq_len_host shape [batch_size], type int32_t
  //   7: forward_shape (CUDA 13D + Kunlun [13]) — see forwarding_context.cpp.
  //      PagedAttentionLayer reads shape[9] = dp_single_token_request_num as batch_size.
  //   8: atb_attention_attr shape: [2], content: 0: layers_slot_mapping_dim_1; 1: max_num_blocks_per_query
  // output_tensors:
  //   0: paged_attention_output

  int64_t batch_size = input_tensors[7].shape[9];
  int64_t hidden_units = static_cast<int64_t>(this->num_heads_) * this->head_size_;

  output_tensors[0].shape = {static_cast<uint64_t>(batch_size), static_cast<uint64_t>(hidden_units)};
  output_tensors[0].dtype = input_tensors[0].dtype;

  namespace xpukernel = xpytorch::xpu::api;
  void* stream = context_->device->GetPrimaryComputeStream().Get();
  xdnn::Context* xctx = static_cast<xdnn::Context*>(context_->device->ext->GetXdnnContext(stream));
  auto* xpuctx = static_cast<xpukernel::Context*>(context_->device->ext->GetXpuKernelContext(stream));

  T* qkv_ptr = input_tensors[0].GetPtr<T>();
  int64_t* positions = input_tensors[1].GetPtr<int64_t>();
  T* k_cache = input_tensors[4].GetPtr<T>();
  T* v_cache = input_tensors[5].GetPtr<T>();
  int32_t* seq_len_host = input_tensors[6].GetPtr<int32_t>();
  T* output = output_tensors[0].GetPtr<T>();

  uint64_t* atb_attention_attr_ptr = input_tensors[8].GetPtr<uint64_t>();
  uint64_t slot_mapping_dim_1 = atb_attention_attr_ptr[0];
  uint64_t max_num_blocks_per_query = atb_attention_attr_ptr[1];

  int32_t* slot_mapping = input_tensors[2].GetPtr<int32_t>() + this->layer_index_ * slot_mapping_dim_1;
  int32_t* block_table =
      input_tensors[3].GetPtr<int32_t>() + this->layer_index_ * batch_size * max_num_blocks_per_query;

  int q_dim = num_heads_ * head_size_;
  int kv_dim = num_kv_heads_ * head_size_;
  int qkv_stride = q_dim + 2 * kv_dim;
  int32_t total_token_num = static_cast<int32_t>(batch_size);  // decode: 1 token per sequence

  size_t q_bytes = static_cast<size_t>(batch_size) * q_dim * sizeof(T);
  size_t kv_bytes = static_cast<size_t>(batch_size) * kv_dim * sizeof(T);
  T* workspace = workspace_buffer_->GetPtr<T>();
  T* q_cont = workspace;
  T* k_cont = reinterpret_cast<T*>(reinterpret_cast<char*>(workspace) + q_bytes);
  T* v_cont = reinterpret_cast<T*>(reinterpret_cast<char*>(workspace) + q_bytes + kv_bytes);

  // Fused path: split(QKV) + RMSNorm(Q,K) + RoPE(Q,K) in one kernel when qk_norm enabled.
  // Norm weights are kept in original dtype (T) — no fp32 cast needed.
  bool use_fused_split_norm_rope = false;
  if constexpr (std::is_same_v<T, float16> || std::is_same_v<T, bfloat16>) {
    if ((use_qk_norm_ || enable_qk_pre_norm_before_rotary_pos_) && cos_sin_cache_neox_ptr_ != nullptr && !no_rope_ &&
        input_tensors.size() > 10) {
      const T* q_norm_weight = input_tensors[9].GetPtr<T>(/*check_nullptr=*/false);
      const T* k_norm_weight = input_tensors[10].GetPtr<T>(/*check_nullptr=*/false);
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
          KLLM_LOG_WARNING << "PagedAttentionLayer: fused split_norm_rope_neox failed (ret=" << fused_ret
                           << "), falling back to separate kernels";
        }
      }
    }

    // Models WITHOUT QK-Norm: still fuse split(QKV)+RoPE into a single kernel.
    // Replaces 3x copy2d + 1x rotary_embedding_neox with one launch.
    // Skipped for total_token_num == 1, where the split below is already a
    // zero-copy pointer alias and fusing would only add a QKV round-trip.
    if (!use_fused_split_norm_rope && !split_rope_fusion_failed_ && IsSplitRopeFusionEnabled() && !use_qk_norm_ &&
        !enable_qk_pre_norm_before_rotary_pos_ && !no_rope_ && cos_sin_cache_ptr_ != nullptr && total_token_num > 1) {
      int fused_ret = XpuSplitRopeNeox<T>(xctx, qkv_ptr, reinterpret_cast<const T*>(cos_sin_cache_ptr_), positions,
                                          q_cont, k_cont, v_cont, static_cast<int64_t>(total_token_num),
                                          static_cast<int64_t>(max_position_embeddings_),
                                          static_cast<int64_t>(num_heads_), static_cast<int64_t>(num_kv_heads_),
                                          static_cast<int64_t>(head_size_), static_cast<int64_t>(head_size_));
      if (fused_ret == 0) {
        use_fused_split_norm_rope = true;
      } else {
        split_rope_fusion_failed_ = true;
        KLLM_LOG_WARNING << "PagedAttentionLayer: fused split_rotary_embedding failed (ret=" << fused_ret
                         << "), permanently falling back to separate kernels for this layer";
      }
    }
  }

  if (!use_fused_split_norm_rope) {
    // 1. QKV split
    //
    // total_token_num == 1: QKV 只有一行，[Q|K|V] 在 qkv_ptr 上已经天然连续，
    // split 退化为纯指针运算，直接别名即可省掉 3 次 copy2d launch。
    //
    // 注意：此时 q_cont/k_cont 别名到 input_tensors[0]，后续 1.5 的 QK-Norm 与
    // 2. 的 RoPE 都是 in-place，会就地改写调用方传入的 QKV buffer。这是安全的，
    // 依赖以下 buffer 协议（勿在不复核该协议的前提下让 attention 之后再读 QKV）：
    //   - QKV 由紧邻上游的 qkv_proj 写入 hidden_buffer_1，随后 swap 进 hidden_buffer_0
    //     (modules/attention/multihead_attention.cpp)；
    //   - 本层从 hidden_buffer_0 读 QKV、把 attention 输出写 hidden_buffer_1，返回前再 swap
    //     (modules/basic/paged_attention.cpp)，于是 QKV 那块内存降级为 scratch；
    //   - 紧接着 o_proj 直接把结果覆盖写进这块 scratch
    //     (modules/attention/common_attention.cpp)，原始 QKV 内容此后无人读取。
    //   - XPU Graph 模式下 capture 的是整个 LayerForward（含 qkv_proj GEMM），
    //     每次 replay 都会先重写该 buffer，就地修改对 replay 幂等。
    // residual 走独立的 local_residual_buffer_，与 hidden_buffer_0/1 无别名。
    if (total_token_num == 1) {
      q_cont = qkv_ptr;
      k_cont = qkv_ptr + q_dim;
      v_cont = qkv_ptr + q_dim + kv_dim;
    } else {
      // Split the packed [total_token_num, qkv_stride] QKV into contiguous Q/K/V
      // outputs along axis=1 — one launch, equivalent to the 3x XpuCopy2d above.
      // XpuSplit is the canonical Kunlun split (see SplitLayer); split_list sums to
      // qkv_stride (= q_dim + 2*kv_dim).
      int ret = XpuSplit<T>(xpuctx, qkv_ptr, {q_cont, k_cont, v_cont},
                            {static_cast<int64_t>(total_token_num), static_cast<int64_t>(qkv_stride)},
                            {static_cast<int64_t>(q_dim), static_cast<int64_t>(kv_dim), static_cast<int64_t>(kv_dim)},
                            /*axis=*/1);
      if (ret != 0) return Status(RET_RUNTIME_FAILED, "split QKV failed");
    }

    // 1.5 QK Norm
    if ((use_qk_norm_ || enable_qk_pre_norm_before_rotary_pos_) && input_tensors.size() > 10) {
      const T* q_norm = input_tensors[9].GetPtr<T>(/*check_nullptr=*/false);
      const T* k_norm = input_tensors[10].GetPtr<T>(/*check_nullptr=*/false);
      if (q_norm != nullptr && k_norm != nullptr) {
        int64_t q_rows = static_cast<int64_t>(total_token_num) * num_heads_;
        int64_t k_rows = static_cast<int64_t>(total_token_num) * num_kv_heads_;
        int64_t cols = head_size_;
        XpuRmsLayerNorm<T, T, T>(xctx, q_cont, q_cont, q_rows, cols, layernorm_eps_, q_norm, /*bias=*/nullptr,
                                 /*var=*/nullptr);
        XpuRmsLayerNorm<T, T, T>(xctx, k_cont, k_cont, k_rows, cols, layernorm_eps_, k_norm, /*bias=*/nullptr,
                                 /*var=*/nullptr);
      }
    }

    // 2. RoPE
    if constexpr (std::is_same_v<T, float16> || std::is_same_v<T, bfloat16>) {
      if (!no_rope_ && cos_sin_cache_ptr_ != nullptr) {
        const int rot_dim = head_size_;
        int rope_ret =
            XpuRotaryEmbeddingNeox<T>(xctx, positions, q_cont, k_cont, reinterpret_cast<const T*>(cos_sin_cache_ptr_),
                                      rot_dim, num_heads_, num_kv_heads_, head_size_, total_token_num);

        KLLM_CHECK_WITH_INFO(rope_ret == 0,
            fmt::format("PagedAttentionLayer: XpuRotaryEmbeddingNeox failed, ret={}", rope_ret));
      }
    }
  }  // end !use_fused_split_norm_rope

  // 3. KV cache scatter
  {
    int64_t num_blocks = input_tensors[4].shape[0];
    int ret = XpuReshapeCachedKV<T>(
        xctx, k_cont, v_cont, k_cache, v_cache, slot_mapping, static_cast<int64_t>(total_token_num), num_blocks,
        static_cast<int64_t>(num_kv_heads_), static_cast<int64_t>(head_size_), static_cast<int64_t>(block_token_num_));
    if (ret != 0) {
      KLLM_LOG_ERROR << "PagedAttentionLayer: reshape_cached_kv_v3 failed, ret=" << ret;
      return Status(RET_RUNTIME_FAILED, "reshape_cached_kv_v3 failed");
    }
  }

  // 4. Paged Attention: Q vs cached K/V
  {
    // Use block-table capacity as upper bound so the value stays valid across
    // XPU Graph replay (where kernel args are frozen at capture time).
    int64_t max_context_len = static_cast<int64_t>(max_num_blocks_per_query) * block_token_num_;

    // seq_len may already reside on device (XPU Graph mode).
    // Otherwise, copy from host into workspace tail on the compute stream.
    int32_t* context_len_device =
        reinterpret_cast<int32_t*>(reinterpret_cast<char*>(workspace) + q_bytes + 2 * kv_bytes);
    if (input_tensors[6].location == MemoryLocation::LOCATION_HOST) {
      MemcpyAsync(context_len_device, seq_len_host, batch_size * sizeof(int32_t), MEMCPY_HOST_TO_DEVICE,
                  context_->device->GetPrimaryComputeStream());
    } else {
      // XPU Graph mode: seq_len already resides on device; seq_len_host (the
      // GetPtr result) is the device pointer here, so reuse it directly.
      context_len_device = seq_len_host;
    }

    float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_size_));

    int ret = XpuPagedAttention<T, int32_t>(xctx, output, q_cont, k_cache, v_cache, static_cast<int64_t>(num_kv_heads_),
                                            softmax_scale, block_table, context_len_device,
                                            static_cast<int64_t>(block_token_num_), max_context_len, batch_size,
                                            static_cast<int64_t>(num_heads_), static_cast<int64_t>(head_size_),
                                            static_cast<int64_t>(max_num_blocks_per_query));
    if (ret != 0) {
      KLLM_LOG_ERROR << "PagedAttentionLayer: paged_attention failed, ret=" << ret;
      return Status(RET_RUNTIME_FAILED, "paged_attention failed");
    }
  }

  return Status();
}

// Explicit instantiations for fp16 and bf16. KV_DTYPE=kAuto is the only path on Kunlun.
template Status PagedAttentionLayer::ForwardT<float16, float16, llm_kernels::utils::KVCacheType::kAuto>(
    const PagedAttentionInputs&, std::vector<Tensor>&);
template Status PagedAttentionLayer::ForwardT<bfloat16, bfloat16, llm_kernels::utils::KVCacheType::kAuto>(
    const PagedAttentionInputs&, std::vector<Tensor>&);

}  // namespace ksana_llm
