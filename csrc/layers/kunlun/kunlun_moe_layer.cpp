/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/kunlun_moe_layer.h"

#include <type_traits>

#include "device/kunlun/kunlun_device_context_impl.h"
#include "kernels/kunlun/moe_kernel_wrapper.h"
#include "utils/logger.h"

namespace ksana_llm {

Status KunlunMoeLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                            std::shared_ptr<Context> context, int rank) {
  context_ = context;
  rank_ = rank;
  inter_data_type_ = runtime_config.inter_data_type;

  // All MoE layer variants now receive a single MoeLayerParameters struct (see MoeLayerFactory).
  params_ = std::any_cast<MoeLayerParameters>(parameters.front());
  max_token_num_ = params_.max_token_num;
  layer_idx_ = params_.layer_idx;
  expert_num_per_node_ = params_.expert_num_per_node;
  expert_hidden_size_ = params_.expert_hidden_size;
  expert_inter_size_ = params_.expert_inter_size;
  expert_topk_ = params_.expert_topk;
  num_expert_group_ = params_.num_expert_group;
  topk_group_ = params_.expert_groups_topk;
  scoring_func_ = params_.scoring_func;
  norm_topk_prob_ = params_.norm_topk_prob;
  routed_scaling_factor_ = params_.routed_scaling_factor;
  use_e_score_correction_bias_ = params_.use_e_score_correction_bias;
  // Expert weight quant dtype drives the W4A8 (INT4x2) dispatch below.
  const DataType int_weight_dtype = params_.int_weight_dtype;

  // Determine if expert weights are W4A8 quantized (INT4 packed in pairs, aligned with CUDA dispatch)
  is_w4a8_quantized_ = (int_weight_dtype == DataType::TYPE_INT4x2);

  // Expert parallelism config (same formula as nvidia/moe_layer.cpp)
  KLLM_CHECK_WITH_INFO(runtime_config.parallel_basic_config.expert_world_size == 1,
                       "Kunlun platform does not support multi-node expert parallelism (expert_world_size must be 1)");
  ep_size_ = runtime_config.parallel_basic_config.expert_parallel_size *
             runtime_config.parallel_basic_config.expert_world_size;
  ep_rank_ =
      context_->global->GetExpertParallelExpertNodeRank() * runtime_config.parallel_basic_config.expert_parallel_size +
      rank_;
  expert_num_total_ = expert_num_per_node_ * ep_size_;
  // TODO(zhongzhicao): 引入 w4a8 模型 ep=1 的专用实现 (non-EP W4A8 pre-sorted + quant kernel)
  KLLM_CHECK_WITH_INFO(!is_w4a8_quantized_ || ep_size_ != 1,
                       "Kunlun W4A8 quantized MoE only supports ep_size>1; ep_size=1 quant kernel is not adapted");

  KLLM_LOG_INFO << fmt::format(
      "KunlunMoeLayer::Init layer={} int_weight_dtype_raw={} is_w4a8={} "
      "experts={} topk={} hidden={} inter={} max_tokens={} scoring={} use_bias={} "
      "ep_size={} ep_rank={} num_expert_group={} topk_group={}",
      layer_idx_, GetTypeString(int_weight_dtype), is_w4a8_quantized_, expert_num_per_node_, expert_topk_,
      expert_hidden_size_, expert_inter_size_, max_token_num_, scoring_func_, use_e_score_correction_bias_, ep_size_,
      ep_rank_, num_expert_group_, topk_group_);

  if (scoring_func_ == "sigmoid" && use_e_score_correction_bias_) {
    KLLM_LOG_WARNING << fmt::format(
        "KunlunMoeLayer::Init layer={}: using XpuMoeSigmoidTopkNorm (scale={}, n_group={}, topk_group={})", layer_idx_,
        routed_scaling_factor_, num_expert_group_, topk_group_);
  }

  return Status();
}

size_t KunlunMoeLayer::GetWorkspaceSize() {
  // Workspace layout (for max_token_num_ tokens):
  //   m = max_token_num_, K = expert_topk_, H = expert_hidden_size_
  //   I = expert_inter_size_, E = expert_num_per_node_
  //
  // Using xsglang-style buffer reuse:
  //   buffer1: max(m*K*2*I, m*K*H) * sizeof(T)  → fc1_out / fc2_out
  //   buffer2: max(m*K*I, m*K*H) * sizeof(T)    → act_out / sorted_hidden
  //   topk_scores:    m * K * sizeof(float)
  //   topk_indices:   m * K * sizeof(int)
  //   block_stat:     64 * E * sizeof(int32_t)   // assume max 64 clusters
  //   moe_index:      m * K * sizeof(int)
  //   expert_m:       E * sizeof(int)
  //   lod:            (E+1) * sizeof(int)
  const size_t m = max_token_num_;
  const size_t K = expert_topk_;
  const size_t H = expert_hidden_size_;
  const size_t I = expert_inter_size_;
  const size_t E = expert_num_per_node_;
  const size_t E_total = expert_num_total_;
  const size_t type_size = (inter_data_type_ == DataType::TYPE_FP32) ? 4 : 2;

  // ncluster upper bound (Kunlun KL3 P800 has 8 clusters typically)
  const size_t max_ncluster = 64;

  size_t buffer1_size = std::max(m * K * 2 * I, m * K * H) * type_size;
  size_t buffer2_size = std::max(m * K * I, m * K * H) * type_size;
  size_t topk_scores_size = m * K * sizeof(float);
  size_t topk_indices_size = m * K * sizeof(int);
  size_t block_stat_size = max_ncluster * E_total * sizeof(int32_t);
  size_t moe_index_size = m * K * sizeof(int);
  size_t expert_m_size = E_total * sizeof(int);
  size_t lod_size = (E + 1) * sizeof(int);

  workspace_size_ = buffer1_size + buffer2_size + topk_scores_size + topk_indices_size + block_stat_size +
                    moe_index_size + expert_m_size + lod_size;

  // Align to 256 bytes
  workspace_size_ = (workspace_size_ + 255) & ~255;

  KLLM_LOG_DEBUG << fmt::format("KunlunMoeLayer workspace_size={} bytes (m={} K={} H={} I={} E={})", workspace_size_, m,
                                K, H, I, E);
  return workspace_size_;
}

Status KunlunMoeLayer::SetWorkspaceBuffer(const std::shared_ptr<Tensor>& workspace_buffer) {
  workspace_buffer_ = workspace_buffer;
  return Status();
}

Status KunlunMoeLayer::Forward(const Tensor* hidden_states, const Tensor* gating_output, const Tensor* w1,
                               const Tensor* w2, const Tensor* eplb_map, Tensor* output, Tensor* workspace,
                               const Tensor* e_score_bias, const Tensor* input_ids, const LayerExecContext& exec_ctx) {
  // Adapt the typed BaseMoeLayer interface to the vector-based ForwardT below.
  // Kunlun ignores eplb_map / input_ids (no expert-parallel or hash-gating), but it DOES support
  // the sigmoid-routing e-score-correction bias. ForwardT reads that bias from input_tensors[5]
  // (with [4] the expert_map placeholder slot), so when a bias is supplied it must be appended
  // here. Previously the wrapper built only 4 elements, so ForwardT's `input_tensors.size() > 5`
  // guard never fired and e_score_bias stayed nullptr — silently disabling e-score correction and
  // corrupting DeepSeek/Kimi grouped-topk routing (garbage output).
  std::vector<Tensor> input_tensors = {*hidden_states, *gating_output, *w1, *w2};
  if (use_e_score_correction_bias_ && e_score_bias != nullptr) {
    input_tensors.push_back(eplb_map != nullptr ? *eplb_map : Tensor());  // [4] expert_map (ignored)
    input_tensors.push_back(*e_score_bias);                               // [5] e_score_correction_bias
  }
  std::vector<Tensor> output_tensors = {*output};

  Status status;
  switch (inter_data_type_) {
    case DataType::TYPE_FP16:
      status = ForwardT<float16>(input_tensors, output_tensors);
      break;
    case DataType::TYPE_BF16:
      status = ForwardT<bfloat16>(input_tensors, output_tensors);
      break;
    case DataType::TYPE_FP32:
      status = ForwardT<float>(input_tensors, output_tensors);
      break;
    default:
      KLLM_THROW(fmt::format("{}: Unsupported Dtype type: {}.", __PRETTY_FUNCTION__, inter_data_type_));
  }
  // Propagate the shape/dtype ForwardT set on the local output tensor back to the caller.
  *output = output_tensors[0];
  return status;
}

template <typename T>
Status KunlunMoeLayer::ForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  // Input tensors:
  //   [0] hidden_states [num_tokens, hidden_units]
  //   [1] gating_output [num_tokens, num_experts]
  //   [2] up_gate_proj_weight [num_experts, 2*inter_size, hidden_units(/2 if W4)]
  //   [3] down_proj_weight [num_experts, hidden_units, inter_size(/2 if W4)]
  //   [4] expert_map (ignored for kunlun)
  //   [5] e_score_correction_bias [num_experts] (optional, only when use_e_score_correction_bias_)
  // Output tensors:
  //   [0] output [num_tokens, hidden_units]
  //   [1] workspace tensor

  const int64_t m = static_cast<int64_t>(input_tensors[0].shape[0]);  // num_tokens
  const int64_t H = static_cast<int64_t>(expert_hidden_size_);
  const int64_t I = static_cast<int64_t>(expert_inter_size_);
  const int64_t E = static_cast<int64_t>(expert_num_per_node_);     // local experts (weight batch dim)
  const int64_t E_total = static_cast<int64_t>(expert_num_total_);  // total experts (routing dim)
  const int64_t K = static_cast<int64_t>(expert_topk_);
  const size_t type_size = sizeof(T);

  // Skip if no tokens
  if (m == 0) {
    output_tensors[0].shape = {0, static_cast<size_t>(H)};
    output_tensors[0].dtype = input_tensors[0].dtype;
    return Status();
  }

  // Get device pointers
  const T* hidden_states = input_tensors[0].GetPtr<T>();
  const T* gating_output = input_tensors[1].GetPtr<T>();
  T* output = output_tensors[0].GetPtr<T>();

  // Expert weights (may be BF16 or INT8-packed-INT4)
  const void* w1_ptr = input_tensors[2].GetPtr<void>();
  const void* w2_ptr = input_tensors[3].GetPtr<void>();

  // Expert weight scales (for W4A8 path, accessed via weight_scales pointer)
  const float* w1_scale = nullptr;
  const float* w2_scale = nullptr;
  if (is_w4a8_quantized_) {
    KLLM_CHECK_WITH_INFO(input_tensors[2].weight_scales != nullptr,
                         "W4A8 MoE: up_gate_proj weight must have weight_scales bound");
    KLLM_CHECK_WITH_INFO(input_tensors[3].weight_scales != nullptr,
                         "W4A8 MoE: down_proj weight must have weight_scales bound");
    w1_scale = input_tensors[2].weight_scales->GetPtr<float>();
    w2_scale = input_tensors[3].weight_scales->GetPtr<float>();
  }

  // E-score correction bias (optional)
  const T* e_score_bias = nullptr;
  if (use_e_score_correction_bias_ && input_tensors.size() > 5) {
    e_score_bias = input_tensors[5].GetPtr<T>();
  }

  xdnn::Context* ctx = static_cast<xdnn::Context*>(
      context_->device->ext->GetXdnnContext(context_->device->GetPrimaryComputeStream().Get()));
  const int64_t ncluster = ctx->ncluster();

  // Workspace layout (carve out from workspace_buffer_)
  char* ws_base = workspace_buffer_->GetPtr<char>();
  size_t offset = 0;

  size_t buffer1_bytes = std::max(m * K * 2 * I, m * K * H) * type_size;
  size_t buffer2_bytes = std::max(m * K * I, m * K * H) * type_size;
  size_t topk_scores_bytes = m * K * sizeof(float);
  size_t topk_indices_bytes = m * K * sizeof(int);
  size_t block_stat_bytes = ncluster * E_total * sizeof(int32_t);
  size_t moe_index_bytes = m * K * sizeof(int);
  size_t expert_m_bytes = E_total * sizeof(int);
  size_t lod_bytes = (E + 1) * sizeof(int);

  T* buffer1 = reinterpret_cast<T*>(ws_base + offset);
  offset += buffer1_bytes;
  T* buffer2 = reinterpret_cast<T*>(ws_base + offset);
  offset += buffer2_bytes;
  float* topk_scores = reinterpret_cast<float*>(ws_base + offset);
  offset += topk_scores_bytes;
  int* topk_indices = reinterpret_cast<int*>(ws_base + offset);
  offset += topk_indices_bytes;
  int32_t* block_stat = reinterpret_cast<int32_t*>(ws_base + offset);
  offset += block_stat_bytes;
  int* moe_index = reinterpret_cast<int*>(ws_base + offset);
  offset += moe_index_bytes;
  int* expert_m = reinterpret_cast<int*>(ws_base + offset);
  offset += expert_m_bytes;
  int* lod = reinterpret_cast<int*>(ws_base + offset);
  offset += lod_bytes;

  // Assign buffer roles
  T* sorted_hidden = buffer2;
  T* fc1_out = buffer1;
  T* act_out = buffer2;
  T* fc2_out = buffer1;

  int ret;

  // ═══ Step 1: Scoring + TopK routing ═══
  if (scoring_func_ == "sigmoid") {
    if (use_e_score_correction_bias_) {
      // Kimi-K2.6: sigmoid + grouped_topk + norm + scale bake
      // Aligned with xsglang: infer_ops::moe_sigmoid_topk_norm_fusion
      // Called when use_grouped_topk=true && correction_bias exists (xsglang always sets use_grouped_topk=true)
      const float* bias_fp32 = (input_tensors.size() > 5) ? input_tensors[5].GetPtr<float>() : nullptr;
      ret = XpuMoeSigmoidTopkNorm<T, float, int>(ctx, gating_output, bias_fp32, routed_scaling_factor_, topk_scores,
                                                 topk_indices, block_stat, m, E_total, num_expert_group_, topk_group_,
                                                 K, ncluster);
    } else {
      // Sigmoid scoring without correction_bias (fallback)
      ret =
          XpuMoeFusedGateKimiK2<T, float, int>(ctx, gating_output, e_score_bias, topk_scores, topk_indices, m, E_total,
                                               K, norm_topk_prob_, routed_scaling_factor_, block_stat, ncluster);
    }
  } else if (norm_topk_prob_) {
    // Softmax + TopK + Normalization (Qwen3-MoE style)
    ret = XpuMoeSoftmaxTopkNorm<T, float, int>(ctx, gating_output, topk_scores, topk_indices, block_stat, m, E_total, K,
                                               ncluster, /*stable=*/true);
  } else {
    // Softmax + TopK only (Qwen2-MoE style)
    ret = XpuMoeSoftmaxTopk<T, float, int>(ctx, gating_output, topk_scores, topk_indices, block_stat, m, E_total, K,
                                           ncluster);
  }
  if (ret != 0) {
    return Status(RET_RUNTIME_FAILED,
                  fmt::format("KunlunMoeLayer: Scoring/TopK failed (scoring={}, ret={})", scoring_func_, ret));
  }

  // ═══ Step 2~6: Expert computation (pre-sort → FC1 → act → FC2 → post-fusion) ═══
  // Each branch calls its own EP-aware pre-sort variant.
  if (is_w4a8_quantized_) {
    // ─── W4A8 INT4 quantized expert path (aligned with klx_ep_moe.cc) ───
    if constexpr (std::is_same_v<T, float>) {
      return Status(RET_RUNTIME_FAILED, "KunlunMoeLayer: W4A8 MoE not supported for fp32 dtype");
    } else {
      // Allocate quantization buffers
      xdnn::ctx_guard guard(ctx);
      int8_t* sorted_hidden_int8 = guard.alloc_l3_or_gm<int8_t>(m * K * H);
      float* act_scale_1 = guard.alloc_l3_or_gm<float>(m * K);
      int8_t* act_int8 = guard.alloc_l3_or_gm<int8_t>(m * K * I);
      float* act_scale_2 = guard.alloc_l3_or_gm<float>(m * K);

      if (!sorted_hidden_int8 || !act_scale_1 || !act_int8 || !act_scale_2) {
        return Status(RET_RUNTIME_FAILED, "KunlunMoeLayer: W4A8 workspace alloc failed");
      }

      // Step 2+3a: EP fused pre-sort + INT8 quantization
      ret =
          XpuMoeEpPreSortedQuant2d<T, int>(ctx, hidden_states, topk_indices, block_stat, sorted_hidden_int8, moe_index,
                                           expert_m, lod, act_scale_1, m, H, E_total, K, ncluster, ep_size_, ep_rank_);
      if (ret != 0)
        return Status(RET_RUNTIME_FAILED, fmt::format("KunlunMoeLayer: W4A8 EpPreSortedQuant2d failed (ret={})", ret));

      // MoeDesc tokens/topk: EP mode flattens topk dimension (matches klx_ep_moe.cc)
      // ep_size==1: tokens=m, topk=K (matches klx_fused_moe.cc)
      // ep_size>1:  tokens=m*K, topk=1 (matches klx_ep_moe.cc, pre_sorted already expanded)
      const int64_t fc_tokens = (ep_size_ > 1) ? m * K : m;
      const int64_t fc_topk = (ep_size_ > 1) ? 1 : K;

      // Step 3b: Expert FC1 (gate_up projection) via wrapper
      ret = XpuMoeW4A8FcFusion<T, int>(ctx, sorted_hidden_int8, w1_ptr, fc1_out, act_scale_1, w1_scale, fc_tokens,
                                       fc_topk, m * K, 2 * I, H, E, moe_index, lod);
      if (ret != 0) return Status(RET_RUNTIME_FAILED, fmt::format("KunlunMoeLayer: W4A8 FC1 failed (ret={})", ret));

      // Step 4+5a: Fused SwiGLU + INT8 quantization
      ret = XpuSiluAndMulQuantUnt<T>(ctx, fc1_out, act_int8, act_scale_2, m * K, 2 * I, m * K, lod + E);
      if (ret != 0)
        return Status(RET_RUNTIME_FAILED, fmt::format("KunlunMoeLayer: W4A8 SiluAndMulQuantUnt failed (ret={})", ret));

      // Step 5b: Expert FC2 (down projection) via wrapper
      ret = XpuMoeW4A8FcFusion<T, int>(ctx, act_int8, w2_ptr, fc2_out, act_scale_2, w2_scale, fc_tokens, fc_topk, m * K,
                                       H, I, E, moe_index, lod);
      if (ret != 0) return Status(RET_RUNTIME_FAILED, fmt::format("KunlunMoeLayer: W4A8 FC2 failed (ret={})", ret));

      // Step 6: EP post-fusion (weighted reduce)
      // TODO(zhongzhicao): xsglang TopK直接输出bf16 score，post_fusion用<bf16,int,float,bf16>实例化；
      // 当前KsanaLLM TopK输出float score，用<bf16,int,float,float>实例化。功能正确，可优化对齐。
      ret = XpuMoeEpFfnPostFusion<T, int>(ctx, fc2_out, moe_index, topk_scores, output, m, H, K);
      if (ret != 0) return Status(RET_RUNTIME_FAILED, "KunlunMoeLayer: W4A8 PostFusion failed");
    }  // end if constexpr (!float)
  } else {
    // ─── BF16 expert path ───
    const T* w1 = static_cast<const T*>(w1_ptr);
    const T* w2 = static_cast<const T*>(w2_ptr);

    if (ep_size_ == 1) {
      // ─── klx_fused_moe 流程 (ep=1, single-node) ───
      // Aligned with xsglang klx_fused_moe.cc: uses moe_ffn_pre_sorted + fixed params + moe_ffn_post_fusion

      // Step 2: Pre-sort (非EP版本，固定参数)
      ret = XpuMoeFfnPreSorted<T, int>(ctx, hidden_states, topk_indices, block_stat, sorted_hidden, moe_index, expert_m,
                                       lod, m, H, E, K, ncluster);
      if (ret != 0)
        return Status(RET_RUNTIME_FAILED, fmt::format("KunlunMoeLayer: BF16 FfnPreSorted failed (ret={})", ret));

      // Step 3: Expert FC1 (gate_up projection, tokens=m, topk=K 固定)
      ret = XpuMoeFcFusion<T, int>(ctx, sorted_hidden, w1, fc1_out, m, K, m * K, 2 * I, H, E, moe_index, lod);
      if (ret != 0) return Status(RET_RUNTIME_FAILED, "KunlunMoeLayer: BF16 FC1 failed");

      // Step 4: SwiGLU activation
      ret = XpuSwiglu<T>(ctx, fc1_out, act_out, m * K, I);
      if (ret != 0) return Status(RET_RUNTIME_FAILED, "KunlunMoeLayer: SwiGLU failed");

      // Step 5: Expert FC2 (down projection, tokens=m, topk=K 固定)
      ret = XpuMoeFcFusion<T, int>(ctx, act_out, w2, fc2_out, m, K, m * K, H, I, E, moe_index, lod);
      if (ret != 0) return Status(RET_RUNTIME_FAILED, "KunlunMoeLayer: BF16 FC2 failed");

      // Step 6: Post-fusion (klx_fused_moe版本，处理float score)
      ret = XpuMoeFfnPostFusionF32Score<T, int>(ctx, fc2_out, moe_index, topk_scores, output, m, H, K);
      if (ret != 0) return Status(RET_RUNTIME_FAILED, "KunlunMoeLayer: BF16 PostFusion failed");

    } else {
      // ─── klx_ep_moe 流程 (ep>1, expert parallel) ───
      // Aligned with xsglang klx_ep_moe.cc: uses moe_ep_ffn_pre_sorted + dynamic params + moe_ep_ffn_post_fusion

      // Step 2: EP pre-sort (动态参数，支持多节点专家并行)
      ret = XpuMoeEpFfnPreSorted<T, int>(ctx, hidden_states, topk_indices, block_stat, sorted_hidden, moe_index,
                                         expert_m, lod, m, H, E_total, K, ncluster, ep_size_, ep_rank_);
      if (ret != 0)
        return Status(RET_RUNTIME_FAILED, fmt::format("KunlunMoeLayer: BF16 EpFfnPreSorted failed (ret={})", ret));

      // EP模式下tokens/topk固定为 m*K / 1 (pre_sorted已展开topk维度)
      const int64_t bf16_fc_tokens = m * K;
      const int64_t bf16_fc_topk = 1;

      // Step 3: Expert FC1 (gate_up projection, EP参数)
      ret = XpuMoeFcFusion<T, int>(ctx, sorted_hidden, w1, fc1_out, bf16_fc_tokens, bf16_fc_topk, m * K, 2 * I, H, E,
                                   moe_index, lod);
      if (ret != 0) return Status(RET_RUNTIME_FAILED, "KunlunMoeLayer: BF16 FC1 failed");

      // Step 4: SwiGLU activation
      ret = XpuSwiglu<T>(ctx, fc1_out, act_out, m * K, I);
      if (ret != 0) return Status(RET_RUNTIME_FAILED, "KunlunMoeLayer: SwiGLU failed");

      // Step 5: Expert FC2 (down projection, EP参数)
      ret = XpuMoeFcFusion<T, int>(ctx, act_out, w2, fc2_out, bf16_fc_tokens, bf16_fc_topk, m * K, H, I, E, moe_index,
                                   lod);
      if (ret != 0) return Status(RET_RUNTIME_FAILED, "KunlunMoeLayer: BF16 FC2 failed");

      // Step 6: EP post-fusion (weighted scatter-add)
      ret = XpuMoeEpFfnPostFusion<T, int>(ctx, fc2_out, moe_index, topk_scores, output, m, H, K);
      if (ret != 0) return Status(RET_RUNTIME_FAILED, "KunlunMoeLayer: BF16 PostFusion failed");
    }
  }

  // Set output shape
  output_tensors[0].shape = {static_cast<size_t>(m), static_cast<size_t>(H)};
  output_tensors[0].dtype = input_tensors[0].dtype;

  return Status();
}

}  // namespace ksana_llm
