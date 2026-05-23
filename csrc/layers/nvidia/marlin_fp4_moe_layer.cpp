/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "layers/marlin_fp4_moe_layer.h"

#include "csrc/kernels/nvidia/moe_utils/moe_utils.h"
#include "csrc/kernels/nvidia/others/sglang/main/marlin_moe_wna16/marlin_moe_wna16_wrapper.h"
#include "csrc/kernels/nvidia/others/sglang/main/marlin_moe_wna16/ops.h"
#include "csrc/kernels/nvidia/others/sglang/main/marlin_moe_wna16/scalar_type.hpp"
#include "layers/grouped_topk_layer.h"
#include "layers/topk_softplus_sqrt_layer.h"
#include "utils/utils.h"

namespace ksana_llm {

#define MARLIN_FP4_MOE_CHUNK_SIZE ((size_t)(32 * 1024))

// Minimum MOE block size for workspace pre-allocation (worst-case padding)
static constexpr int64_t kMarlinFp4MoeMinBlockSize = 8;

// Dynamically select moe_block_size based on token density per expert.
// Matches sglang's heuristic: iterate [8, 16, 32, 48, 64] and pick the first
// block_size where (M * topk / num_experts / block_size) < 0.9.
// Smaller block sizes reduce padding waste at low token density (decode),
// while larger block sizes improve SM utilization at high density (prefill).
static inline int64_t SelectMoeBlockSize(size_t num_tokens, size_t topk, size_t num_experts) {
  static constexpr int64_t kBlockSizes[] = {8, 16, 32, 48, 64};
  for (int64_t bs : kBlockSizes) {
    double density = static_cast<double>(num_tokens * topk) / static_cast<double>(num_experts * bs);
    if (density < 0.9) return bs;
  }
  return 64;
}

inline size_t AlignAddress(size_t size) { return (size + 255) & (~255); }

Status MarlinFp4MoeLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                               std::shared_ptr<Context> context, int rank) {
  inter_data_type_ = runtime_config.inter_data_type;
  DISPATCH_BY_3_DTYPE(inter_data_type_, InitT, parameters, runtime_config, context, rank);
}

template <typename T>
Status MarlinFp4MoeLayer::InitT(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                std::shared_ptr<Context> context, int rank) {
  context_ = context;
  rank_ = rank;

  int parameter_index = 0;
  moe_scale_norm_mode_ = std::any_cast<const MoeScaleNormMode>(parameters[parameter_index++]);
  max_token_num_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  layer_idx_ = std::any_cast<int>(parameters[parameter_index++]);
  expert_num_per_node_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  expert_hidden_size_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  expert_inter_size_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  expert_topk_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  tp_size_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  use_vllm_moe_ = std::any_cast<bool>(parameters[parameter_index++]);
  num_expert_group_ = std::any_cast<uint32_t>(parameters[parameter_index++]);
  expert_groups_topk_ = std::any_cast<uint32_t>(parameters[parameter_index++]);
  scoring_func_ = std::any_cast<std::string>(parameters[parameter_index++]);
  topk_method_ = std::any_cast<std::string>(parameters[parameter_index++]);
  norm_topk_prob_ = std::any_cast<bool>(parameters[parameter_index++]);
  routed_scaling_factor_ = std::any_cast<float>(parameters[parameter_index++]);
  use_e_score_correction_bias_ = std::any_cast<bool>(parameters[parameter_index++]);
  enable_full_shared_expert_ = std::any_cast<bool>(parameters[parameter_index++]);
  [[maybe_unused]] DataType fp8_weight_dtype = std::any_cast<DataType>(parameters[parameter_index++]);
  [[maybe_unused]] DataType int_weight_dtype = std::any_cast<DataType>(parameters[parameter_index++]);
  [[maybe_unused]] int group_size = std::any_cast<int>(parameters[parameter_index++]);
  apply_weight_ = std::any_cast<bool>(parameters[parameter_index++]);
  swiglu_limit_ = std::any_cast<float>(parameters[parameter_index++]);
  has_swiglu_limit_ = (swiglu_limit_ > 0.0f);
  num_hash_layers_ = std::any_cast<uint32_t>(parameters[parameter_index++]);
  vocab_size_ = std::any_cast<uint32_t>(parameters[parameter_index++]);

  // Compute global EP size and rank
  global_expert_para_size_ = runtime_config.parallel_basic_config.expert_parallel_size *
                             runtime_config.parallel_basic_config.expert_world_size;

  // Multi-node + DP + EP enables deepep
  using_deepep_ = runtime_config.parallel_basic_config.expert_world_size > 1 &&
                  runtime_config.parallel_basic_config.attn_data_parallel_size > 1 &&
                  runtime_config.parallel_basic_config.expert_parallel_size > 1;

  use_topk_softplus_sqrt_ = (scoring_func_ == "sqrtsoftplus");
  if (use_topk_softplus_sqrt_) {
    use_hash_ = (layer_idx_ < static_cast<int>(num_hash_layers_));
    TopKSoftplusSqrtLayerParameters softplus_sqrt_params{};
    softplus_sqrt_params.topk = static_cast<uint32_t>(expert_topk_);
    softplus_sqrt_params.num_experts = static_cast<uint32_t>(expert_num_per_node_ * global_expert_para_size_);
    softplus_sqrt_params.max_token_num = static_cast<uint32_t>(max_token_num_);
    softplus_sqrt_params.renormalize = norm_topk_prob_;
    softplus_sqrt_params.routed_scaling_factor = routed_scaling_factor_;
    softplus_sqrt_params.use_hash = use_hash_;
    softplus_sqrt_params.use_correction_bias = !use_hash_ && use_e_score_correction_bias_;
    softplus_sqrt_params.topk_ids_dtype = TYPE_INT32;

    topk_softplus_sqrt_layer_ = std::make_shared<TopKSoftplusSqrtLayer>();
    topk_softplus_sqrt_layer_->Init({softplus_sqrt_params}, runtime_config, context, rank);
    softplus_sqrt_workspace_size_ = topk_softplus_sqrt_layer_->GetWorkspaceSize();
  } else {
    // Initialize GroupedTopkLayer for V3/GLM4/Qwen3-MoE path
    grouped_topk_layer_ = std::make_shared<GroupedTopkLayer>();
    float score_threshold = runtime_config.moe_topk_score_threshold;
    GroupedTopkLayerParameters grouped_topk_params{.topk = expert_topk_,
                                                   .renormalize = norm_topk_prob_,
                                                   .num_expert_group = num_expert_group_,
                                                   .topk_group = expert_groups_topk_,
                                                   .scoring_func = scoring_func_,
                                                   .routed_scaling_factor = routed_scaling_factor_,
                                                   .use_e_score_correction_bias = use_e_score_correction_bias_,
                                                   .score_threshold = score_threshold};
    grouped_topk_layer_->Init({grouped_topk_params}, runtime_config, context, rank);
  }

  // EPLB data dump configuration
  enable_dump_eplb_data_ = runtime_config.enable_dump_eplb_data;
  if (enable_dump_eplb_data_) {
    eplb_dump_path_ = runtime_config.dump_eplb_path;
  }

  // EPLB acceleration or EP requires expert_map
  enable_load_eplb_config_ = runtime_config.enable_load_eplb_config;
  if (enable_load_eplb_config_ || !using_deepep_) {
    size_t global_expert_para_rank = context_->global->GetExpertParallelExpertNodeRank() *
                                         runtime_config.parallel_basic_config.expert_parallel_size +
                                     rank_;
    expert_map_ = std::make_shared<llm_kernels::nvidia::moe::ExpertMap>(
        global_expert_para_size_, global_expert_para_rank, expert_num_per_node_ * global_expert_para_size_);
  }

  KLLM_LOG_INFO << fmt::format("Rank[{}] MarlinFp4MoeLayer Init", rank_);
  return Status();
}

void MarlinFp4MoeLayer::SetExpertParallelManager(std::shared_ptr<ExpertParallelManager> expert_parallel_manager) {
  expert_parallel_manager_ = expert_parallel_manager;
}

size_t MarlinFp4MoeLayer::GetWorkspaceSize() { DISPATCH_BY_3_DTYPE(inter_data_type_, GetWorkspaceSizeT); }

template <typename T>
size_t MarlinFp4MoeLayer::GetWorkspaceSizeT() {
  size_t m = std::min(MARLIN_FP4_MOE_CHUNK_SIZE, max_token_num_);

  topk_weights_ptr_size = AlignAddress(max_token_num_ * expert_topk_ * sizeof(float));
  topk_ids_ptr_size = AlignAddress(max_token_num_ * expert_topk_ * sizeof(int32_t));

  // fused_id_buffer: fill_info_vec(4) + sorted_ids + expert_ids + num_tokens_post_pad + cumsum
  size_t numel = m * expert_topk_;
  size_t max_num_tokens_padded = numel + expert_num_per_node_ * (kMarlinFp4MoeMinBlockSize - 1);
  size_t max_num_m_blocks = (max_num_tokens_padded + kMarlinFp4MoeMinBlockSize - 1) / kMarlinFp4MoeMinBlockSize;
  // fill_info(4) + sorted_ids(max_num_tokens_padded) + expert_ids(max_num_m_blocks) + num_tokens_post_pad(1) +
  // cumsum(expert_num_per_node+1)
  max_fused_id_buffer_size = AlignAddress(
      (SGL_MOE_ALIGN_BLOCK_VEC_SIZE + max_num_tokens_padded + max_num_m_blocks + 1 + expert_num_per_node_ + 1) *
      sizeof(int32_t));

  // Intermediate caches
  // cache1: [m * topk, inter_size * 2] — output of w1 GEMM (gate+up)
  intermediate_cache1_size = AlignAddress(m * expert_topk_ * expert_inter_size_ * 2 * sizeof(T));
  // cache2: [m * topk, inter_size] — output of SiluAndMul
  intermediate_cache2_size = AlignAddress(m * expert_topk_ * expert_inter_size_ * sizeof(T));
  // cache3: [m * topk, hidden_size] — output of w2 GEMM
  // NOTE: cache3 reuses cache1's memory (P5 optimization) since their lifetimes don't overlap:
  //   cache1 is consumed by SiluAndMul before cache3 is produced by GEMM2.
  //   We still track the size for documentation, but don't add it to total allocation.
  intermediate_cache3_size = AlignAddress(m * expert_topk_ * expert_hidden_size_ * sizeof(T));

  // Marlin workspace: sized for the larger of the two GEMMs
  // Use min block size for worst-case workspace allocation
  int64_t max_n = std::max(expert_inter_size_ * 2, expert_hidden_size_);
  marlin_workspace_size = AlignAddress(llm_kernels::nvidia::marlin_moe_wna16::get_workspace_size(
                                           max_n, max_num_tokens_padded, kMarlinFp4MoeMinBlockSize) *
                                       sizeof(int));

  // c_tmp buffer for fp32 global reduce (P1 optimization)
  // Matches sglang: use_fp32_reduce=true requires a fp32 temp buffer for cross-SM reduction
  c_tmp_size =
      AlignAddress(llm_kernels::nvidia::marlin_moe_wna16::get_c_tmp_size(
                       max_n, max_num_tokens_padded, kMarlinFp4MoeMinBlockSize, ::marlin_moe_wna16::kFE2M1f.id()) *
                   sizeof(float));

  const size_t softplus_sqrt_ws_aligned = use_topk_softplus_sqrt_ ? AlignAddress(softplus_sqrt_workspace_size_) : 0;

  // Total workspace: cache1 is shared with cache3, so we use max(cache1, cache3) instead of sum
  size_t cache13_size = std::max(intermediate_cache1_size, intermediate_cache3_size);
  max_ws_bytes_ = topk_weights_ptr_size + topk_ids_ptr_size + max_fused_id_buffer_size + cache13_size +
                  intermediate_cache2_size + marlin_workspace_size + c_tmp_size + softplus_sqrt_ws_aligned;

  KLLM_LOG_DEBUG << fmt::format("Rank[{}] Request {} for MarlinFp4MoeLayer", rank_, max_ws_bytes_);
  return max_ws_bytes_;
}

Status MarlinFp4MoeLayer::Preprocess(const ModelConfig& model_config, const RuntimeConfig& runtime_config) {
  // Marlin FP4 kernel does not require profiling/preprocessing
  return Status();
}

Status MarlinFp4MoeLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, input_tensors, output_tensors);
}

template <typename T>
Status MarlinFp4MoeLayer::ForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  // input_tensors:
  //  0: hidden_states [num_tokens, hidden_size]
  //  1: gating_output [num_tokens, num_experts]
  //  2: up_gate_proj_weight (w1) — packed FP4
  //  3: down_proj_weight (w2) — packed FP4
  //  4: eplb_expert_map
  //  (*)5: e_score_correction_bias_weight / tid2eid
  //  (*)6: input_ids (hash path)

  // Assign workspace pointers on first call
  if (set_workspace_buffer_info_) {
    set_workspace_buffer_info_ = false;
    topk_weights_ptr_ = workspace_buffer_->GetPtr<void>();
    topk_ids_ptr_ = topk_weights_ptr_ + topk_weights_ptr_size;
    fused_id_buffer_ = topk_ids_ptr_ + topk_ids_ptr_size;
    intermediate_cache1_ = fused_id_buffer_ + max_fused_id_buffer_size;
    intermediate_cache2_ = intermediate_cache1_ + std::max(intermediate_cache1_size, intermediate_cache3_size);
    // cache3 shares memory with cache1 (non-overlapping lifetimes: cache1 consumed before cache3 produced)
    intermediate_cache3_ = intermediate_cache1_;
    marlin_workspace_ = reinterpret_cast<int*>(intermediate_cache2_ + intermediate_cache2_size);
    c_tmp_ = reinterpret_cast<float*>(reinterpret_cast<char*>(marlin_workspace_) + marlin_workspace_size);

    if (use_topk_softplus_sqrt_ && softplus_sqrt_workspace_size_ > 0) {
      const size_t slice_offset = max_ws_bytes_ - AlignAddress(softplus_sqrt_workspace_size_);
      auto slice = std::make_shared<Tensor>(
          workspace_buffer_->location, TYPE_UINT8, std::vector<size_t>{softplus_sqrt_workspace_size_},
          reinterpret_cast<int8_t*>(workspace_buffer_->GetPtr<void>()) + slice_offset);
      topk_softplus_sqrt_layer_->SetWorkspaceBuffer(slice);
    }
  }

  const size_t num_tokens = input_tensors[0].shape[0];
  const size_t hidden_size = input_tensors[0].shape[1];

  // Execute TopK gating
  if (use_topk_softplus_sqrt_) {
    STATUS_CHECK_RETURN(ExecuteTopkSoftplusSqrt<T>(input_tensors, output_tensors));
  } else {
    STATUS_CHECK_RETURN(ExecuteGroupedTopk<T>(input_tensors, output_tensors));
  }

  // Get weight pointers
  void* w1 = input_tensors[2].GetPtr<void>();
  void* w2 = input_tensors[3].GetPtr<void>();
  void* w1_scales =
      (input_tensors[2].weight_scales != nullptr) ? input_tensors[2].weight_scales->GetPtr<void>() : nullptr;
  void* w2_scales =
      (input_tensors[3].weight_scales != nullptr) ? input_tensors[3].weight_scales->GetPtr<void>() : nullptr;

  // Determine compute dtype for Marlin kernel: 0=fp16, 1=bf16
  int marlin_compute_dtype = (inter_data_type_ == DataType::TYPE_BF16) ? 1 : 0;

  auto ProcessChunks = [&](void* hidden_states_ptr, void* output_ptr, size_t total_tokens) {
    cudaStream_t stream = context_->device->GetPrimaryComputeStream().Get();
    const bool use_expert_parallel = (global_expert_para_size_ > 1 && !using_deepep_);

    for (size_t chunk_start = 0; chunk_start < total_tokens; chunk_start += MARLIN_FP4_MOE_CHUNK_SIZE) {
      size_t tokens_in_chunk = std::min(MARLIN_FP4_MOE_CHUNK_SIZE, total_tokens - chunk_start);

      // Current chunk pointers
      void* curr_hidden_states = static_cast<char*>(hidden_states_ptr) + chunk_start * hidden_size * sizeof(T);
      void* curr_output = static_cast<char*>(output_ptr) + chunk_start * hidden_size * sizeof(T);
      void* curr_topk_ids = static_cast<char*>(topk_ids_ptr_) + chunk_start * expert_topk_ * sizeof(int32_t);
      void* curr_topk_weights = static_cast<char*>(topk_weights_ptr_) + chunk_start * expert_topk_ * sizeof(float);

      // ----- moe_align_block_size -----
      // Dynamically select block size based on token density per expert (matches sglang heuristic)
      int64_t moe_block_size = SelectMoeBlockSize(tokens_in_chunk, expert_topk_, expert_num_per_node_);

      int64_t numel = static_cast<int64_t>(tokens_in_chunk * expert_topk_);
      int64_t max_num_tokens_padded = numel + expert_num_per_node_ * (moe_block_size - 1);
      int64_t max_num_m_blocks = (max_num_tokens_padded + moe_block_size - 1) / moe_block_size;

      // Layout within fused_id_buffer: [fill_info(4)] [sorted_ids] [expert_ids] [num_tokens_post_pad] [cumsum]
      void* fill_info_buffer_on_device = fused_id_buffer_;
      void* curr_fused_id_buffer =
          static_cast<char*>(fill_info_buffer_on_device) + SGL_MOE_ALIGN_BLOCK_VEC_SIZE * sizeof(int32_t);
      int id_buffer_offset = 0;

      int32_t* sorted_ids_ptr = reinterpret_cast<int32_t*>(curr_fused_id_buffer) + id_buffer_offset;

      // Fill sorted_ids with sentinel value (numel) for experts < 224
      if (expert_num_per_node_ < 224) {
        std::vector<int> fill_info(3);
        fill_info[0] = id_buffer_offset;
        fill_info[1] = static_cast<int>(max_num_tokens_padded);
        fill_info[2] = static_cast<int>(numel);
        llm_kernels::nvidia::InvokeFillIntToBuffer(reinterpret_cast<int32_t*>(curr_fused_id_buffer),
                                                   fill_info_buffer_on_device, reinterpret_cast<int*>(fill_info.data()),
                                                   fill_info.size(), stream);
      }
      id_buffer_offset += max_num_tokens_padded;

      int32_t* expert_ids_ptr = reinterpret_cast<int32_t*>(curr_fused_id_buffer) + id_buffer_offset;
      id_buffer_offset += max_num_m_blocks;

      int32_t* num_tokens_post_pad_ptr = reinterpret_cast<int32_t*>(curr_fused_id_buffer) + id_buffer_offset;
      id_buffer_offset += 1;

      if (expert_num_per_node_ >= 224) {
        int32_t* cumsum_ptr = reinterpret_cast<int32_t*>(curr_fused_id_buffer) + id_buffer_offset;
        llm_kernels::nvidia::InvokeSglMoeAlignBlockSize<int32_t>(
            reinterpret_cast<int32_t*>(curr_topk_ids), sorted_ids_ptr, expert_ids_ptr, num_tokens_post_pad_ptr,
            static_cast<int32_t>(max_num_tokens_padded), static_cast<int32_t>(expert_num_per_node_),
            static_cast<int32_t>(moe_block_size), static_cast<size_t>(numel), cumsum_ptr, stream);
      } else {
        if (use_expert_parallel) {
          llm_kernels::nvidia::InvokeMoeAlignBlockSize<int32_t, uint16_t, true>(
              reinterpret_cast<int32_t*>(curr_topk_ids), sorted_ids_ptr, expert_ids_ptr, num_tokens_post_pad_ptr,
              static_cast<int32_t>(expert_topk_), static_cast<int32_t>(expert_num_per_node_),
              static_cast<int32_t>(global_expert_para_size_), static_cast<int32_t>(moe_block_size),
              static_cast<size_t>(numel), rank_, stream);
        } else {
          llm_kernels::nvidia::InvokeMoeAlignBlockSize<int32_t, uint16_t, false>(
              reinterpret_cast<int32_t*>(curr_topk_ids), sorted_ids_ptr, expert_ids_ptr, num_tokens_post_pad_ptr,
              static_cast<int32_t>(expert_topk_), static_cast<int32_t>(expert_num_per_node_),
              static_cast<int32_t>(global_expert_para_size_), static_cast<int32_t>(moe_block_size),
              static_cast<size_t>(numel), rank_, stream);
        }
      }

      // ----- GEMM 1: hidden_states × w1 → intermediate_cache1 [tokens*topk, inter_size*2] -----
      int64_t size_k_w1 = static_cast<int64_t>(hidden_size);
      int64_t size_n_w1 = static_cast<int64_t>(expert_inter_size_ * 2);
      int64_t num_groups_w1 = size_k_w1 / 32;  // MXFP4 group size = 32

      // Zero workspace locks before kernel launch to prevent deadlock from stale lock values
      cudaMemsetAsync(marlin_workspace_, 0, marlin_workspace_size, stream);

      llm_kernels::nvidia::marlin_moe_wna16::moe_wna16_marlin_gemm(
          curr_hidden_states,                           // a
          intermediate_cache1_,                         // c
          c_tmp_,                                       // c_tmp (fp32 global reduce buffer)
          w1,                                           // b_q_weight
          nullptr,                                      // b_bias
          w1_scales,                                    // b_scales
          nullptr,                                      // global_scale
          nullptr,                                      // b_zeros (none for MXFP4)
          nullptr,                                      // g_idx (is_k_full=true)
          nullptr,                                      // perm (is_k_full=true)
          nullptr,                                      // a_tmp (no act_order)
          sorted_ids_ptr,                               // sorted_token_ids
          expert_ids_ptr,                               // expert_ids
          num_tokens_post_pad_ptr,                      // num_tokens_past_padded
          reinterpret_cast<float*>(curr_topk_weights),  // topk_weights
          marlin_workspace_,                            // workspace
          moe_block_size,                               // moe_block_size
          static_cast<int64_t>(expert_topk_),           // top_k
          false,                                        // mul_topk_weights (not for w1)
          use_expert_parallel,                          // is_ep
          ::marlin_moe_wna16::kFE2M1f.id(),             // b_q_type_id (MXFP4)
          3,                                            // scale_dtype (fp8_e8m0fnu)
          static_cast<int64_t>(tokens_in_chunk),        // size_m
          size_n_w1,                                    // size_n
          size_k_w1,                                    // size_k
          num_groups_w1,                                // num_groups
          max_num_tokens_padded,                        // sorted_token_ids_len
          0,                                            // g_idx_len (is_k_full)
          0,                                            // perm_len (is_k_full)
          0,                                            // b_zeros_len (no zeros)
          true,                                         // is_k_full
          false,                                        // use_atomic_add
          true,                                         // use_fp32_reduce
          false,                                        // is_zp_float
          marlin_compute_dtype,                         // compute_dtype
          stream);

      // ----- SiluAndMul: intermediate_cache1 → intermediate_cache2 -----
      const size_t elements_num = tokens_in_chunk * expert_topk_ * expert_inter_size_ * 2;
      if (use_expert_parallel) {
        if (has_swiglu_limit_) {
          llm_kernels::nvidia::InvokeSiluAndMul<T, true, true>(
              reinterpret_cast<const T*>(intermediate_cache1_), reinterpret_cast<T*>(intermediate_cache2_),
              reinterpret_cast<const int*>(curr_topk_ids), elements_num, expert_inter_size_, swiglu_limit_, stream);
        } else {
          llm_kernels::nvidia::InvokeSiluAndMul<T, true, false>(
              reinterpret_cast<const T*>(intermediate_cache1_), reinterpret_cast<T*>(intermediate_cache2_),
              reinterpret_cast<const int*>(curr_topk_ids), elements_num, expert_inter_size_, 0.0f, stream);
        }
      } else {
        if (has_swiglu_limit_) {
          llm_kernels::nvidia::InvokeSiluAndMul<T, false, true>(
              reinterpret_cast<const T*>(intermediate_cache1_), reinterpret_cast<T*>(intermediate_cache2_),
              reinterpret_cast<const int*>(curr_topk_ids), elements_num, expert_inter_size_, swiglu_limit_, stream);
        } else {
          llm_kernels::nvidia::InvokeSiluAndMul<T, false, false>(
              reinterpret_cast<const T*>(intermediate_cache1_), reinterpret_cast<T*>(intermediate_cache2_),
              reinterpret_cast<const int*>(curr_topk_ids), elements_num, expert_inter_size_, 0.0f, stream);
        }
      }

      // ----- GEMM 2: intermediate_cache2 × w2 → intermediate_cache3 [tokens*topk, hidden_size] -----
      int64_t size_k_w2 = static_cast<int64_t>(expert_inter_size_);
      int64_t size_n_w2 = static_cast<int64_t>(hidden_size);
      int64_t num_groups_w2 = size_k_w2 / 32;

      // Zero workspace locks before kernel launch to prevent deadlock from stale lock values
      cudaMemsetAsync(marlin_workspace_, 0, marlin_workspace_size, stream);

      llm_kernels::nvidia::marlin_moe_wna16::moe_wna16_marlin_gemm(
          intermediate_cache2_,                                  // a
          intermediate_cache3_,                                  // c
          c_tmp_,                                                // c_tmp (fp32 global reduce buffer)
          w2,                                                    // b_q_weight
          nullptr,                                               // b_bias
          w2_scales,                                             // b_scales
          nullptr,                                               // global_scale
          nullptr,                                               // b_zeros
          nullptr,                                               // g_idx
          nullptr,                                               // perm
          nullptr,                                               // a_tmp
          sorted_ids_ptr,                                        // sorted_token_ids
          expert_ids_ptr,                                        // expert_ids
          num_tokens_post_pad_ptr,                               // num_tokens_past_padded
          reinterpret_cast<float*>(curr_topk_weights),           // topk_weights
          marlin_workspace_,                                     // workspace
          moe_block_size,                                        // moe_block_size
          1,                                                     // top_k (reading from expanded cache)
          true,                                                  // mul_topk_weights (apply weights in GEMM)
          use_expert_parallel,                                   // is_ep
          ::marlin_moe_wna16::kFE2M1f.id(),                      // b_q_type_id
          3,                                                     // scale_dtype (fp8_e8m0fnu)
          static_cast<int64_t>(tokens_in_chunk * expert_topk_),  // size_m
          size_n_w2,                                             // size_n
          size_k_w2,                                             // size_k
          num_groups_w2,                                         // num_groups
          max_num_tokens_padded,                                 // sorted_token_ids_len
          0,                                                     // g_idx_len
          0,                                                     // perm_len
          0,                                                     // b_zeros_len
          true,                                                  // is_k_full
          false,                                                 // use_atomic_add
          true,                                                  // use_fp32_reduce
          false,                                                 // is_zp_float
          marlin_compute_dtype,                                  // compute_dtype
          stream);

      // ----- MoeSum: intermediate_cache3 → output [tokens, hidden_size] -----
      if (use_expert_parallel) {
        llm_kernels::nvidia::InvokeMoeSum<T, true>(intermediate_cache3_, curr_output, curr_topk_ids,
                                                   static_cast<int>(tokens_in_chunk), static_cast<int>(expert_topk_),
                                                   static_cast<int>(hidden_size), stream);
      } else {
        llm_kernels::nvidia::InvokeMoeSum<T, false>(intermediate_cache3_, curr_output, curr_topk_ids,
                                                    static_cast<int>(tokens_in_chunk), static_cast<int>(expert_topk_),
                                                    static_cast<int>(hidden_size), stream);
      }
    }
  };

  if (using_deepep_) {
    // LCOV_EXCL_START
    const size_t dispatch_num_tokens = output_tensors[1].shape[0];
    if (dispatch_num_tokens > 0) {
      expert_map_->InvokeExpertMapInverseInplace(static_cast<int32_t*>(topk_ids_ptr_),
                                                 dispatch_num_tokens * expert_topk_,
                                                 context_->device->GetPrimaryComputeStream().Get());
      ProcessChunks(output_tensors[1].GetPtr<void>(), output_tensors[1].GetPtr<void>(), dispatch_num_tokens);
    }
    Combine({output_tensors[1]}, output_tensors);
    // LCOV_EXCL_STOP
  } else if (global_expert_para_size_ > 1) {
    // EP without deepep
    expert_map_->InvokeExpertMapInplace(static_cast<int32_t*>(topk_ids_ptr_), num_tokens * expert_topk_,
                                        context_->device->GetPrimaryComputeStream().Get());
    ProcessChunks(input_tensors[0].GetPtr<void>(), output_tensors[0].GetPtr<void>(), num_tokens);
  } else {
    // Single-node, no EP
    ProcessChunks(input_tensors[0].GetPtr<void>(), output_tensors[0].GetPtr<void>(), num_tokens);
  }

  output_tensors[0].shape = input_tensors[0].shape;
  output_tensors[0].dtype = input_tensors[0].dtype;
  return Status();
}

template <typename T>
Status MarlinFp4MoeLayer::ExecuteGroupedTopk(const std::vector<Tensor>& input_tensors,
                                             std::vector<Tensor>& output_tensors) {
  std::vector<Tensor> grouped_topk_input_tensors;
  std::vector<Tensor> grouped_topk_output_tensors;

  size_t num_tokens = input_tensors[0].shape[0];

  // Input: gating_output
  grouped_topk_input_tensors.push_back(input_tensors[1]);

  // Input: e_bias (optional)
  if (input_tensors.size() > 5) {
    grouped_topk_input_tensors.push_back(input_tensors[5]);
  }

  // Output: topk_weights
  Tensor topk_weights_tensor(input_tensors[1].location, TYPE_FP32, {num_tokens, expert_topk_}, topk_weights_ptr_);
  grouped_topk_output_tensors.push_back(topk_weights_tensor);

  // Output: topk_ids
  Tensor topk_ids_tensor(input_tensors[1].location, TYPE_INT32, {num_tokens, expert_topk_}, topk_ids_ptr_);
  grouped_topk_output_tensors.push_back(topk_ids_tensor);

  Status status = grouped_topk_layer_->Forward(grouped_topk_input_tensors, grouped_topk_output_tensors);
  if (!status.OK()) {
    KLLM_LOG_ERROR << "MarlinFp4MoeLayer::ExecuteGroupedTopk ERROR: failed to forward grouped_topk_layer";
    return status;
  }

  if (enable_dump_eplb_data_) {
    DumpEplbData(topk_ids_tensor);
  }

  if (enable_load_eplb_config_ && input_tensors[4].shape[0] > layer_idx_) {
    size_t num_experts = input_tensors[4].shape[1];
    void* layer_expert_ptr = input_tensors[4].GetPtr<void>() + layer_idx_ * num_experts * sizeof(int);
    expert_map_->InvokeExpertMapInplace(static_cast<int32_t*>(topk_ids_ptr_), num_tokens * expert_topk_,
                                        reinterpret_cast<int32_t*>(layer_expert_ptr),
                                        context_->device->GetPrimaryComputeStream().Get());
  }

  if (using_deepep_) {
    // LCOV_EXCL_START
    std::vector<Tensor> deepep_input_tensors = {input_tensors[0], topk_ids_tensor, topk_weights_tensor};
    std::vector<Tensor>& deepep_output_tensors = output_tensors;
    Dispatch(deepep_input_tensors, deepep_output_tensors);
    topk_ids_tensor.shape[0] = deepep_output_tensors[1].shape[0];
    topk_weights_tensor.shape[0] = deepep_output_tensors[1].shape[0];
    // LCOV_EXCL_STOP
  }
  return Status();
}

template <typename T>
Status MarlinFp4MoeLayer::ExecuteTopkSoftplusSqrt(const std::vector<Tensor>& input_tensors,
                                                  std::vector<Tensor>& output_tensors) {
  std::vector<Tensor> softplus_inputs;
  std::vector<Tensor> softplus_outputs;

  const size_t num_tokens = input_tensors[0].shape[0];

  // Router logits
  softplus_inputs.push_back(input_tensors[1]);

  if (use_hash_) {
    KLLM_CHECK_WITH_INFO(input_tensors.size() >= 7,
                         "MarlinFp4MoeLayer: hash path requires input_ids (slot 6) and tid2eid (slot 5)");
    softplus_inputs.push_back(input_tensors[6]);  // input_ids
    softplus_inputs.push_back(input_tensors[5]);  // tid2eid
  } else if (use_e_score_correction_bias_) {
    KLLM_CHECK_WITH_INFO(input_tensors.size() >= 6, "MarlinFp4MoeLayer: correction-bias path requires bias at slot 5");
    softplus_inputs.push_back(input_tensors[5]);  // bias
  }

  Tensor topk_weights_tensor(input_tensors[1].location, TYPE_FP32, {num_tokens, static_cast<size_t>(expert_topk_)},
                             topk_weights_ptr_);
  Tensor topk_ids_tensor(input_tensors[1].location, TYPE_INT32, {num_tokens, static_cast<size_t>(expert_topk_)},
                         topk_ids_ptr_);
  softplus_outputs.push_back(topk_weights_tensor);
  softplus_outputs.push_back(topk_ids_tensor);

  Status status = topk_softplus_sqrt_layer_->Forward(softplus_inputs, softplus_outputs);
  if (!status.OK()) {
    KLLM_LOG_ERROR << "MarlinFp4MoeLayer::ExecuteTopkSoftplusSqrt ERROR: forward failed";
    return status;
  }

  if (enable_dump_eplb_data_) {
    DumpEplbData(topk_ids_tensor);
  }

  if (enable_load_eplb_config_ && input_tensors[4].shape[0] > layer_idx_) {
    size_t num_experts = input_tensors[4].shape[1];
    void* layer_expert_ptr = input_tensors[4].GetPtr<void>() + layer_idx_ * num_experts * sizeof(int);
    expert_map_->InvokeExpertMapInplace(static_cast<int32_t*>(topk_ids_ptr_), num_tokens * expert_topk_,
                                        reinterpret_cast<int32_t*>(layer_expert_ptr),
                                        context_->device->GetPrimaryComputeStream().Get());
  }

  if (using_deepep_) {
    // LCOV_EXCL_START
    std::vector<Tensor> deepep_input_tensors = {input_tensors[0], topk_ids_tensor, topk_weights_tensor};
    std::vector<Tensor>& deepep_output_tensors = output_tensors;
    Dispatch(deepep_input_tensors, deepep_output_tensors);
    topk_ids_tensor.shape[0] = deepep_output_tensors[1].shape[0];
    topk_weights_tensor.shape[0] = deepep_output_tensors[1].shape[0];
    // LCOV_EXCL_STOP
  }

  return Status();
}

Status MarlinFp4MoeLayer::Dispatch(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  return expert_parallel_manager_->Dispatch(input_tensors, output_tensors, context_->device->GetPrimaryComputeStream());
}

Status MarlinFp4MoeLayer::Combine(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  return expert_parallel_manager_->Combine(input_tensors, output_tensors, context_->device->GetPrimaryComputeStream());
}

Status MarlinFp4MoeLayer::DumpEplbData(Tensor& topk_ids) {
  topk_ids.SaveToNpyFile(
      fmt::format("{}/layer_{}/topk_ids_{}_{}.npy", eplb_dump_path_, layer_idx_, eplb_dump_step_, rank_));
  eplb_dump_step_ += 1;
  return Status();
}

}  // namespace ksana_llm
