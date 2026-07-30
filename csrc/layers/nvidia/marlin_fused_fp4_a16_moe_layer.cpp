/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "layers/marlin_fused_fp4_a16_moe_layer.h"

#include "csrc/kernels/nvidia/moe_utils/moe_utils.h"
#include "csrc/kernels/nvidia/others/sglang/main/marlin_moe_wna16/marlin_moe_wna16_wrapper.h"
#include "csrc/kernels/nvidia/others/sglang/main/marlin_moe_wna16/ops.h"
#include "csrc/kernels/nvidia/others/sglang/main/marlin_moe_wna16/scalar_type.hpp"
#include "layers/grouped_topk_layer.h"
#include "layers/topk_softplus_sqrt_layer.h"
#include "utils/utils.h"

namespace ksana_llm {

#define MARLIN_FP4_MOE_CHUNK_SIZE ((size_t)(32 * 1024))

// Maximum MOE block size for workspace pre-allocation.  SelectMoeBlockSize can return
// up to 64 at high token density, so the workspace must be sized for the worst case.
// Must be >= the largest value in SelectMoeBlockSize's kBlockSizes[] (currently 64).
static constexpr int64_t kMarlinFp4MoeMaxBlockSize = 64;

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

Status MarlinFusedFp4A16MoeLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                       std::shared_ptr<Context> context, int rank) {
  inter_data_type_ = runtime_config.inter_data_type;
  DISPATCH_BY_3_DTYPE(inter_data_type_, InitT, parameters, runtime_config, context, rank);
}

template <typename T>
Status MarlinFusedFp4A16MoeLayer::InitT(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
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
  situ_beta_ = std::any_cast<float>(parameters[parameter_index++]);
  situ_linear_beta_ = std::any_cast<float>(parameters[parameter_index++]);

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

  KLLM_LOG_INFO << fmt::format("Rank[{}] MarlinFusedFp4A16MoeLayer Init", rank_);
  return Status();
}

void MarlinFusedFp4A16MoeLayer::SetExpertParallelManager(
    std::shared_ptr<ExpertParallelManager> expert_parallel_manager) {
  expert_parallel_manager_ = expert_parallel_manager;
}

size_t MarlinFusedFp4A16MoeLayer::GetWorkspaceSize() { DISPATCH_BY_3_DTYPE(inter_data_type_, GetWorkspaceSizeT); }

template <typename T>
size_t MarlinFusedFp4A16MoeLayer::GetWorkspaceSizeT() {
  size_t m = std::min(MARLIN_FP4_MOE_CHUNK_SIZE, max_token_num_);

  topk_weights_ptr_size = AlignAddress(max_token_num_ * expert_topk_ * sizeof(float));
  topk_ids_ptr_size = AlignAddress(max_token_num_ * expert_topk_ * sizeof(int32_t));

  // fused_id_buffer: fill_info_vec(4) + sorted_ids + expert_ids + num_tokens_post_pad + cumsum
  size_t numel = m * expert_topk_;
  size_t max_num_tokens_padded = numel + expert_num_per_node_ * (kMarlinFp4MoeMaxBlockSize - 1);
  size_t max_num_m_blocks = (max_num_tokens_padded + kMarlinFp4MoeMaxBlockSize - 1) / kMarlinFp4MoeMaxBlockSize;
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
                                           max_n, max_num_tokens_padded, kMarlinFp4MoeMaxBlockSize) *
                                       sizeof(int));

  // c_tmp buffer for fp32 global reduce (P1 optimization)
  // Matches sglang: use_fp32_reduce=true requires a fp32 temp buffer for cross-SM reduction
  c_tmp_size =
      AlignAddress(llm_kernels::nvidia::marlin_moe_wna16::get_c_tmp_size(
                       max_n, max_num_tokens_padded, kMarlinFp4MoeMaxBlockSize, ::marlin_moe_wna16::kFE2M1f.id()) *
                   sizeof(float));

  const size_t softplus_sqrt_ws_aligned = use_topk_softplus_sqrt_ ? AlignAddress(softplus_sqrt_workspace_size_) : 0;

  // Total workspace: cache1 is shared with cache3, so we use max(cache1, cache3) instead of sum
  size_t cache13_size = std::max(intermediate_cache1_size, intermediate_cache3_size);
  max_ws_bytes_ = topk_weights_ptr_size + topk_ids_ptr_size + max_fused_id_buffer_size + cache13_size +
                  intermediate_cache2_size + marlin_workspace_size + c_tmp_size + softplus_sqrt_ws_aligned;

  KLLM_LOG_DEBUG << fmt::format("Rank[{}] Request {} for MarlinFusedFp4A16MoeLayer", rank_, max_ws_bytes_);
  return max_ws_bytes_;
}

Status MarlinFusedFp4A16MoeLayer::Preprocess(const ModelConfig& model_config, const RuntimeConfig& runtime_config) {
  // Marlin FP4 kernel does not require profiling/preprocessing
  return Status();
}

template <typename T>
Status MarlinFusedFp4A16MoeLayer::ForwardT(const Tensor* hidden_states, const Tensor* gating_output, const Tensor* w1,
                                           const Tensor* w2, const Tensor* eplb_map, Tensor* output, Tensor* workspace,
                                           const Tensor* e_score_bias, const Tensor* input_ids) {
  // Lazy creation of profiling CUDA events (created once, reused across forwards)
  bool do_profile = (perf_stats_ != nullptr);
  if (do_profile) {
    if (evt_gemm1_start_ == nullptr) {
      cudaEventCreate(&evt_gemm1_start_);
      cudaEventCreate(&evt_gemm1_end_);
      cudaEventCreate(&evt_gemm2_start_);
      cudaEventCreate(&evt_gemm2_end_);
    }
  }

  // Assign workspace pointers on first call
  if (set_workspace_buffer_info_) {
    set_workspace_buffer_info_ = false;
    topk_weights_ptr_ = workspace_buffer_->GetPtr<void>();
    topk_ids_ptr_ = reinterpret_cast<uint8_t*>(topk_weights_ptr_) + topk_weights_ptr_size;
    fused_id_buffer_ = reinterpret_cast<uint8_t*>(topk_ids_ptr_) + topk_ids_ptr_size;
    intermediate_cache1_ = reinterpret_cast<uint8_t*>(fused_id_buffer_) + max_fused_id_buffer_size;
    intermediate_cache2_ =
        reinterpret_cast<uint8_t*>(intermediate_cache1_) + std::max(intermediate_cache1_size, intermediate_cache3_size);
    // cache3 shares memory with cache1 (non-overlapping lifetimes: cache1 consumed before cache3 produced)
    intermediate_cache3_ = intermediate_cache1_;
    marlin_workspace_ =
        reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(intermediate_cache2_) + intermediate_cache2_size);
    c_tmp_ = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(marlin_workspace_) + marlin_workspace_size);

    if (use_topk_softplus_sqrt_ && softplus_sqrt_workspace_size_ > 0) {
      const size_t slice_offset = max_ws_bytes_ - AlignAddress(softplus_sqrt_workspace_size_);
      auto slice = std::make_shared<Tensor>(
          workspace_buffer_->location, TYPE_UINT8, std::vector<size_t>{softplus_sqrt_workspace_size_},
          reinterpret_cast<int8_t*>(workspace_buffer_->GetPtr<void>()) + slice_offset);
      topk_softplus_sqrt_layer_->SetWorkspaceBuffer(slice);
    }
  }

  const size_t num_tokens = hidden_states->shape[0];
  const size_t hidden_size = hidden_states->shape[1];

  // Execute TopK gating
  if (use_topk_softplus_sqrt_) {
    STATUS_CHECK_RETURN(
        ExecuteTopkSoftplusSqrt<T>(hidden_states, gating_output, eplb_map, workspace, e_score_bias, input_ids));
  } else {
    STATUS_CHECK_RETURN(ExecuteGroupedTopk<T>(hidden_states, gating_output, eplb_map, workspace, e_score_bias));
  }

  // Get weight pointers
  void* w1_ptr = w1->GetPtr<void>();
  void* w2_ptr = w2->GetPtr<void>();
  void* w1_scales = (w1->weight_scales != nullptr) ? w1->weight_scales->GetPtr<void>() : nullptr;
  void* w2_scales = (w2->weight_scales != nullptr) ? w2->weight_scales->GetPtr<void>() : nullptr;

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

      // Dynamically select block size based on token density per expert (matches sglang heuristic)
      int64_t moe_block_size = SelectMoeBlockSize(tokens_in_chunk, expert_topk_, expert_num_per_node_);
      if (do_profile) {
        perf_stats_->last_moe_block_size_1 = static_cast<int32_t>(moe_block_size);
        perf_stats_->last_moe_block_size_2 = static_cast<int32_t>(moe_block_size);
      }

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

      if (do_profile) cudaEventRecord(evt_gemm1_start_, stream);

      llm_kernels::nvidia::marlin_moe_wna16::moe_wna16_marlin_gemm(
          curr_hidden_states,                           // a
          intermediate_cache1_,                         // c
          c_tmp_,                                       // c_tmp (fp32 global reduce buffer)
          w1_ptr,                                       // b_q_weight
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

      if (do_profile) cudaEventRecord(evt_gemm1_end_, stream);

      // ----- Activation: intermediate_cache1 → intermediate_cache2 -----
      const size_t elements_num = tokens_in_chunk * expert_topk_ * expert_inter_size_ * 2;
      if (situ_beta_ > 0.0f) {
        if (use_expert_parallel) {
          llm_kernels::nvidia::InvokeSituAndMul<T, true>(
              reinterpret_cast<const T*>(intermediate_cache1_), reinterpret_cast<T*>(intermediate_cache2_),
              reinterpret_cast<const int*>(curr_topk_ids), elements_num, expert_inter_size_, situ_beta_,
              situ_linear_beta_, stream);
        } else {
          llm_kernels::nvidia::InvokeSituAndMul<T, false>(
              reinterpret_cast<const T*>(intermediate_cache1_), reinterpret_cast<T*>(intermediate_cache2_),
              reinterpret_cast<const int*>(curr_topk_ids), elements_num, expert_inter_size_, situ_beta_,
              situ_linear_beta_, stream);
        }
      } else if (use_expert_parallel) {
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

      if (do_profile) cudaEventRecord(evt_gemm2_start_, stream);

      llm_kernels::nvidia::marlin_moe_wna16::moe_wna16_marlin_gemm(
          intermediate_cache2_,                                  // a
          intermediate_cache3_,                                  // c
          c_tmp_,                                                // c_tmp (fp32 global reduce buffer)
          w2_ptr,                                                // b_q_weight
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

      if (do_profile) cudaEventRecord(evt_gemm2_end_, stream);

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

      // Accumulate per-chunk GEMM times
      // NOTE: need to sync the latest event before reading elapsed times — all prior stream
      // events are guaranteed complete once the last event is done since kernels execute in order.
      if (do_profile) {
        cudaEventSynchronize(evt_gemm2_end_);
        float gemm1_ms = 0.f, gemm2_ms = 0.f;
        cudaEventElapsedTime(&gemm1_ms, evt_gemm1_start_, evt_gemm1_end_);
        cudaEventElapsedTime(&gemm2_ms, evt_gemm2_start_, evt_gemm2_end_);
        perf_stats_->gemm1_time_ms += gemm1_ms;
        perf_stats_->gemm2_time_ms += gemm2_ms;
      }
    }
  };

  if (using_deepep_) {
    // LCOV_EXCL_START
    const size_t dispatch_num_tokens = workspace->shape[0];
    if (dispatch_num_tokens > 0) {
      expert_map_->InvokeExpertMapInverseInplace(static_cast<int32_t*>(topk_ids_ptr_),
                                                 dispatch_num_tokens * expert_topk_,
                                                 context_->device->GetPrimaryComputeStream().Get());
      ProcessChunks(workspace->GetPtr<void>(), workspace->GetPtr<void>(), dispatch_num_tokens);
    }
    std::vector<Tensor> combine_outputs = {*output};
    Combine({*workspace}, combine_outputs);
    *output = combine_outputs[0];
    // LCOV_EXCL_STOP
  } else if (global_expert_para_size_ > 1) {
    // EP without deepep
    expert_map_->InvokeExpertMapInplace(static_cast<int32_t*>(topk_ids_ptr_), num_tokens * expert_topk_,
                                        context_->device->GetPrimaryComputeStream().Get());
    ProcessChunks(hidden_states->GetPtr<void>(), output->GetPtr<void>(), num_tokens);
  } else {
    // Single-node, no EP
    ProcessChunks(hidden_states->GetPtr<void>(), output->GetPtr<void>(), num_tokens);
  }

  output->shape = hidden_states->shape;
  output->dtype = hidden_states->dtype;
  return Status();
}

template <typename T>
Status MarlinFusedFp4A16MoeLayer::ExecuteGroupedTopk(const Tensor* hidden_states, const Tensor* gating_output,
                                                     const Tensor* eplb_map, Tensor* workspace,
                                                     const Tensor* e_score_bias) {
  const size_t num_tokens = hidden_states->shape[0];
  Tensor topk_weights_tensor(gating_output->location, TYPE_FP32, {num_tokens, expert_topk_}, topk_weights_ptr_);
  Tensor topk_ids_tensor(gating_output->location, TYPE_INT32, {num_tokens, expert_topk_}, topk_ids_ptr_);
  Status status = grouped_topk_layer_->Forward(gating_output, &topk_weights_tensor, &topk_ids_tensor, e_score_bias);
  if (!status.OK()) {
    KLLM_LOG_ERROR << "MarlinFusedFp4A16MoeLayer::ExecuteGroupedTopk ERROR: failed to forward grouped_topk_layer";
    return status;
  }

  if (enable_dump_eplb_data_) {
    DumpEplbData(topk_ids_tensor);
  }

  if (enable_load_eplb_config_ && eplb_map->shape[0] > static_cast<size_t>(layer_idx_)) {
    size_t num_experts = eplb_map->shape[1];
    void* layer_expert_ptr =
        reinterpret_cast<uint8_t*>(eplb_map->GetPtr<void>()) + layer_idx_ * num_experts * sizeof(int);
    expert_map_->InvokeExpertMapInplace(static_cast<int32_t*>(topk_ids_ptr_), num_tokens * expert_topk_,
                                        reinterpret_cast<int32_t*>(layer_expert_ptr),
                                        context_->device->GetPrimaryComputeStream().Get());
  }

  if (using_deepep_) {
    // LCOV_EXCL_START
    std::vector<Tensor> deepep_input_tensors = {*hidden_states, topk_ids_tensor, topk_weights_tensor};
    std::vector<Tensor> deepep_output_tensors = {*workspace};
    Dispatch(deepep_input_tensors, deepep_output_tensors);
    topk_ids_tensor.shape[0] = deepep_output_tensors[0].shape[0];
    topk_weights_tensor.shape[0] = deepep_output_tensors[0].shape[0];
    workspace->shape = deepep_output_tensors[0].shape;
    // LCOV_EXCL_STOP
  }
  return Status();
}

template <typename T>
Status MarlinFusedFp4A16MoeLayer::ExecuteTopkSoftplusSqrt(const Tensor* hidden_states, const Tensor* gating_output,
                                                          const Tensor* eplb_map, Tensor* workspace,
                                                          const Tensor* e_score_bias, const Tensor* input_ids) {
  const size_t num_tokens = hidden_states->shape[0];

  // aux1: input_ids (hash) or correction_bias; aux2: tid2eid (hash only).
  const Tensor* aux1_ptr = nullptr;
  const Tensor* aux2_ptr = nullptr;
  if (use_hash_) {
    KLLM_CHECK_WITH_INFO(input_ids != nullptr,
                         "MarlinFusedFp4A16MoeLayer: hash path requires input_ids and e_score_bias (tid2eid)");
    aux1_ptr = input_ids;     // input_ids
    aux2_ptr = e_score_bias;  // tid2eid
  } else if (use_e_score_correction_bias_) {
    aux1_ptr = e_score_bias;  // correction_bias
  }

  Tensor topk_weights_tensor(gating_output->location, TYPE_FP32, {num_tokens, static_cast<size_t>(expert_topk_)},
                             topk_weights_ptr_);
  Tensor topk_ids_tensor(gating_output->location, TYPE_INT32, {num_tokens, static_cast<size_t>(expert_topk_)},
                         topk_ids_ptr_);
  Status status =
      topk_softplus_sqrt_layer_->Forward(gating_output, &topk_weights_tensor, &topk_ids_tensor, aux1_ptr, aux2_ptr);
  if (!status.OK()) {
    KLLM_LOG_ERROR << "MarlinFusedFp4A16MoeLayer::ExecuteTopkSoftplusSqrt ERROR: forward failed";
    return status;
  }

  if (enable_dump_eplb_data_) {
    DumpEplbData(topk_ids_tensor);
  }

  if (enable_load_eplb_config_ && eplb_map->shape[0] > static_cast<size_t>(layer_idx_)) {
    size_t num_experts = eplb_map->shape[1];
    void* layer_expert_ptr =
        reinterpret_cast<uint8_t*>(eplb_map->GetPtr<void>()) + layer_idx_ * num_experts * sizeof(int);
    expert_map_->InvokeExpertMapInplace(static_cast<int32_t*>(topk_ids_ptr_), num_tokens * expert_topk_,
                                        reinterpret_cast<int32_t*>(layer_expert_ptr),
                                        context_->device->GetPrimaryComputeStream().Get());
  }

  if (using_deepep_) {
    // LCOV_EXCL_START
    std::vector<Tensor> deepep_input_tensors = {*hidden_states, topk_ids_tensor, topk_weights_tensor};
    std::vector<Tensor> deepep_output_tensors = {*workspace};
    Dispatch(deepep_input_tensors, deepep_output_tensors);
    topk_ids_tensor.shape[0] = deepep_output_tensors[0].shape[0];
    topk_weights_tensor.shape[0] = deepep_output_tensors[0].shape[0];
    workspace->shape = deepep_output_tensors[0].shape;
    // LCOV_EXCL_STOP
  }

  return Status();
}

Status MarlinFusedFp4A16MoeLayer::Dispatch(const std::vector<Tensor>& input_tensors,
                                           std::vector<Tensor>& output_tensors) {
  return expert_parallel_manager_->Dispatch(input_tensors, output_tensors, context_->device->GetPrimaryComputeStream());
}

Status MarlinFusedFp4A16MoeLayer::Combine(const std::vector<Tensor>& input_tensors,
                                          std::vector<Tensor>& output_tensors) {
  return expert_parallel_manager_->Combine(input_tensors, output_tensors, context_->device->GetPrimaryComputeStream());
}

Status MarlinFusedFp4A16MoeLayer::DumpEplbData(Tensor& topk_ids) {
  topk_ids.SaveToNpyFile(
      fmt::format("{}/layer_{}/topk_ids_{}_{}.npy", eplb_dump_path_, layer_idx_, eplb_dump_step_, rank_));
  eplb_dump_step_ += 1;
  return Status();
}

Status MarlinFusedFp4A16MoeLayer::Forward(const Tensor* hidden_states, const Tensor* gating_output, const Tensor* w1,
                                          const Tensor* w2, const Tensor* eplb_map, Tensor* output, Tensor* workspace,
                                          const Tensor* e_score_bias, const Tensor* input_ids,
                                          const LayerExecContext& exec_ctx) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, hidden_states, gating_output, w1, w2, eplb_map, output, workspace,
                      e_score_bias, input_ids);
}

}  // namespace ksana_llm
