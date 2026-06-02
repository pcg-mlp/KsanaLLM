/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "layers/humming_fp4_moe_layer.h"

#include "csrc/kernels/nvidia/moe_utils/moe_utils.h"
#include "csrc/kernels/nvidia/others/humming/jit/humming_jit_compiler.h"
#include "csrc/kernels/nvidia/others/humming/ops/ops.h"
#include "csrc/kernels/nvidia/others/humming/ops/wrapper.h"
#include "layers/topk_softplus_sqrt_layer.h"
#include "utils/utils.h"

namespace ksana_llm {

// Static member definitions
std::shared_ptr<llm_kernels::nvidia::humming::HummingJitCompiler> HummingFp4MoeLayer::jit_compiler_;
std::mutex HummingFp4MoeLayer::jit_mutex_;

// Maximum moe_block_size for workspace pre-allocation (actual value is determined at runtime).
// Must be >= the largest possible BLOCK_SHAPE_M from the H20 heuristic (currently 64).
static constexpr int64_t kHummingMoeMaxBlockSize = 64;

inline size_t AlignAddress(size_t size) { return (size + 255) & (~255); }

Status HummingFp4MoeLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                std::shared_ptr<Context> context, int rank) {
  inter_data_type_ = runtime_config.inter_data_type;
  DISPATCH_BY_3_DTYPE(inter_data_type_, InitT, parameters, runtime_config, context, rank);
}

template <typename T>
Status HummingFp4MoeLayer::InitT(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
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
  [[maybe_unused]] uint32_t num_expert_group = std::any_cast<uint32_t>(parameters[parameter_index++]);
  [[maybe_unused]] uint32_t expert_groups_topk = std::any_cast<uint32_t>(parameters[parameter_index++]);
  scoring_func_ = std::any_cast<std::string>(parameters[parameter_index++]);
  [[maybe_unused]] std::string topk_method = std::any_cast<std::string>(parameters[parameter_index++]);
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

  KLLM_CHECK_WITH_INFO(scoring_func_ == "sqrtsoftplus",
                       fmt::format("HummingFp4MoeLayer requires scoring_func='sqrtsoftplus', got '{}'", scoring_func_));

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

  // Determine humming compute dtype from runtime config
  humming_compute_dtype_ = (runtime_config.fp4_moe_backend == FP4_MOE_BACKEND_TYPE::FP4HummingW4A8)
                               ? llm_kernels::nvidia::humming::ComputeDtype::FP8
                               : llm_kernels::nvidia::humming::ComputeDtype::BF16;

  // Initialize JIT compiler (lazy singleton with mutex).
  // W4A16 and W4A8 share the same compiler but need different compute_dtype kernels.
  // The first Init creates the compiler; subsequent Inits add their own kernels.
  {
    std::lock_guard<std::mutex> lock(jit_mutex_);
    if (!jit_compiler_) {
      jit_compiler_ = std::make_shared<llm_kernels::nvidia::humming::HummingJitCompiler>();
      // Register the JIT compiler with the ops dispatch layer so humming_moe_gemm can find compiled kernels
      llm_kernels::nvidia::humming::SetJitCompiler(jit_compiler_.get());
    }
  }
  // JIT compiler handles shape-level dedup internally — no layer-side flag needed.
  int64_t n1 = expert_inter_size_ * 2, k1 = expert_hidden_size_;  // w1: gate+up
  int64_t n2 = expert_hidden_size_, k2 = expert_inter_size_;      // w2: down
  auto gemm_type = llm_kernels::nvidia::humming::GemmType::INDEXED;

  jit_compiler_->CompileKernels(n1, k1, static_cast<int64_t>(expert_num_per_node_), humming_compute_dtype_, gemm_type);
  jit_compiler_->CompileKernels(n2, k2, static_cast<int64_t>(expert_num_per_node_), humming_compute_dtype_, gemm_type);

  KLLM_LOG_INFO << fmt::format("Rank[{}] HummingFp4MoeLayer Init", rank_);
  return Status();
}

void HummingFp4MoeLayer::SetExpertParallelManager(std::shared_ptr<ExpertParallelManager> expert_parallel_manager) {
  expert_parallel_manager_ = expert_parallel_manager;
}

size_t HummingFp4MoeLayer::GetWorkspaceSize() { DISPATCH_BY_3_DTYPE(inter_data_type_, GetWorkspaceSizeT); }

template <typename T>
size_t HummingFp4MoeLayer::GetWorkspaceSizeT() {
  size_t m = max_token_num_;

  topk_weights_ptr_size_ = AlignAddress(max_token_num_ * expert_topk_ * sizeof(float));
  topk_ids_ptr_size_ = AlignAddress(max_token_num_ * expert_topk_ * sizeof(int32_t));

  // fused_id_buffer: fill_info_vec(4) + sorted_ids + expert_ids + num_tokens_post_pad + cumsum
  size_t numel = m * expert_topk_;
  size_t max_num_tokens_padded = numel + expert_num_per_node_ * (kHummingMoeMaxBlockSize - 1);
  size_t max_num_m_blocks = (max_num_tokens_padded + kHummingMoeMaxBlockSize - 1) / kHummingMoeMaxBlockSize;
  // fill_info(4) + sorted_ids(max_num_tokens_padded) + expert_ids(max_num_m_blocks) + num_tokens_post_pad(1) +
  // cumsum(expert_num_per_node+1)
  max_fused_id_buffer_size_ = AlignAddress(
      (SGL_MOE_ALIGN_BLOCK_VEC_SIZE + max_num_tokens_padded + max_num_m_blocks + 1 + expert_num_per_node_ + 1) *
      sizeof(int32_t));

  // Intermediate caches
  // cache1: [m * topk, inter_size * 2] — output of w1 GEMM (gate+up)
  intermediate_cache1_size_ = AlignAddress(m * expert_topk_ * expert_inter_size_ * 2 * sizeof(T));
  // cache2: [m * topk, inter_size] — output of SiluAndMul
  intermediate_cache2_size_ = AlignAddress(m * expert_topk_ * expert_inter_size_ * sizeof(T));
  // cache3: [m * topk, hidden_size] — output of w2 GEMM
  // NOTE: cache3 reuses cache1's memory (P5 optimization) since their lifetimes don't overlap:
  //   cache1 is consumed by SiluAndMul before cache3 is produced by GEMM2.
  intermediate_cache3_size_ = AlignAddress(m * expert_topk_ * expert_hidden_size_ * sizeof(T));

  // Humming workspace: sized for the larger of the two GEMMs
  int64_t max_n = std::max(expert_inter_size_ * 2, expert_hidden_size_);
  int64_t max_k = std::max(expert_hidden_size_, expert_inter_size_);
  humming_locks_size_ = AlignAddress(llm_kernels::nvidia::humming::get_locks_size(max_token_num_ * expert_topk_, max_n,
                                                                                  max_k, humming_compute_dtype_) *
                                     sizeof(int32_t));
  humming_workspace_size_ = AlignAddress(llm_kernels::nvidia::humming::get_workspace_size(
      max_token_num_ * expert_topk_, max_n, max_k, expert_num_per_node_, expert_topk_, humming_compute_dtype_));

  // W4A8 FP8 activation buffers
  fp8_activation_buffer_size_ = 0;
  fp8_scale_buffer_size_ = 0;
  if (humming_compute_dtype_ == llm_kernels::nvidia::humming::ComputeDtype::FP8) {
    // FP8 buffer: max of GEMM1 input [M, hidden_size] and GEMM2 input [M*topk, inter_size]
    fp8_activation_buffer_size_ = AlignAddress(std::max(static_cast<int64_t>(m * expert_hidden_size_),
                                                        static_cast<int64_t>(m * expert_topk_ * expert_inter_size_)) *
                                               sizeof(uint8_t));
    // Per-token scale buffer: max tokens = M * topk (GEMM2 has more tokens)
    fp8_scale_buffer_size_ = AlignAddress(static_cast<int64_t>(m * expert_topk_) * sizeof(float));
  }

  const size_t softplus_sqrt_ws_aligned = AlignAddress(softplus_sqrt_workspace_size_);

  // Total workspace: cache1 is shared with cache3, so we use max(cache1, cache3) instead of sum
  size_t cache13_size = std::max(intermediate_cache1_size_, intermediate_cache3_size_);
  max_ws_bytes_ = topk_weights_ptr_size_ + topk_ids_ptr_size_ + max_fused_id_buffer_size_ + cache13_size +
                  intermediate_cache2_size_ + humming_locks_size_ + humming_workspace_size_ + softplus_sqrt_ws_aligned +
                  fp8_activation_buffer_size_ + fp8_scale_buffer_size_;

  KLLM_LOG_DEBUG << fmt::format("Rank[{}] Request {} for HummingFp4MoeLayer", rank_, max_ws_bytes_);
  return max_ws_bytes_;
}

Status HummingFp4MoeLayer::Preprocess(const ModelConfig& model_config, const RuntimeConfig& runtime_config) {
  // Humming FP4 kernel does not require profiling/preprocessing
  return Status();
}

Status HummingFp4MoeLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors,
                                   const LayerExecContext& exec_ctx) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, input_tensors, output_tensors);
}

template <typename T>
Status HummingFp4MoeLayer::ForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
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
    topk_ids_ptr_ = topk_weights_ptr_ + topk_weights_ptr_size_;
    fused_id_buffer_ = topk_ids_ptr_ + topk_ids_ptr_size_;
    intermediate_cache1_ = fused_id_buffer_ + max_fused_id_buffer_size_;
    intermediate_cache2_ = intermediate_cache1_ + std::max(intermediate_cache1_size_, intermediate_cache3_size_);
    // cache3 shares memory with cache1 (non-overlapping lifetimes: cache1 consumed before cache3 produced)
    intermediate_cache3_ = intermediate_cache1_;
    humming_locks_ = reinterpret_cast<int32_t*>(intermediate_cache2_ + intermediate_cache2_size_);
    humming_workspace_ = reinterpret_cast<char*>(humming_locks_) + humming_locks_size_;

    // W4A8 FP8 buffers (after humming_workspace_)
    if (humming_compute_dtype_ == llm_kernels::nvidia::humming::ComputeDtype::FP8) {
      fp8_activation_buffer_ = reinterpret_cast<char*>(humming_workspace_) + humming_workspace_size_;
      fp8_scale_buffer_ =
          reinterpret_cast<float*>(reinterpret_cast<char*>(fp8_activation_buffer_) + fp8_activation_buffer_size_);
    }

    if (softplus_sqrt_workspace_size_ > 0) {
      const size_t slice_offset = max_ws_bytes_ - AlignAddress(softplus_sqrt_workspace_size_);
      auto slice = std::make_shared<Tensor>(
          workspace_buffer_->location, TYPE_UINT8, std::vector<size_t>{softplus_sqrt_workspace_size_},
          reinterpret_cast<int8_t*>(workspace_buffer_->GetPtr<void>()) + slice_offset);
      topk_softplus_sqrt_layer_->SetWorkspaceBuffer(slice);
    }
  }

  const size_t num_tokens = input_tensors[0].shape[0];
  const size_t hidden_size = input_tensors[0].shape[1];

  // Execute TopK gating (sqrtsoftplus only)
  STATUS_CHECK_RETURN(ExecuteTopkSoftplusSqrt<T>(input_tensors, output_tensors));

  // Get weight pointers
  void* w1 = input_tensors[2].GetPtr<void>();
  void* w2 = input_tensors[3].GetPtr<void>();
  void* w1_scales =
      (input_tensors[2].weight_scales != nullptr) ? input_tensors[2].weight_scales->GetPtr<void>() : nullptr;
  void* w2_scales =
      (input_tensors[3].weight_scales != nullptr) ? input_tensors[3].weight_scales->GetPtr<void>() : nullptr;

  // W4A8: Get per-expert global_scale from weight tensors
  if (humming_compute_dtype_ == llm_kernels::nvidia::humming::ComputeDtype::FP8) {
    if (input_tensors[2].weight_global_scale != nullptr) {
      w1_global_scale_ = input_tensors[2].weight_global_scale->GetPtr<float>();
    }
    if (input_tensors[3].weight_global_scale != nullptr) {
      w2_global_scale_ = input_tensors[3].weight_global_scale->GetPtr<float>();
    }
  }

  auto ProcessTokens = [&](void* hidden_states_ptr, void* output_ptr, size_t total_tokens) {
    cudaStream_t stream = context_->device->GetPrimaryComputeStream().Get();
    const bool use_expert_parallel = (global_expert_para_size_ > 1 && !using_deepep_);

    // No chunking needed — humming uses stream-K internally, process all tokens at once
    void* curr_topk_ids = topk_ids_ptr_;
    void* curr_topk_weights = topk_weights_ptr_;

    // ----- Determine moe_block_size from GEMM1's kernel BLOCK_SHAPE_M -----
    // The humming kernel reads expert_ids[row / BLOCK_SHAPE_M], so expert boundaries
    // MUST be padded to multiples of BLOCK_SHAPE_M. We use GEMM1's block_m as the
    // shared moe_block_size, and force GEMM2 to use a kernel with the same block_m.
    int64_t size_n_w1 = static_cast<int64_t>(expert_inter_size_ * 2);
    int64_t size_k_w1 = static_cast<int64_t>(hidden_size);
    int32_t moe_block_size_i32 = llm_kernels::nvidia::humming::get_kernel_block_m(
        static_cast<int64_t>(total_tokens), size_n_w1, size_k_w1, llm_kernels::nvidia::humming::GemmType::INDEXED,
        humming_compute_dtype_);
    int64_t moe_block_size = static_cast<int64_t>(moe_block_size_i32);

    int64_t numel = static_cast<int64_t>(total_tokens * expert_topk_);
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

    // W4A8: Quantize activations to FP8 before GEMM1
    const void* gemm1_a = hidden_states_ptr;
    const void* gemm1_a_scale = nullptr;
    const void* gemm1_global_scale = nullptr;

    if (humming_compute_dtype_ == llm_kernels::nvidia::humming::ComputeDtype::FP8) {
      // Zero-initialize scale buffer to avoid NaN from uninitialized memory
      cudaMemsetAsync(fp8_scale_buffer_, 0, fp8_scale_buffer_size_, stream);
      llm_kernels::nvidia::InvokeDynamicPerTokenScaledFP8Quant<T>(
          reinterpret_cast<const T*>(hidden_states_ptr), fp8_scale_buffer_,
          nullptr,  // scale_ub (no upper bound)
          fp8_activation_buffer_, static_cast<int>(size_k_w1), static_cast<int>(total_tokens), stream);
      gemm1_a = fp8_activation_buffer_;
      gemm1_a_scale = fp8_scale_buffer_;
      gemm1_global_scale = w1_global_scale_;
    }

    // Zero locks buffer before kernel launch to prevent deadlock from stale lock values
    cudaMemsetAsync(humming_locks_, 0, humming_locks_size_, stream);

    llm_kernels::nvidia::humming::humming_moe_gemm(gemm1_a,                  // a
                                                   w1,                       // b (FP4 WGMMA layout)
                                                   intermediate_cache1_,     // c
                                                   gemm1_a_scale,            // a_scale (nullptr for W4A16)
                                                   w1_scales,                // b_scale
                                                   gemm1_global_scale,       // global_scale
                                                   nullptr,                  // b_zero_point
                                                   nullptr,                  // bias
                                                   sorted_ids_ptr,           // sorted_ids
                                                   expert_ids_ptr,           // expert_ids
                                                   num_tokens_post_pad_ptr,  // num_tokens_padded
                                                   nullptr,                  // expert_layout (INDEXED uses sorted_ids)
                                                   reinterpret_cast<const float*>(curr_topk_weights),  // topk_weights
                                                   humming_locks_,                                     // locks
                                                   humming_workspace_,                                 // workspace
                                                   llm_kernels::nvidia::humming::GemmType::INDEXED,
                                                   humming_compute_dtype_,
                                                   static_cast<int64_t>(total_tokens),          // size_m
                                                   size_n_w1,                                   // size_n
                                                   size_k_w1,                                   // size_k
                                                   static_cast<int64_t>(expert_num_per_node_),  // num_experts
                                                   static_cast<int64_t>(expert_topk_),          // top_k
                                                   32,                                          // group_size
                                                   stream);

    const size_t elements_num = total_tokens * expert_topk_ * expert_inter_size_ * 2;
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

    // W4A8: Quantize activations to FP8 before GEMM2
    const void* gemm2_a = intermediate_cache2_;
    const void* gemm2_a_scale = nullptr;
    const void* gemm2_global_scale = nullptr;

    if (humming_compute_dtype_ == llm_kernels::nvidia::humming::ComputeDtype::FP8) {
      // Zero-initialize scale buffer to avoid NaN from uninitialized memory
      cudaMemsetAsync(fp8_scale_buffer_, 0, fp8_scale_buffer_size_, stream);
      llm_kernels::nvidia::InvokeDynamicPerTokenScaledFP8Quant<T>(
          reinterpret_cast<const T*>(intermediate_cache2_), fp8_scale_buffer_,
          nullptr,  // scale_ub (no upper bound)
          fp8_activation_buffer_, static_cast<int>(size_k_w2), static_cast<int>(total_tokens * expert_topk_), stream);
      gemm2_a = fp8_activation_buffer_;
      gemm2_a_scale = fp8_scale_buffer_;
      gemm2_global_scale = w2_global_scale_;
    }

    // Zero locks buffer before kernel launch to prevent deadlock from stale lock values
    cudaMemsetAsync(humming_locks_, 0, humming_locks_size_, stream);

    // NOTE: For GEMM2, top_k must be 1 (not expert_topk_). The activation tensor
    // intermediate_cache2 is already expanded to [tokens*topk, inter_size] — each row is
    // one token-expert pair. With top_k=1, the kernel uses sorted_ids[i] directly as the
    // row index into intermediate_cache2 (no division by top_k). Using expert_topk_ here
    // would cause the kernel to divide sorted_ids by topk, reading only 1/topk of the rows.
    llm_kernels::nvidia::humming::humming_moe_gemm(
        gemm2_a,                  // a
        w2,                       // b (FP4 WGMMA layout)
        intermediate_cache3_,     // c
        gemm2_a_scale,            // a_scale (nullptr for W4A16)
        w2_scales,                // b_scale
        gemm2_global_scale,       // global_scale
        nullptr,                  // b_zero_point
        nullptr,                  // bias
        sorted_ids_ptr,           // sorted_ids
        expert_ids_ptr,           // expert_ids
        num_tokens_post_pad_ptr,  // num_tokens_padded
        nullptr,                  // expert_layout (INDEXED uses sorted_ids)
        nullptr,                  // topk_weights (not used by humming kernel)
        humming_locks_,           // locks
        humming_workspace_,       // workspace
        llm_kernels::nvidia::humming::GemmType::INDEXED, humming_compute_dtype_,
        static_cast<int64_t>(total_tokens * expert_topk_),  // size_m (already expanded)
        size_n_w2,                                          // size_n
        size_k_w2,                                          // size_k
        static_cast<int64_t>(expert_num_per_node_),         // num_experts
        1,                                                  // top_k = 1 (data already expanded)
        32,                                                 // group_size
        stream,
        moe_block_size_i32);  // required_block_m: use same as GEMM1

    // ----- Fused MulTopkWeight + Sum: intermediate_cache3 → output [tokens, hidden_size] -----
    // Single kernel fusing topk_weight multiplication and expert-sum reduction.
    // Eliminates intermediate write-back + re-read of cache3 (~48KB/token for DSv4).
    // intermediate_cache3: [total_tokens * topk, hidden_size] (read-only, not modified in-place)
    // topk_weights: [total_tokens * topk] (flat float, one weight per token-expert row)
    // output: [total_tokens, hidden_size]
    if (use_expert_parallel) {
      llm_kernels::nvidia::InvokeMoeFusedMulSum<T, true>(
          intermediate_cache3_, reinterpret_cast<const float*>(curr_topk_weights), curr_topk_ids, output_ptr,
          static_cast<int>(total_tokens), static_cast<int>(expert_topk_), static_cast<int>(hidden_size), stream);
    } else {
      llm_kernels::nvidia::InvokeMoeFusedMulSum<T, false>(
          intermediate_cache3_, reinterpret_cast<const float*>(curr_topk_weights), curr_topk_ids, output_ptr,
          static_cast<int>(total_tokens), static_cast<int>(expert_topk_), static_cast<int>(hidden_size), stream);
    }
  };

  if (using_deepep_) {
    // LCOV_EXCL_START
    const size_t dispatch_num_tokens = output_tensors[1].shape[0];
    if (dispatch_num_tokens > 0) {
      expert_map_->InvokeExpertMapInverseInplace(static_cast<int32_t*>(topk_ids_ptr_),
                                                 dispatch_num_tokens * expert_topk_,
                                                 context_->device->GetPrimaryComputeStream().Get());
      ProcessTokens(output_tensors[1].GetPtr<void>(), output_tensors[1].GetPtr<void>(), dispatch_num_tokens);
    }
    Combine({output_tensors[1]}, output_tensors);
    // LCOV_EXCL_STOP
  } else if (global_expert_para_size_ > 1) {
    // EP without deepep
    expert_map_->InvokeExpertMapInplace(static_cast<int32_t*>(topk_ids_ptr_), num_tokens * expert_topk_,
                                        context_->device->GetPrimaryComputeStream().Get());
    ProcessTokens(input_tensors[0].GetPtr<void>(), output_tensors[0].GetPtr<void>(), num_tokens);
  } else {
    // Single-node, no EP
    ProcessTokens(input_tensors[0].GetPtr<void>(), output_tensors[0].GetPtr<void>(), num_tokens);
  }

  output_tensors[0].shape = input_tensors[0].shape;
  output_tensors[0].dtype = input_tensors[0].dtype;
  return Status();
}

template <typename T>
Status HummingFp4MoeLayer::ExecuteTopkSoftplusSqrt(const std::vector<Tensor>& input_tensors,
                                                   std::vector<Tensor>& output_tensors) {
  std::vector<Tensor> softplus_inputs;
  std::vector<Tensor> softplus_outputs;

  const size_t num_tokens = input_tensors[0].shape[0];

  // Router logits
  softplus_inputs.push_back(input_tensors[1]);

  if (use_hash_) {
    KLLM_CHECK_WITH_INFO(input_tensors.size() >= 7,
                         "HummingFp4MoeLayer: hash path requires input_ids (slot 6) and tid2eid (slot 5)");
    softplus_inputs.push_back(input_tensors[6]);  // input_ids
    softplus_inputs.push_back(input_tensors[5]);  // tid2eid
  } else if (use_e_score_correction_bias_) {
    KLLM_CHECK_WITH_INFO(input_tensors.size() >= 6, "HummingFp4MoeLayer: correction-bias path requires bias at slot 5");
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
    KLLM_LOG_ERROR << "HummingFp4MoeLayer::ExecuteTopkSoftplusSqrt ERROR: forward failed";
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

Status HummingFp4MoeLayer::Dispatch(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  return expert_parallel_manager_->Dispatch(input_tensors, output_tensors, context_->device->GetPrimaryComputeStream());
}

Status HummingFp4MoeLayer::Combine(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  return expert_parallel_manager_->Combine(input_tensors, output_tensors, context_->device->GetPrimaryComputeStream());
}

Status HummingFp4MoeLayer::DumpEplbData(Tensor& topk_ids) {
  topk_ids.SaveToNpyFile(
      fmt::format("{}/layer_{}/topk_ids_{}_{}.npy", eplb_dump_path_, layer_idx_, eplb_dump_step_, rank_));
  eplb_dump_step_ += 1;
  return Status();
}

}  // namespace ksana_llm
