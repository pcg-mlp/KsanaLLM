/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "layers/humming_fp4_moe_layer.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <set>

#include "csrc/kernels/nvidia/moe_utils/moe_utils.h"
#include "csrc/kernels/nvidia/others/humming/jit/humming_jit_compiler.h"
#include "csrc/kernels/nvidia/others/humming/launcher/launcher.h"
#include "csrc/kernels/nvidia/others/humming/ops/ops.h"
#include "csrc/kernels/nvidia/others/humming/ops/wrapper.h"
#include "csrc/kernels/nvidia/others/humming/preprocess/humming_preprocess.h"
#include "csrc/utils/nvidia/cuda_utils.h"
#include "kernels/nvidia/triton_wrapper.h"
#include "layers/topk_softplus_sqrt_layer.h"
#include "utils/memory_utils.h"
#include "utils/singleton.h"
#include "utils/utils.h"

namespace ksana_llm {

// Static member definitions
std::shared_ptr<llm_kernels::nvidia::humming::HummingJitCompiler> HummingFp4MoeLayer::jit_compiler_;
std::mutex HummingFp4MoeLayer::jit_mutex_;

const std::vector<llm_kernels::nvidia::humming::KernelMeta>* HummingFp4MoeLayer::GetJitKernelTable() {
  if (!jit_compiler_) return nullptr;
  return &jit_compiler_->GetKernelTable();
}

// Maximum moe_block_size for workspace pre-allocation (actual value is determined at runtime).
// Must be >= the largest possible BLOCK_SHAPE_M from the humming heuristic (currently 64).
static constexpr int64_t kHummingMoeMaxBlockSize = 64;

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
  InitTopkSoftplusSqrt(runtime_config, rank);

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

  // Read profiling iteration count once here (default 15).  When 0, profiling is
  // disabled and only heuristic kernels are compiled (no extend/variant configs).
  // Preprocess() reuses profile_iters_/warmup_iters_ instead of re-reading the env.
  profile_iters_ = static_cast<size_t>(GetEnvAsPositiveInt("QUANT_PROFILE", 15));
  warmup_iters_ = std::max(1UL, profile_iters_ / 3);
  const bool compile_extend = (profile_iters_ != 0);

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

  jit_compiler_->CompileKernels(n1, k1, static_cast<int64_t>(expert_num_per_node_), humming_compute_dtype_, gemm_type,
                                compile_extend);
  jit_compiler_->CompileKernels(n2, k2, static_cast<int64_t>(expert_num_per_node_), humming_compute_dtype_, gemm_type,
                                compile_extend);

  KLLM_LOG_INFO << fmt::format("Rank[{}] HummingFp4MoeLayer Init", rank_);
  return Status();
}

void HummingFp4MoeLayer::InitTopkSoftplusSqrt(const RuntimeConfig& runtime_config, int rank) {
  TopKSoftplusSqrtLayerParameters softplus_sqrt_params{
      .topk = static_cast<uint32_t>(expert_topk_),
      .num_experts = static_cast<uint32_t>(expert_num_per_node_ * global_expert_para_size_),
      .max_token_num = static_cast<uint32_t>(max_token_num_),
      .renormalize = norm_topk_prob_,
      .routed_scaling_factor = routed_scaling_factor_,
      .use_hash = use_hash_,
      .use_correction_bias = !use_hash_ && use_e_score_correction_bias_,
      .topk_ids_dtype = TYPE_INT32,
  };

  topk_softplus_sqrt_layer_ = std::make_shared<TopKSoftplusSqrtLayer>();
  topk_softplus_sqrt_layer_->Init({softplus_sqrt_params}, runtime_config, context_, rank);
  softplus_sqrt_workspace_size_ = topk_softplus_sqrt_layer_->GetWorkspaceSize();
}

void HummingFp4MoeLayer::SetExpertParallelManager(std::shared_ptr<ExpertParallelManager> expert_parallel_manager) {
  expert_parallel_manager_ = expert_parallel_manager;
}

size_t HummingFp4MoeLayer::GetWorkspaceSize() { DISPATCH_BY_3_DTYPE(inter_data_type_, GetWorkspaceSizeT); }

template <typename T>
size_t HummingFp4MoeLayer::GetWorkspaceSizeT() {
  size_t m = max_token_num_;

  ws_.gating.topk_weights_size = RoundUp(max_token_num_ * expert_topk_ * sizeof(float), 256);
  ws_.gating.topk_ids_size = RoundUp(max_token_num_ * expert_topk_ * sizeof(int32_t), 256);

  // fused_id_buffer: fill_info_vec(4) + sorted_ids + expert_ids + num_tokens_post_pad + cumsum
  size_t numel = m * expert_topk_;
  size_t max_num_tokens_padded = numel + expert_num_per_node_ * (kHummingMoeMaxBlockSize - 1);
  size_t max_num_m_blocks = (max_num_tokens_padded + kHummingMoeMaxBlockSize - 1) / kHummingMoeMaxBlockSize;
  // fill_info(4) + sorted_ids(max_num_tokens_padded) + expert_ids(max_num_m_blocks) + num_tokens_post_pad(1) +
  // cumsum(expert_num_per_node+1)
  ws_.align_gemm1.total_size =
      RoundUp((SGL_MOE_ALIGN_BLOCK_VEC_SIZE + max_num_tokens_padded + max_num_m_blocks + 1 + expert_num_per_node_ + 1) *
                  sizeof(int32_t),
              256);

  // GEMM2 fused_id_buffer: same sizing as GEMM1 (worst case block_m=64)
  ws_.align_gemm2.total_size = ws_.align_gemm1.total_size;

  // Intermediate caches
  // cache1: [m * topk, inter_size * 2] — output of w1 GEMM (gate+up)
  ws_.cache.gemm1_size = RoundUp(m * expert_topk_ * expert_inter_size_ * 2 * sizeof(T), 256);
  // cache2: [m * topk, inter_size] — output of SiluAndMul
  ws_.cache.silu_size = RoundUp(m * expert_topk_ * expert_inter_size_ * sizeof(T), 256);
  // cache3: [m * topk, hidden_size] — output of w2 GEMM
  // NOTE: cache3 reuses cache1's memory (P5 optimization) since their lifetimes don't overlap:
  //   cache1 is consumed by SiluAndMul before cache3 is produced by GEMM2.
  ws_.cache.gemm2_size = RoundUp(m * expert_topk_ * expert_hidden_size_ * sizeof(T), 256);

  // Humming workspace: sized for the larger of the two GEMMs
  int64_t max_n = std::max(expert_inter_size_ * 2, expert_hidden_size_);
  int64_t max_k = std::max(expert_hidden_size_, expert_inter_size_);
  ws_.kernel.locks_size = RoundUp(llm_kernels::nvidia::humming::get_locks_size(max_token_num_ * expert_topk_, max_n,
                                                                               max_k, humming_compute_dtype_) *
                                      sizeof(int32_t),
                                  256);

  // Compute workspace size based on actual max num_ctas_per_sm from compiled kernels.
  // The humming heuristic may produce kernels with num_ctas_per_sm > 3 (e.g. 4 for
  // small-K, large-N MoE GEMMs), which the hardcoded get_workspace_size() doesn't account for.
  // Locks buffer has 4x over-provision so it's safe, but the TMA descriptor buffer is exact-sized.
  int max_ctas_per_sm = (humming_compute_dtype_ == llm_kernels::nvidia::humming::ComputeDtype::FP8) ? 3 : 2;
  if (jit_compiler_) {
    for (const auto& meta : jit_compiler_->GetKernelTable()) {
      if (meta.compute_dtype != humming_compute_dtype_) continue;
      if (meta.gemm_type != llm_kernels::nvidia::humming::GemmType::INDEXED) continue;
      max_ctas_per_sm = std::max(max_ctas_per_sm, meta.num_ctas_per_sm);
    }
  }
  int32_t sm_count = llm_kernels::utils::GetSMCount();
  int64_t total_ctas = static_cast<int64_t>(max_ctas_per_sm) * sm_count;
  ws_.kernel.tma_size = RoundUp(total_ctas * 128, 256);

  // W4A8 FP8 activation buffers
  ws_.fp8.activation_size = 0;
  ws_.fp8.scale_size = 0;
  if (humming_compute_dtype_ == llm_kernels::nvidia::humming::ComputeDtype::FP8) {
    // FP8 buffer: max of GEMM1 input [M, hidden_size] and GEMM2 input [M*topk, inter_size]
    ws_.fp8.activation_size = RoundUp(std::max(static_cast<int64_t>(m * expert_hidden_size_),
                                               static_cast<int64_t>(m * expert_topk_ * expert_inter_size_)) *
                                          sizeof(uint8_t),
                                      256);
    // Per-token scale buffer: max tokens = M * topk (GEMM2 has more tokens)
    ws_.fp8.scale_size = RoundUp(static_cast<int64_t>(m * expert_topk_) * sizeof(float), 256);
  }

  ws_.softplus_sqrt_size = RoundUp(softplus_sqrt_workspace_size_, 256);

  // Total workspace: cache1 is shared with cache3, so we use max(cache1, cache3) instead of sum
  size_t cache13_size = std::max(ws_.cache.gemm1_size, ws_.cache.gemm2_size);
  ws_.total_size = ws_.gating.topk_weights_size + ws_.gating.topk_ids_size + ws_.align_gemm1.total_size +
                   ws_.align_gemm2.total_size + cache13_size + ws_.cache.silu_size + ws_.kernel.locks_size +
                   ws_.kernel.tma_size + ws_.softplus_sqrt_size + ws_.fp8.activation_size + ws_.fp8.scale_size;

  KLLM_LOG_DEBUG << fmt::format("Rank[{}] Request {} for HummingFp4MoeLayer", rank_, ws_.total_size);
  return ws_.total_size;
}

Status HummingFp4MoeLayer::Preprocess(const ModelConfig& model_config, const RuntimeConfig& runtime_config) {
  // Reuse profile_iters_/warmup_iters_ read in Init() to avoid inconsistent reads of QUANT_PROFILE.
  if (profile_iters_ == 0) {
    KLLM_LOG_DEBUG << "QUANT_PROFILE==0, skipping humming profiling";
    return Status();
  }

  static std::mutex g_mtx;  // process-level serialization, mirrors cutlass
  std::lock_guard<std::mutex> guard(g_mtx);

  int dt = static_cast<int>(humming_compute_dtype_);
  int gt = static_cast<int>(llm_kernels::nvidia::humming::GemmType::INDEXED);
  size_t E = expert_num_per_node_;

  // GEMM1: N=inter*2, K=hidden, top_k=expert_topk, size_m=tokens
  int64_t n1 = static_cast<int64_t>(expert_inter_size_ * 2);
  int64_t k1 = static_cast<int64_t>(expert_hidden_size_);
  if (!Singleton<HummingMoeSearchStatus>::GetInstance()->IsProfiled(n1, k1, E, dt, gt)) {
    auto schedule = ProfileAllBuckets(n1, k1, E, expert_topk_, warmup_iters_, profile_iters_, /*size_m_multiplier=*/1);
    Singleton<HummingMoeSearchStatus>::GetInstance()->AddProfiledSchedule(n1, k1, E, dt, gt, schedule);
    KLLM_LOG_INFO << fmt::format("Rank[{}] HummingFp4MoeLayer GEMM1 profiled (N={},K={})", rank_, n1, k1);
  } else {
    KLLM_LOG_INFO << fmt::format("Rank[{}] Reusing profiled GEMM1 (N={},K={})", rank_, n1, k1);
  }

  // GEMM2: N=hidden, K=inter, top_k=1, size_m=tokens*topk
  int64_t n2 = static_cast<int64_t>(expert_hidden_size_);
  int64_t k2 = static_cast<int64_t>(expert_inter_size_);
  if (!Singleton<HummingMoeSearchStatus>::GetInstance()->IsProfiled(n2, k2, E, dt, gt)) {
    auto schedule =
        ProfileAllBuckets(n2, k2, E, /*top_k=*/1, warmup_iters_, profile_iters_, /*size_m_multiplier=*/expert_topk_);
    Singleton<HummingMoeSearchStatus>::GetInstance()->AddProfiledSchedule(n2, k2, E, dt, gt, schedule);
    KLLM_LOG_INFO << fmt::format("Rank[{}] HummingFp4MoeLayer GEMM2 profiled (N={},K={})", rank_, n2, k2);
  } else {
    KLLM_LOG_INFO << fmt::format("Rank[{}] Reusing profiled GEMM2 (N={},K={})", rank_, n2, k2);
  }
  return Status();
}

HummingFp4MoeLayer::ProfAlignBuffers::ProfAlignBuffers(int64_t max_num_tokens_padded, int64_t max_num_m_blocks,
                                                       int64_t num_experts) {
  auto opt_i32 = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32);
  sorted_ids = torch::empty({max_num_tokens_padded}, opt_i32);
  expert_ids = torch::empty({max_num_m_blocks}, opt_i32);
  num_tokens_post_pad = torch::empty({1}, opt_i32);
  cumsum = torch::empty({num_experts + 1}, opt_i32);
}

torch::Tensor HummingFp4MoeLayer::MakeDummyTopkIds(int64_t num_tokens, int64_t top_k, int64_t num_experts) {
  // Random uniform sampling with fixed seed — matches the benchmark's expert distribution
  // (benchmark uses torch::randn for gate values, then topk → random uniform expert assignment).
  // The previous round-robin distribution (i+j)%E was perfectly uniform, which is unrealistic:
  // at small M most experts get 0 tokens in practice, and at large M the distribution converges
  // to uniform only statistically. Random sampling naturally produces this behavior without
  // parameter tuning. Fixed seed ensures reproducibility across runs.
  std::mt19937 rng(42);
  std::uniform_int_distribution<int32_t> dist(0, static_cast<int32_t>(num_experts - 1));
  std::vector<int32_t> host_ids(num_tokens * top_k);
  for (int64_t i = 0; i < num_tokens * top_k; ++i) {
    host_ids[i] = dist(rng);
  }
  auto cpu_opts = torch::TensorOptions().dtype(torch::kInt32);
  auto gpu_opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32);
  return torch::from_blob(host_ids.data(), {num_tokens * top_k}, cpu_opts).to(gpu_opts);
}

HummingFp4MoeLayer::ProfWeights HummingFp4MoeLayer::PrepareDummyWeights(int64_t size_n, int64_t size_k,
                                                                        int64_t num_experts, cudaStream_t stream) {
  using namespace llm_kernels::nvidia::humming;

  const bool is_fp8 = (humming_compute_dtype_ == ComputeDtype::FP8);
  auto opt_u8 = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kUInt8);
  auto opt_f32 = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);

  // Generate raw FP4 weights: random [0, 256)
  torch::Tensor b_raw = torch::empty({num_experts, size_n, size_k / 2}, opt_u8).random_(0, 256);
  // Generate raw scales: random [119, 123) — safe E8M0 range (matches test)
  torch::Tensor b_scale_raw = torch::empty({num_experts, size_n, size_k / 32}, opt_u8).random_(119, 123);

  ProfWeights out;
  out.b = torch::empty({num_experts, size_n, size_k / 2}, opt_u8);
  out.b_scale = torch::empty({num_experts, size_k / 32, size_n}, opt_u8);

  if (is_fp8) {
    // W4A8: process_mxfp4_w4a8 modifies weight+scale in-place, outputs global_scale
    out.global_scale = torch::empty({num_experts}, opt_f32);
    torch::Tensor b_copy = b_raw.clone();
    torch::Tensor s_copy = b_scale_raw.clone();
    process_mxfp4_w4a8_weight(stream, b_copy.data_ptr(), s_copy.data_ptr(), out.global_scale.data_ptr<float>(),
                              num_experts, size_n, size_k, 32);
    repack_weight_wgmma(stream, b_copy.data_ptr(), out.b.data_ptr(), num_experts, size_n, size_k,
                        humming_compute_dtype_);
    process_scales(stream, s_copy.data_ptr(), out.b_scale.data_ptr(), num_experts, size_n, size_k, 32,
                   humming_compute_dtype_);
  } else {
    // W4A16: repack + process_scales directly
    repack_weight_wgmma(stream, b_raw.data_ptr(), out.b.data_ptr(), num_experts, size_n, size_k,
                        humming_compute_dtype_);
    process_scales(stream, b_scale_raw.data_ptr(), out.b_scale.data_ptr(), num_experts, size_n, size_k, 32,
                   humming_compute_dtype_);
  }
  cudaStreamSynchronize(stream);
  return out;
}

// --- Deduplicated helpers ---

HummingFp4MoeLayer::GemmMetaSelection HummingFp4MoeLayer::SelectGemmMeta(int64_t size_n, int64_t size_k,
                                                                         int64_t size_m) const {
  GemmMetaSelection sel;
  sel.meta = Singleton<HummingMoeSearchStatus>::GetInstance()->GetProfiledMeta(
      size_n, size_k, expert_num_per_node_, static_cast<int>(humming_compute_dtype_),
      static_cast<int>(llm_kernels::nvidia::humming::GemmType::INDEXED), size_m);

  if (sel.meta != nullptr) {
    sel.block_m = sel.meta->block_m;
    sel.required_block_m = 0;
  } else {
    const llm_kernels::nvidia::humming::KernelMeta* heuristic_meta = nullptr;
    sel.block_m = llm_kernels::nvidia::humming::get_kernel_block_m(size_m, size_n, size_k,
                                                                   llm_kernels::nvidia::humming::GemmType::INDEXED,
                                                                   humming_compute_dtype_, &heuristic_meta);
    sel.meta = heuristic_meta;
    // When SelectKernel returned no meta (all kernels range-based, size_m outside ranges),
    // fall back to required_block_m search in humming_moe_gemm (block_m-based loop lookup).
    sel.required_block_m = (heuristic_meta != nullptr) ? 0 : sel.block_m;
  }
  return sel;
}

void HummingFp4MoeLayer::InvokeAlignBlockSize(int32_t* topk_ids, int32_t* sorted_ids, int32_t* expert_ids,
                                              int32_t* num_tokens_post_pad, int32_t* cumsum,
                                              int32_t max_num_tokens_padded, size_t numel, int32_t moe_block_size,
                                              int32_t top_k, bool use_expert_parallel, int32_t expert_para_size,
                                              int rank, cudaStream_t stream) {
  if (expert_num_per_node_ >= 224) {
    llm_kernels::nvidia::InvokeSglMoeAlignBlockSize<int32_t>(
        topk_ids, sorted_ids, expert_ids, num_tokens_post_pad, max_num_tokens_padded,
        static_cast<int32_t>(expert_num_per_node_), moe_block_size, numel, cumsum, stream);
  } else {
    if (use_expert_parallel) {
      llm_kernels::nvidia::InvokeMoeAlignBlockSize<int32_t, uint16_t, true>(
          topk_ids, sorted_ids, expert_ids, num_tokens_post_pad, top_k, static_cast<int32_t>(expert_num_per_node_),
          expert_para_size, moe_block_size, numel, rank, stream);
    } else {
      llm_kernels::nvidia::InvokeMoeAlignBlockSize<int32_t, uint16_t, false>(
          topk_ids, sorted_ids, expert_ids, num_tokens_post_pad, top_k, static_cast<int32_t>(expert_num_per_node_),
          expert_para_size, moe_block_size, numel, rank, stream);
    }
  }
}

void HummingFp4MoeLayer::InvokeFillInfo(int32_t* fused_id_buffer, void* fill_info_device, int id_offset,
                                        int max_num_tokens_padded, int numel, cudaStream_t stream) {
  if (expert_num_per_node_ >= 224) return;  // SglMoe path doesn't need fill_info
  int fill_info[3] = {id_offset, max_num_tokens_padded, numel};
  llm_kernels::nvidia::InvokeFillIntToBuffer(fused_id_buffer, fill_info_device, fill_info, 3, stream);
}

template <typename T>
bool HummingFp4MoeLayer::QuantizeToFp8(const void* input, int size_k, int num_tokens, cudaStream_t stream,
                                       const void*& out_a, const void*& out_a_scale) {
  if (humming_compute_dtype_ != llm_kernels::nvidia::humming::ComputeDtype::FP8) return false;
  Singleton<TritonWrapper>::GetInstance()->InvokeHummingFp8Quant<T>(reinterpret_cast<const T*>(input), ws_.fp8.scales,
                                                                    ws_.fp8.activation, size_k, num_tokens, stream);
  out_a = ws_.fp8.activation;
  out_a_scale = ws_.fp8.scales;
  return true;
}

const llm_kernels::nvidia::humming::KernelMeta* HummingFp4MoeLayer::ProfileSingleGemm(
    int64_t size_m, int64_t size_n, int64_t size_k, int64_t num_experts, int64_t top_k, size_t warmup,
    size_t profile_iters, cudaStream_t stream, const ProfWeights& weights) {
  using namespace llm_kernels::nvidia::humming;

  // 1. Filter candidates by (N, K, gemm_type, dtype)
  // Also filter by heuristic M range: skip candidates whose max_shape_m < size_m.
  // This prevents small-M kernels (e.g. block_m=8 with range [1,384]) from being
  // selected at large M where they're suboptimal, even if profiling noise makes
  // them appear faster.
  std::vector<const KernelMeta*> candidates;
  for (const auto& meta : jit_compiler_->GetKernelTable()) {
    if (meta.size_n != size_n || meta.size_k != size_k) continue;
    if (meta.gemm_type != GemmType::INDEXED) continue;
    if (meta.compute_dtype != humming_compute_dtype_) continue;
    if (meta.max_shape_m < size_m) continue;
    candidates.push_back(&meta);
  }
  if (candidates.empty()) return nullptr;

  if (profile_iters <= 0 || num_experts <= 0 || size_m <= 0 || size_n <= 0 || size_k <= 0) {
    return nullptr;
  }

  // 2. Collect distinct block_m values among candidates
  std::set<int> distinct_block_ms;
  for (const auto* m : candidates) distinct_block_ms.insert(m->block_m);

  const bool is_fp8 = (humming_compute_dtype_ == ComputeDtype::FP8);

  // Set torch stream to our compute stream so torch ops run on the right stream
  auto original_stream = at::cuda::getCurrentCUDAStream();
  at::cuda::setCurrentCUDAStream(at::cuda::getStreamFromExternal(stream, original_stream.device_index()));

  auto opt_u8 = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kUInt8);
  auto opt_f32 = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
  auto opt_i32 = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32);

  // Weights are pre-prepared by caller (PrepareDummyWeights), reused across buckets.
  void* b_ptr = weights.b.data_ptr();
  void* b_scale_ptr = weights.b_scale.data_ptr();
  void* global_scale_ptr = is_fp8 ? weights.global_scale.data_ptr() : nullptr;

  // 3. Allocate dummy activation + output buffers
  // FP8: 1 byte/element, BF16: 2 bytes/element.
  int64_t a_bytes = size_m * size_k * (is_fp8 ? 1 : 2);
  torch::Tensor a_buf = torch::zeros({a_bytes}, opt_u8);
  // Output C: must hold num_tokens_post_pad rows (after MoE alignment padding).
  // Output is always BF16 (2 bytes/element).
  int64_t numel = size_m * top_k;
  int max_block_m_for_c = 1;
  for (const auto* m : candidates) {
    max_block_m_for_c = std::max(max_block_m_for_c, m->block_m);
  }
  int64_t max_num_tokens_padded = numel + num_experts * (max_block_m_for_c - 1);
  torch::Tensor c_buf = torch::empty({max_num_tokens_padded, size_n * 2}, opt_u8);

  // Activation scale for W4A8: 1.0 (valid FP8 per-token scale)
  torch::Tensor a_scale_buf;
  if (is_fp8) {
    a_scale_buf = torch::full({size_m}, 1.0f, opt_f32);
  }

  // locks and workspace — sized for max num_ctas_per_sm among candidates
  int max_ctas_per_sm = is_fp8 ? 3 : 2;
  for (const auto* m : candidates) {
    max_ctas_per_sm = std::max(max_ctas_per_sm, m->num_ctas_per_sm);
  }
  int32_t sm_count = llm_kernels::utils::GetSMCount();
  int64_t locks_count = get_locks_size(size_m, size_n, size_k, humming_compute_dtype_);
  int64_t total_ctas = static_cast<int64_t>(max_ctas_per_sm) * sm_count;
  int64_t workspace_size = RoundUp(total_ctas * 128, 256);
  torch::Tensor locks_buf = torch::zeros({locks_count}, opt_i32);
  torch::Tensor workspace_buf = torch::empty({workspace_size}, opt_u8);

  // 4. Prepare align buffers per distinct block_m
  torch::Tensor topk_ids_buf = MakeDummyTopkIds(size_m, top_k, num_experts);

  std::map<int, std::unique_ptr<ProfAlignBuffers>> align_map;
  for (int block_m : distinct_block_ms) {
    int64_t padded = numel + num_experts * (block_m - 1);
    int64_t max_m_blocks = (padded + block_m - 1) / block_m;
    auto ab = std::make_unique<ProfAlignBuffers>(padded, max_m_blocks, num_experts);

    InvokeAlignBlockSize(topk_ids_buf.data_ptr<int32_t>(), ab->sorted_ids.data_ptr<int32_t>(),
                         ab->expert_ids.data_ptr<int32_t>(), ab->num_tokens_post_pad.data_ptr<int32_t>(),
                         ab->cumsum.data_ptr<int32_t>(), static_cast<int32_t>(padded), static_cast<size_t>(numel),
                         static_cast<int32_t>(block_m), static_cast<int32_t>(top_k),
                         /*use_expert_parallel=*/false, /*expert_para_size=*/1, /*rank=*/0, stream);
    align_map[block_m] = std::move(ab);
  }
  cudaStreamSynchronize(stream);

  // 5. Profile each candidate
  const KernelMeta* best_meta = nullptr;
  float best_latency = std::numeric_limits<float>::max();

  void* a_ptr = a_buf.data_ptr();
  void* c_ptr = c_buf.data_ptr();
  void* a_scale_ptr = is_fp8 ? a_scale_buf.data_ptr() : nullptr;
  int32_t* locks_ptr = locks_buf.data_ptr<int32_t>();
  void* workspace_ptr = workspace_buf.data_ptr();

  for (const auto* meta : candidates) {
    auto& ab = align_map.at(meta->block_m);

    // Skip if actual tiles < num_ctas (stream-K deadlock)
    int64_t num_ctas = static_cast<int64_t>(meta->num_ctas_per_sm) * sm_count;
    int32_t host_ntpp = ab->num_tokens_post_pad.item<int32_t>();
    int64_t num_m_blocks = (host_ntpp + meta->block_m - 1) / meta->block_m;
    int64_t num_n_blocks = size_n / meta->block_n;
    int64_t actual_tiles = num_m_blocks * num_n_blocks;
    if (actual_tiles < num_ctas) continue;

    int32_t* sorted_ids_ptr = ab->sorted_ids.data_ptr<int32_t>();
    int32_t* expert_ids_ptr = ab->expert_ids.data_ptr<int32_t>();
    int32_t* ntpp_ptr = ab->num_tokens_post_pad.data_ptr<int32_t>();

    try {
      auto launch_fn = [&]() {
        // Zero locks before each launch (prevent stale lock deadlock)
        cudaMemsetAsync(locks_ptr, 0, locks_count * sizeof(int32_t), stream);
        launch_kernel(meta, a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, is_fp8 ? global_scale_ptr : nullptr,
                      /*b_zero_point=*/nullptr, /*bias=*/nullptr, sorted_ids_ptr, expert_ids_ptr, ntpp_ptr,
                      /*expert_layout=*/nullptr, locks_ptr, workspace_ptr, size_m, size_n, size_k, num_experts, top_k,
                      /*use_int64_expert_layout=*/false, stream);
      };

      float latency_ms = llm_kernels::utils::MeasureCudaExecutionTime(launch_fn, stream, static_cast<int>(warmup),
                                                                      static_cast<int>(profile_iters));

      if (latency_ms < best_latency) {
        best_latency = latency_ms;
        best_meta = meta;
      }
    } catch (const std::exception& e) {
      // Launch failed (smem exceeded, illegal memory access, etc.) — skip this candidate.
      // Must clear the sticky CUDA error so subsequent candidate tests are not affected.
      cudaDeviceSynchronize();
      cudaGetLastError();
    }
  }

  // Restore original torch stream
  at::cuda::setCurrentCUDAStream(original_stream);

  return best_meta;
}

HummingMoeSearchStatus::ProfiledSchedule HummingFp4MoeLayer::ProfileAllBuckets(int64_t n, int64_t k, int64_t e,
                                                                               int64_t top_k, size_t warmup,
                                                                               size_t profile_iters,
                                                                               int64_t size_m_multiplier) {
  HummingMoeSearchStatus::ProfiledSchedule result;
  cudaStream_t stream = context_->device->GetPrimaryComputeStream().Get();

  // Search points: mixed power-of-2 + intermediate values (× size_m_multiplier for GEMM2).
  // Intermediate values (3, 6, 12, 24, 48, 96) cover common decode batch sizes;
  // midpoints (192, 384, 768, 1536, 3072, 6144) halve the right-align gap.
  static constexpr std::array<int64_t, 26> kBuckets = {1,   2,    3,    4,    6,    8,    12,   16,  24,
                                                       32,  48,   64,   96,   128,  192,  256,  384, 512,
                                                       768, 1024, 1536, 2048, 3072, 4096, 6144, 8192};
  int num_buckets = static_cast<int>(kBuckets.size());

  KLLM_LOG_INFO << fmt::format("Rank[{}] HummingFp4MoeLayer profiling {} buckets for N={} K={}", rank_, num_buckets, n,
                               k);

  // Prepare dummy weights once — repack/process is expensive and only depends on (N, K, E, dtype).
  ProfWeights weights = PrepareDummyWeights(n, k, e, stream);

  for (int64_t bucket : kBuckets) {
    int64_t rep_m = bucket * size_m_multiplier;

    const llm_kernels::nvidia::humming::KernelMeta* best =
        ProfileSingleGemm(rep_m, n, k, e, top_k, warmup, profile_iters, stream, weights);
    if (best != nullptr) {
      result.size_ms.push_back(rep_m);
      result.metas.push_back(best);
      result.block_ms.push_back(best->block_m);
    }
    KLLM_LOG_INFO << fmt::format("Rank[{}] HummingFp4MoeLayer rep_m={} done best={}", rank_, rep_m,
                                 static_cast<const void*>(best));
  }
  return result;
}

template <typename T>
Status HummingFp4MoeLayer::ForwardT(const Tensor* hidden_states, const Tensor* gating_output, const Tensor* w1,
                                    const Tensor* w2, const Tensor* eplb_map, Tensor* output, Tensor* workspace,
                                    const Tensor* e_score_bias, const Tensor* input_ids) {
  // Lazy creation of profiling CUDA events (created once, reused across forwards)
  bool do_profile = (perf_stats_ != nullptr);
  if (do_profile) {
    EnsureProfilingEvents();
  }

  // Assign workspace pointers on first call
  if (set_workspace_buffer_info_) {
    set_workspace_buffer_info_ = false;
    uint8_t* p = reinterpret_cast<uint8_t*>(workspace_buffer_->GetPtr<void>());

    ws_.gating.topk_weights = p;
    p += ws_.gating.topk_weights_size;
    ws_.gating.topk_ids = p;
    p += ws_.gating.topk_ids_size;
    ws_.align_gemm1.base = p;
    p += ws_.align_gemm1.total_size;
    ws_.align_gemm2.base = p;
    p += ws_.align_gemm2.total_size;
    ws_.cache.gemm1_output = p;
    ws_.cache.gemm2_output = p;  // cache3 shares cache1 memory (non-overlapping lifetimes)
    p += std::max(ws_.cache.gemm1_size, ws_.cache.gemm2_size);
    ws_.cache.silu_output = p;
    p += ws_.cache.silu_size;
    ws_.kernel.locks = reinterpret_cast<int32_t*>(p);
    p += ws_.kernel.locks_size;
    ws_.kernel.tma_descriptors = p;
    p += ws_.kernel.tma_size;
    // W4A8 FP8 buffers (after kernel workspace)
    if (humming_compute_dtype_ == llm_kernels::nvidia::humming::ComputeDtype::FP8) {
      ws_.fp8.activation = p;
      p += ws_.fp8.activation_size;
      ws_.fp8.scales = reinterpret_cast<float*>(p);
      p += ws_.fp8.scale_size;
    }

    if (ws_.softplus_sqrt_size > 0) {
      const size_t slice_offset = ws_.total_size - ws_.softplus_sqrt_size;
      auto slice =
          std::make_shared<Tensor>(workspace_buffer_->location, TYPE_UINT8, std::vector<size_t>{ws_.softplus_sqrt_size},
                                   reinterpret_cast<int8_t*>(workspace_buffer_->GetPtr<void>()) + slice_offset);
      topk_softplus_sqrt_layer_->SetWorkspaceBuffer(slice);
    }
  }

  const size_t num_tokens = hidden_states->shape[0];
  const size_t hidden_size = hidden_states->shape[1];

  // Execute TopK gating (sqrtsoftplus only)
  if (do_profile) {
    cudaStream_t stream = context_->device->GetPrimaryComputeStream().Get();
    RecordEvent(Evt::GateStart, stream);
  }
  STATUS_CHECK_RETURN(
      ExecuteTopkSoftplusSqrt<T>(hidden_states, gating_output, eplb_map, workspace, e_score_bias, input_ids));
  if (do_profile) {
    cudaStream_t stream = context_->device->GetPrimaryComputeStream().Get();
    RecordEvent(Evt::GateEnd, stream);
  }

  // Get weight pointers
  void* w1_ptr = w1->GetPtr<void>();
  void* w2_ptr = w2->GetPtr<void>();
  void* w1_scales = (w1->weight_scales != nullptr) ? w1->weight_scales->GetPtr<void>() : nullptr;
  void* w2_scales = (w2->weight_scales != nullptr) ? w2->weight_scales->GetPtr<void>() : nullptr;

  // W4A8: Get per-expert global_scale from weight tensors
  if (humming_compute_dtype_ == llm_kernels::nvidia::humming::ComputeDtype::FP8) {
    if (w1->weight_global_scale != nullptr) {
      w1_global_scale_ = w1->weight_global_scale->GetPtr<float>();
    }
    if (w2->weight_global_scale != nullptr) {
      w2_global_scale_ = w2->weight_global_scale->GetPtr<float>();
    }
  }

  auto ProcessTokens = [&](void* hidden_states_ptr, void* output_ptr, size_t total_tokens) {
    cudaStream_t stream = context_->device->GetPrimaryComputeStream().Get();
    const bool use_expert_parallel = (global_expert_para_size_ > 1 && !using_deepep_);

    // No chunking needed — humming uses stream-K internally, process all tokens at once
    void* curr_topk_ids = ws_.gating.topk_ids;
    void* curr_topk_weights = ws_.gating.topk_weights;

    // ----- Determine moe_block_size from GEMM1's kernel BLOCK_SHAPE_M -----
    int64_t size_n_w1 = static_cast<int64_t>(expert_inter_size_ * 2);
    int64_t size_k_w1 = static_cast<int64_t>(hidden_size);

    // Profiled kernel selection: right-align lookup (smallest profiled size_m >= actual).
    auto sel1 = SelectGemmMeta(size_n_w1, size_k_w1, static_cast<int64_t>(total_tokens));
    int32_t moe_block_size_i32 = sel1.block_m;
    int64_t moe_block_size = static_cast<int64_t>(moe_block_size_i32);

    // Pre-compute GEMM2's block_m to enable merged align
    int64_t size_n_w2 = static_cast<int64_t>(hidden_size);
    int64_t size_k_w2 = static_cast<int64_t>(expert_inter_size_);
    int64_t size_m_gemm2 = static_cast<int64_t>(total_tokens * expert_topk_);
    auto sel2 = SelectGemmMeta(size_n_w2, size_k_w2, size_m_gemm2);
    int32_t moe_block_size_gemm2 = sel2.block_m;

    int64_t numel = static_cast<int64_t>(total_tokens * expert_topk_);
    int64_t max_num_tokens_padded = numel + expert_num_per_node_ * (moe_block_size - 1);
    int64_t max_num_m_blocks = (max_num_tokens_padded + moe_block_size - 1) / moe_block_size;

    // Layout within fused_id_buffer: [fill_info(4)] [sorted_ids] [expert_ids] [num_tokens_post_pad] [cumsum]
    void* fill_info_buffer_on_device = ws_.align_gemm1.base;
    void* curr_fused_id_buffer =
        static_cast<char*>(fill_info_buffer_on_device) + SGL_MOE_ALIGN_BLOCK_VEC_SIZE * sizeof(int32_t);
    int id_buffer_offset = 0;

    int32_t* sorted_ids_ptr = reinterpret_cast<int32_t*>(curr_fused_id_buffer) + id_buffer_offset;

    // Declare GEMM1 align output pointers (used by both merged and non-merged paths)
    int32_t* expert_ids_ptr = nullptr;
    int32_t* num_tokens_post_pad_ptr = nullptr;
    int32_t* cumsum_ptr = nullptr;

    // Prepare GEMM2 align buffers (needed if block_m differs)
    int64_t max_num_tokens_padded_g2 = numel + expert_num_per_node_ * (moe_block_size_gemm2 - 1);
    int64_t max_num_m_blocks_g2 = (max_num_tokens_padded_g2 + moe_block_size_gemm2 - 1) / moe_block_size_gemm2;

    void* fill_info_g2 = ws_.align_gemm2.base;
    void* curr_fused_g2 = static_cast<char*>(fill_info_g2) + SGL_MOE_ALIGN_BLOCK_VEC_SIZE * sizeof(int32_t);
    int id_off_g2 = 0;
    int32_t* sorted_ids_gemm2 = reinterpret_cast<int32_t*>(curr_fused_g2) + id_off_g2;
    int32_t* expert_ids_gemm2;
    int32_t* num_tokens_post_pad_gemm2;
    int32_t* cumsum_g2;

    const bool use_merged_align = (moe_block_size_gemm2 != moe_block_size_i32) && (expert_num_per_node_ >= 224);

    if (use_merged_align) {
      // Merged align: compute both GEMM1 and GEMM2 alignment in one launch
      if (do_profile) RecordEvent(Evt::Align1Start, stream);
      InvokeFillInfo(reinterpret_cast<int32_t*>(curr_fused_id_buffer), fill_info_buffer_on_device, id_buffer_offset,
                     static_cast<int>(max_num_tokens_padded), static_cast<int>(numel), stream);
      id_buffer_offset += max_num_tokens_padded;
      expert_ids_ptr = reinterpret_cast<int32_t*>(curr_fused_id_buffer) + id_buffer_offset;
      id_buffer_offset += max_num_m_blocks;
      num_tokens_post_pad_ptr = reinterpret_cast<int32_t*>(curr_fused_id_buffer) + id_buffer_offset;
      cumsum_ptr = reinterpret_cast<int32_t*>(curr_fused_id_buffer) + id_buffer_offset + 1;

      InvokeFillInfo(reinterpret_cast<int32_t*>(curr_fused_g2), fill_info_g2, id_off_g2,
                     static_cast<int>(max_num_tokens_padded_g2), static_cast<int>(numel), stream);
      id_off_g2 += max_num_tokens_padded_g2;
      expert_ids_gemm2 = reinterpret_cast<int32_t*>(curr_fused_g2) + id_off_g2;
      id_off_g2 += max_num_m_blocks_g2;
      num_tokens_post_pad_gemm2 = reinterpret_cast<int32_t*>(curr_fused_g2) + id_off_g2;
      cumsum_g2 = reinterpret_cast<int32_t*>(curr_fused_g2) + id_off_g2 + 1;

      llm_kernels::nvidia::InvokeSglMoeAlignBlockSizeMerged<int32_t>(
          reinterpret_cast<int32_t*>(curr_topk_ids), sorted_ids_ptr, expert_ids_ptr, num_tokens_post_pad_ptr,
          sorted_ids_gemm2, expert_ids_gemm2, num_tokens_post_pad_gemm2, static_cast<int32_t>(max_num_tokens_padded),
          static_cast<int32_t>(max_num_tokens_padded_g2), static_cast<int32_t>(expert_num_per_node_),
          moe_block_size_i32, moe_block_size_gemm2, static_cast<size_t>(numel), cumsum_ptr, cumsum_g2, stream);
      if (do_profile) RecordEvent(Evt::Align1End, stream);
      if (do_profile) RecordEvent(Evt::Align2Start, stream);
      if (do_profile) RecordEvent(Evt::Align2End, stream);
    } else {
      // Original path: separate align for GEMM1, GEMM2 align deferred
      if (do_profile) RecordEvent(Evt::Align1Start, stream);
      InvokeFillInfo(reinterpret_cast<int32_t*>(curr_fused_id_buffer), fill_info_buffer_on_device, id_buffer_offset,
                     static_cast<int>(max_num_tokens_padded), static_cast<int>(numel), stream);
      id_buffer_offset += max_num_tokens_padded;
      expert_ids_ptr = reinterpret_cast<int32_t*>(curr_fused_id_buffer) + id_buffer_offset;
      id_buffer_offset += max_num_m_blocks;
      num_tokens_post_pad_ptr = reinterpret_cast<int32_t*>(curr_fused_id_buffer) + id_buffer_offset;
      cumsum_ptr = reinterpret_cast<int32_t*>(curr_fused_id_buffer) + id_buffer_offset + 1;

      InvokeAlignBlockSize(reinterpret_cast<int32_t*>(curr_topk_ids), sorted_ids_ptr, expert_ids_ptr,
                           num_tokens_post_pad_ptr, cumsum_ptr, static_cast<int32_t>(max_num_tokens_padded),
                           static_cast<size_t>(numel), static_cast<int32_t>(moe_block_size),
                           static_cast<int32_t>(expert_topk_), use_expert_parallel,
                           static_cast<int32_t>(global_expert_para_size_), rank_, stream);
      if (do_profile) RecordEvent(Evt::Align1End, stream);
      // GEMM2 align will be done later if block_m differs
      sorted_ids_gemm2 = sorted_ids_ptr;
      expert_ids_gemm2 = nullptr;  // will be set in deferred align
      num_tokens_post_pad_gemm2 = nullptr;
    }

    // ----- GEMM 1: hidden_states × w1 → intermediate_cache1 [tokens*topk, inter_size*2] -----

    // W4A8: Quantize activations to FP8 before GEMM1
    const void* gemm1_a = hidden_states_ptr;
    const void* gemm1_a_scale = nullptr;
    const void* gemm1_global_scale = nullptr;

    if (do_profile) RecordEvent(Evt::Quant1Start, stream);
    if (QuantizeToFp8<T>(hidden_states_ptr, static_cast<int>(size_k_w1), static_cast<int>(total_tokens), stream,
                         gemm1_a, gemm1_a_scale)) {
      gemm1_global_scale = w1_global_scale_;
    }
    if (do_profile) RecordEvent(Evt::Quant1End, stream);

    // Zero locks buffer before kernel launch to prevent deadlock from stale lock values
    cudaMemsetAsync(ws_.kernel.locks, 0, ws_.kernel.locks_size, stream);

    if (do_profile) {
      RecordEvent(Evt::Gemm1Start, stream);
      perf_stats_->last_gemm1_meta = sel1.meta;
    }

    llm_kernels::nvidia::humming::humming_moe_gemm(
        gemm1_a,                                            // a
        w1_ptr,                                             // b (FP4 WGMMA layout)
        ws_.cache.gemm1_output,                             // c
        gemm1_a_scale,                                      // a_scale (nullptr for W4A16)
        w1_scales,                                          // b_scale
        gemm1_global_scale,                                 // global_scale
        nullptr,                                            // b_zero_point
        nullptr,                                            // bias
        sorted_ids_ptr,                                     // sorted_ids
        expert_ids_ptr,                                     // expert_ids
        num_tokens_post_pad_ptr,                            // num_tokens_padded
        nullptr,                                            // expert_layout (INDEXED uses sorted_ids)
        reinterpret_cast<const float*>(curr_topk_weights),  // topk_weights
        ws_.kernel.locks,                                   // locks
        ws_.kernel.tma_descriptors,                         // workspace
        llm_kernels::nvidia::humming::GemmType::INDEXED, humming_compute_dtype_,
        static_cast<int64_t>(total_tokens),          // size_m
        size_n_w1,                                   // size_n
        size_k_w1,                                   // size_k
        static_cast<int64_t>(expert_num_per_node_),  // num_experts
        static_cast<int64_t>(expert_topk_),          // top_k
        32,                                          // group_size
        stream,
        sel1.required_block_m,  // >0 falls back to block_m search when no meta
        sel1.meta);             // forced_meta (profiled/selected, or nullptr)

    if (do_profile) RecordEvent(Evt::Gemm1End, stream);

    const size_t elements_num = total_tokens * expert_topk_ * expert_inter_size_ * 2;
    const bool use_fused_silu_quant = (humming_compute_dtype_ == llm_kernels::nvidia::humming::ComputeDtype::FP8);
    if (do_profile) RecordEvent(Evt::SiluStart, stream);
    if (use_fused_silu_quant) {
      // W4A8: fused SiluAndMul + FP8 Quant in one Triton kernel.
      // Reads [M*topk, inter*2] BF16, writes [M*topk, inter] FP8 + [M*topk] FP32 scale.
      const int num_silu_tokens = static_cast<int>(total_tokens * expert_topk_);
      Singleton<TritonWrapper>::GetInstance()->InvokeHummingFusedSiluAndMulFp8Quant<T>(
          reinterpret_cast<const T*>(ws_.cache.gemm1_output), ws_.fp8.scales, ws_.fp8.activation,
          static_cast<int>(expert_inter_size_), num_silu_tokens, swiglu_limit_, stream);
    } else if (use_expert_parallel) {
      if (has_swiglu_limit_) {
        llm_kernels::nvidia::InvokeSiluAndMul<T, true, true>(
            reinterpret_cast<const T*>(ws_.cache.gemm1_output), reinterpret_cast<T*>(ws_.cache.silu_output),
            reinterpret_cast<const int*>(curr_topk_ids), elements_num, expert_inter_size_, swiglu_limit_, stream);
      } else {
        llm_kernels::nvidia::InvokeSiluAndMul<T, true, false>(
            reinterpret_cast<const T*>(ws_.cache.gemm1_output), reinterpret_cast<T*>(ws_.cache.silu_output),
            reinterpret_cast<const int*>(curr_topk_ids), elements_num, expert_inter_size_, 0.0f, stream);
      }
    } else {
      if (has_swiglu_limit_) {
        llm_kernels::nvidia::InvokeSiluAndMul<T, false, true>(
            reinterpret_cast<const T*>(ws_.cache.gemm1_output), reinterpret_cast<T*>(ws_.cache.silu_output),
            reinterpret_cast<const int*>(curr_topk_ids), elements_num, expert_inter_size_, swiglu_limit_, stream);
      } else {
        llm_kernels::nvidia::InvokeSiluAndMul<T, false, false>(
            reinterpret_cast<const T*>(ws_.cache.gemm1_output), reinterpret_cast<T*>(ws_.cache.silu_output),
            reinterpret_cast<const int*>(curr_topk_ids), elements_num, expert_inter_size_, 0.0f, stream);
      }
    }
    if (do_profile) RecordEvent(Evt::SiluEnd, stream);

    // ----- GEMM 2: intermediate_cache2 × w2 → intermediate_cache3 [tokens*topk, hidden_size] -----
    // size_k_w2, size_n_w2, size_m_gemm2, sel2, moe_block_size_gemm2 already computed above

    // GEMM2 alignment: if merged align was used, buffers are already set.
    // If not merged and block_m differs, do deferred align now.
    if (!use_merged_align) {
      if (do_profile) RecordEvent(Evt::Align2Start, stream);
      if (moe_block_size_gemm2 != moe_block_size_i32) {
        int64_t numel_g2 = static_cast<int64_t>(total_tokens * expert_topk_);
        int64_t max_num_tokens_padded_g2_def = numel_g2 + expert_num_per_node_ * (moe_block_size_gemm2 - 1);
        int64_t max_num_m_blocks_g2_def =
            (max_num_tokens_padded_g2_def + moe_block_size_gemm2 - 1) / moe_block_size_gemm2;

        void* fill_info_g2_def = ws_.align_gemm2.base;
        void* curr_fused_g2_def = static_cast<char*>(fill_info_g2_def) + SGL_MOE_ALIGN_BLOCK_VEC_SIZE * sizeof(int32_t);
        int id_off_g2_def = 0;

        sorted_ids_gemm2 = reinterpret_cast<int32_t*>(curr_fused_g2_def) + id_off_g2_def;

        InvokeFillInfo(reinterpret_cast<int32_t*>(curr_fused_g2_def), fill_info_g2_def, id_off_g2_def,
                       static_cast<int>(max_num_tokens_padded_g2_def), static_cast<int>(numel_g2), stream);
        id_off_g2_def += max_num_tokens_padded_g2_def;

        expert_ids_gemm2 = reinterpret_cast<int32_t*>(curr_fused_g2_def) + id_off_g2_def;
        id_off_g2_def += max_num_m_blocks_g2_def;

        num_tokens_post_pad_gemm2 = reinterpret_cast<int32_t*>(curr_fused_g2_def) + id_off_g2_def;
        int32_t* cumsum_g2_def = reinterpret_cast<int32_t*>(curr_fused_g2_def) + id_off_g2_def + 1;

        InvokeAlignBlockSize(reinterpret_cast<int32_t*>(curr_topk_ids), sorted_ids_gemm2, expert_ids_gemm2,
                             num_tokens_post_pad_gemm2, cumsum_g2_def,
                             static_cast<int32_t>(max_num_tokens_padded_g2_def), static_cast<size_t>(numel_g2),
                             static_cast<int32_t>(moe_block_size_gemm2), static_cast<int32_t>(expert_topk_),
                             use_expert_parallel, static_cast<int32_t>(global_expert_para_size_), rank_, stream);
      } else {
        // block_m same: reuse GEMM1's buffers
        sorted_ids_gemm2 = sorted_ids_ptr;
        expert_ids_gemm2 = expert_ids_ptr;
        num_tokens_post_pad_gemm2 = num_tokens_post_pad_ptr;
      }
      if (do_profile) RecordEvent(Evt::Align2End, stream);
    }

    // W4A8: Quantize activations to FP8 before GEMM2
    const void* gemm2_a = ws_.cache.silu_output;
    const void* gemm2_a_scale = nullptr;
    const void* gemm2_global_scale = nullptr;

    if (do_profile) RecordEvent(Evt::Quant2Start, stream);
    if (use_fused_silu_quant) {
      // Fused path: SiluAndMul + FP8 quant already done above.
      // Quant2Start/Quant2End are back-to-back (0 interval) — quant work is fused into Silu phase.
      gemm2_a = ws_.fp8.activation;
      gemm2_a_scale = ws_.fp8.scales;
      gemm2_global_scale = w2_global_scale_;
    } else if (QuantizeToFp8<T>(ws_.cache.silu_output, static_cast<int>(size_k_w2),
                                static_cast<int>(total_tokens * expert_topk_), stream, gemm2_a, gemm2_a_scale)) {
      gemm2_global_scale = w2_global_scale_;
    }
    if (do_profile) RecordEvent(Evt::Quant2End, stream);

    // Zero locks buffer before kernel launch to prevent deadlock from stale lock values
    cudaMemsetAsync(ws_.kernel.locks, 0, ws_.kernel.locks_size, stream);

    if (do_profile) {
      RecordEvent(Evt::Gemm2Start, stream);
      perf_stats_->last_gemm2_meta = sel2.meta;
    }

    // NOTE: For GEMM2, top_k must be 1 (not expert_topk_). The activation tensor
    // intermediate_cache2 is already expanded to [tokens*topk, inter_size] — each row is
    // one token-expert pair. With top_k=1, the kernel uses sorted_ids[i] directly as the
    // row index into intermediate_cache2 (no division by top_k). Using expert_topk_ here
    // would cause the kernel to divide sorted_ids by topk, reading only 1/topk of the rows.

    llm_kernels::nvidia::humming::humming_moe_gemm(
        gemm2_a,                     // a
        w2_ptr,                      // b (FP4 WGMMA layout)
        ws_.cache.gemm2_output,      // c
        gemm2_a_scale,               // a_scale (nullptr for W4A16)
        w2_scales,                   // b_scale
        gemm2_global_scale,          // global_scale
        nullptr,                     // b_zero_point
        nullptr,                     // bias
        sorted_ids_gemm2,            // sorted_ids (GEMM2-specific)
        expert_ids_gemm2,            // expert_ids (GEMM2-specific)
        num_tokens_post_pad_gemm2,   // num_tokens_padded (GEMM2-specific)
        nullptr,                     // expert_layout (INDEXED uses sorted_ids)
        nullptr,                     // topk_weights (not used by humming kernel)
        ws_.kernel.locks,            // locks
        ws_.kernel.tma_descriptors,  // workspace
        llm_kernels::nvidia::humming::GemmType::INDEXED, humming_compute_dtype_,
        static_cast<int64_t>(total_tokens * expert_topk_),  // size_m (already expanded)
        size_n_w2,                                          // size_n
        size_k_w2,                                          // size_k
        static_cast<int64_t>(expert_num_per_node_),         // num_experts
        1,                                                  // top_k = 1 (data already expanded)
        32,                                                 // group_size
        stream,
        sel2.required_block_m,  // >0 falls back to block_m search when no meta
        sel2.meta);             // forced_meta (profiled/selected, or nullptr)

    if (do_profile) RecordEvent(Evt::Gemm2End, stream);

    // ----- Fused MulTopkWeight + Sum: intermediate_cache3 → output [tokens, hidden_size] -----
    if (do_profile) RecordEvent(Evt::MulSumStart, stream);
    if (use_expert_parallel) {
      llm_kernels::nvidia::InvokeMoeFusedMulSum<T, true>(
          ws_.cache.gemm2_output, reinterpret_cast<const float*>(curr_topk_weights), curr_topk_ids, output_ptr,
          static_cast<int>(total_tokens), static_cast<int>(expert_topk_), static_cast<int>(hidden_size), stream);
    } else {
      llm_kernels::nvidia::InvokeMoeFusedMulSum<T, false>(
          ws_.cache.gemm2_output, reinterpret_cast<const float*>(curr_topk_weights), curr_topk_ids, output_ptr,
          static_cast<int>(total_tokens), static_cast<int>(expert_topk_), static_cast<int>(hidden_size), stream);
    }
    if (do_profile) RecordEvent(Evt::MulSumEnd, stream);

    // Record per-GEMM times (humming has no chunk loop, single set of events per Forward;
    // accumulate across Forward calls so the caller can sum over a benchmark loop)
    if (do_profile) {
      cudaEventSynchronize(events_[static_cast<int>(Evt::MulSumEnd)]);
      perf_stats_->gating_time_ms += ElapsedMs(Evt::GateStart, Evt::GateEnd);
      perf_stats_->align_time_ms +=
          ElapsedMs(Evt::Align1Start, Evt::Align1End) + ElapsedMs(Evt::Align2Start, Evt::Align2End);
      perf_stats_->quant_time_ms +=
          ElapsedMs(Evt::Quant1Start, Evt::Quant1End) + ElapsedMs(Evt::Quant2Start, Evt::Quant2End);
      perf_stats_->gemm1_time_ms += ElapsedMs(Evt::Gemm1Start, Evt::Gemm1End);
      perf_stats_->silu_time_ms += ElapsedMs(Evt::SiluStart, Evt::SiluEnd);
      perf_stats_->gemm2_time_ms += ElapsedMs(Evt::Gemm2Start, Evt::Gemm2End);
      perf_stats_->mulsum_time_ms += ElapsedMs(Evt::MulSumStart, Evt::MulSumEnd);
    }
  };

  if (using_deepep_) {
    // LCOV_EXCL_START
    const size_t dispatch_num_tokens = workspace->shape[0];
    if (dispatch_num_tokens > 0) {
      expert_map_->InvokeExpertMapInverseInplace(static_cast<int32_t*>(ws_.gating.topk_ids),
                                                 dispatch_num_tokens * expert_topk_,
                                                 context_->device->GetPrimaryComputeStream().Get());
      ProcessTokens(workspace->GetPtr<void>(), workspace->GetPtr<void>(), dispatch_num_tokens);
    }
    std::vector<Tensor> combine_outputs = {*output};
    Combine({*workspace}, combine_outputs);
    *output = combine_outputs[0];
    // LCOV_EXCL_STOP
  } else if (global_expert_para_size_ > 1) {
    // EP without deepep
    expert_map_->InvokeExpertMapInplace(static_cast<int32_t*>(ws_.gating.topk_ids), num_tokens * expert_topk_,
                                        context_->device->GetPrimaryComputeStream().Get());
    ProcessTokens(hidden_states->GetPtr<void>(), output->GetPtr<void>(), num_tokens);
  } else {
    // Single-node, no EP
    ProcessTokens(hidden_states->GetPtr<void>(), output->GetPtr<void>(), num_tokens);
  }

  output->shape = hidden_states->shape;
  output->dtype = hidden_states->dtype;
  return Status();
}

template <typename T>
Status HummingFp4MoeLayer::ExecuteTopkSoftplusSqrt(const Tensor* hidden_states, const Tensor* gating_output,
                                                   const Tensor* eplb_map, Tensor* workspace,
                                                   const Tensor* e_score_bias, const Tensor* input_ids) {
  const size_t num_tokens = hidden_states->shape[0];

  // aux1: input_ids (hash) or correction_bias; aux2: tid2eid (hash only).
  const Tensor* aux1_ptr = nullptr;
  const Tensor* aux2_ptr = nullptr;
  if (use_hash_) {
    KLLM_CHECK_WITH_INFO(input_ids != nullptr,
                         "HummingFp4MoeLayer: hash path requires input_ids and e_score_bias (tid2eid)");
    aux1_ptr = input_ids;     // input_ids
    aux2_ptr = e_score_bias;  // tid2eid
  } else if (use_e_score_correction_bias_) {
    aux1_ptr = e_score_bias;  // correction_bias
  }

  Tensor topk_weights_tensor(gating_output->location, TYPE_FP32, {num_tokens, static_cast<size_t>(expert_topk_)},
                             ws_.gating.topk_weights);
  Tensor topk_ids_tensor(gating_output->location, TYPE_INT32, {num_tokens, static_cast<size_t>(expert_topk_)},
                         ws_.gating.topk_ids);
  Status status =
      topk_softplus_sqrt_layer_->Forward(gating_output, &topk_weights_tensor, &topk_ids_tensor, aux1_ptr, aux2_ptr);
  if (!status.OK()) {
    KLLM_LOG_ERROR << "HummingFp4MoeLayer::ExecuteTopkSoftplusSqrt ERROR: forward failed";
    return status;
  }

  if (enable_dump_eplb_data_) {
    DumpEplbData(topk_ids_tensor);
  }

  if (enable_load_eplb_config_ && eplb_map->shape[0] > static_cast<size_t>(layer_idx_)) {
    size_t num_experts = eplb_map->shape[1];
    void* layer_expert_ptr =
        reinterpret_cast<uint8_t*>(eplb_map->GetPtr<void>()) + layer_idx_ * num_experts * sizeof(int);
    expert_map_->InvokeExpertMapInplace(static_cast<int32_t*>(ws_.gating.topk_ids), num_tokens * expert_topk_,
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

Status HummingFp4MoeLayer::Forward(const Tensor* hidden_states, const Tensor* gating_output, const Tensor* w1,
                                   const Tensor* w2, const Tensor* eplb_map, Tensor* output, Tensor* workspace,
                                   const Tensor* e_score_bias, const Tensor* input_ids,
                                   const LayerExecContext& exec_ctx) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, hidden_states, gating_output, w1, w2, eplb_map, output, workspace,
                      e_score_bias, input_ids);
}

}  // namespace ksana_llm
