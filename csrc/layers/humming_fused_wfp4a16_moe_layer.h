/* Copyright 2026 Tencent Inc.  All rights reserved.
==============================================================================*/
#pragma once

#include <cstdint>
#include <memory>
#include <mutex>

#ifdef ENABLE_CUDA
#  include <torch/torch.h>

#  include "csrc/kernels/nvidia/moe/expert_map/expert_map.h"
#  include "csrc/kernels/nvidia/others/humming/jit/humming_jit_compiler.h"
#  include "csrc/kernels/nvidia/others/humming/launcher/kernel_meta.h"
#  include "csrc/kernels/nvidia/others/humming/ops/ops.h"
#  include "csrc/kernels/nvidia/others/humming/ops/wrapper.h"
#  include "csrc/utils/nvidia/workspace.h"
#  include "csrc/utils/search_status.h"
#  include "layers/nvidia/humming_jit_cache.h"
#endif
#include "distributed/expert_parallel/expert_parallel_manager.h"
#include "layers/base_moe_layer.h"
#include "layers/moe_perf_stats.h"
#include "layers/topk_softplus_sqrt_layer.h"
#include "models/base/base_weight.h"
#include "models/common_moe/moe_config.h"

namespace ksana_llm {

#ifdef ENABLE_CUDA
class HummingFusedWfp4A16MoeLayer : public BaseMoeLayer {
 public:
  virtual ~HummingFusedWfp4A16MoeLayer() {
    for (auto& e : events_) {
      if (e) cudaEventDestroy(e);
    }
  }

  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;

  virtual void SetExpertParallelManager(std::shared_ptr<ExpertParallelManager> expert_parallel_manager) override;

  virtual size_t GetWorkspaceSize() override;

  virtual Status Preprocess(const ModelConfig& model_config, const RuntimeConfig& runtime_config) override;

  // eplb_map/e_score_bias/input_ids are nullptr when unused.
  virtual Status Forward(const Tensor* hidden_states, const Tensor* gating_output, const Tensor* w1, const Tensor* w2,
                         const Tensor* eplb_map, Tensor* output, Tensor* workspace,
                         const Tensor* e_score_bias = nullptr, const Tensor* input_ids = nullptr,
                         const LayerExecContext& exec_ctx = {}) override;

  // Attach a profiling stats output (nullptr = disabled, default).
  // Non-virtual — HummingFusedWfp4A16MoeLayer owns this API directly.
  void SetPerfStats(MoePerfStats* stats) { perf_stats_ = stats; }

  // Test-only: when true, Forward records profiling events but defers the
  // cudaEventSynchronize + ElapsedMs read to a single FlushPerfStats() call after
  // the benchmark loop. Eliminates the per-Forward sync bubble. Production callers
  // (perf_stats_==nullptr) are unaffected; non-test callers that attach perf_stats_
  // without setting this keep the legacy per-Forward flush.
  void SetDeferPerfFlush(bool defer) { defer_perf_flush_ = defer; }

  // Read the last Forward's event markers and scale by `iterations` to populate
  // perf_stats_ as an accumulated sum (matching the legacy contract). No-op if
  // perf_stats_ is null or defer_perf_flush_ is false. Called by the test driver
  // after its own cudaEventSynchronize(end).
  void FlushPerfStats(size_t iterations);

  // Expose the JIT compiler's kernel table for test/inspection (nullptr if no compiler).
  // The table contains all JIT-compiled kernel variants for all shapes/dtypes compiled
  // by any HummingFusedWfp4A16MoeLayer instance (compiler is the process-global singleton
  // from humming_jit_cache.h, shared across instances).
  static const std::vector<llm_kernels::nvidia::humming::KernelMeta>* GetJitKernelTable();

 private:
  template <typename T>
  Status InitT(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
               std::shared_ptr<Context> context, int rank);

  // Initialize TopKSoftplusSqrt gating sub-layer and query its workspace size.
  void InitTopkSoftplusSqrt(const RuntimeConfig& runtime_config, int rank);

  template <typename T>
  size_t GetWorkspaceSizeT();

  template <typename T>
  Status ForwardT(const Tensor* hidden_states, const Tensor* gating_output, const Tensor* w1, const Tensor* w2,
                  const Tensor* eplb_map, Tensor* output, Tensor* workspace, const Tensor* e_score_bias,
                  const Tensor* input_ids);

  template <typename T>
  Status ExecuteTopkSoftplusSqrt(const Tensor* hidden_states, const Tensor* gating_output, const Tensor* eplb_map,
                                 Tensor* workspace, const Tensor* e_score_bias, const Tensor* input_ids);

  Status Dispatch(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);
  Status Combine(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);

  Status DumpEplbData(Tensor& topk_ids);

  // --- Deduplicated helpers used by both Forward and Profiling ---

  // Select GEMM kernel meta: profiled lookup with heuristic fallback.
  // Returns meta=nullptr with required_block_m>0 when no profiled/heuristic match
  // is found, triggering block_m-based fallback search in humming_moe_gemm.
  struct GemmMetaSelection {
    const llm_kernels::nvidia::humming::KernelMeta* meta = nullptr;
    int32_t block_m = 0;
    int32_t required_block_m = 0;  // >0 when meta is null → block_m-based fallback search
  };
  GemmMetaSelection SelectGemmMeta(int64_t size_n, int64_t size_k, int64_t size_m) const;

  // --- Profiling event infrastructure ---
  // 18 markers covering the full Forward pipeline with no gaps:
  //   Gate | Align1 | Quant1 | Gemm1 | Silu | Align2 | Quant2 | Gemm2 | MulSum
  // Each phase has a Start and End marker. Phases that don't apply (e.g. quant
  // for W4A16, align2 when block_m matches GEMM1) have back-to-back markers → 0 interval.
  enum class Evt : int {
    GateStart,
    GateEnd,
    Align1Start,
    Align1End,
    Quant1Start,
    Quant1End,
    Gemm1Start,
    Gemm1End,
    SiluStart,
    SiluEnd,
    Align2Start,
    Align2End,
    Quant2Start,
    Quant2End,
    Gemm2Start,
    Gemm2End,
    MulSumStart,
    MulSumEnd,
    kCount
  };

  void EnsureProfilingEvents() {
    if (events_.empty()) {
      events_.resize(static_cast<size_t>(Evt::kCount));
      for (auto& e : events_) cudaEventCreate(&e);
    }
  }
  void RecordEvent(Evt e, cudaStream_t stream) { cudaEventRecord(events_[static_cast<int>(e)], stream); }
  float ElapsedMs(Evt a, Evt b) const {
    float ms = 0.f;
    cudaEventElapsedTime(&ms, events_[static_cast<int>(a)], events_[static_cast<int>(b)]);
    return ms;
  }

  // Invoke MoE align block size kernel (EP/TP template branching).
  void InvokeAlignBlockSize(int32_t* topk_ids, int32_t* sorted_ids, int32_t* expert_ids, int32_t* num_tokens_post_pad,
                            int32_t* cumsum, int32_t max_num_tokens_padded, size_t numel, int32_t moe_block_size,
                            bool use_expert_parallel, cudaStream_t stream);

  // Invoke merged MoE align for two block_m values in one launch (EP/TP template branching).
  void InvokeMergedAlignBlockSize(int32_t* topk_ids, int32_t* sorted_ids1, int32_t* expert_ids1,
                                  int32_t* num_tokens_post_pad1, int32_t* sorted_ids2, int32_t* expert_ids2,
                                  int32_t* num_tokens_post_pad2, int32_t max_num_tokens_padded1,
                                  int32_t max_num_tokens_padded2, int32_t moe_block_size1, int32_t moe_block_size2,
                                  size_t numel, int32_t* cumsum1, int32_t* cumsum2, bool use_expert_parallel,
                                  cudaStream_t stream);

  // Prepared dummy weights for profiling (generated once per GEMM shape, reused across buckets).
  struct ProfWeights {
    torch::Tensor b;        // WGMMA-repacked weight [E, N, K/2] uint8
    torch::Tensor b_scale;  // processed scales [E, K/32, N] uint8
  };

  // Per-block_m align buffer set for profiling: sorted_ids, expert_ids, num_tokens_post_pad.
  struct ProfAlignBuffers {
    torch::Tensor sorted_ids;
    torch::Tensor expert_ids;
    torch::Tensor num_tokens_post_pad;
    torch::Tensor cumsum;  // Only used for SglMoeAlignBlockSize (num_experts >= 224)
    ProfAlignBuffers(int64_t max_num_tokens_padded, int64_t max_num_m_blocks, int64_t num_experts);
  };

  // Generate dummy FP4 weights + scales and process them through the same
  // pipeline as test (repack_weight_wgmma + process_scales).
  // Called once per GEMM shape, reused across all buckets.
  ProfWeights PrepareDummyWeights(int64_t size_n, int64_t size_k, int64_t num_experts, cudaStream_t stream);

  // Prepare dummy topk_ids with random uniform expert distribution (fixed seed=42).
  // Matches the benchmark's expert distribution (benchmark uses torch::randn for gate
  // values, then topk → random uniform expert assignment). At small M this naturally
  // produces sparse/skewed load (most experts get 0 tokens), at large M it converges
  // to uniform. Fixed seed ensures reproducibility across runs.
  //
  // EP -1 modeling: when ep_size > 1, each pair is set to -1 (non-local expert) with
  // probability (ep_size-1)/ep_size, mimicking the per-rank topk_ids after expert_map
  // remap in Forward (InvokeExpertMapInplace). Combined with use_expert_parallel=true
  // in InvokeAlignBlockSize, this makes the profile's num_tokens_post_pad match the
  // real Forward workload instead of overestimating it by ~ep_size x. ep_size=1 (TP)
  // is unchanged: 0% -1, fully local distribution (identical to the original behavior).
  static torch::Tensor MakeDummyTopkIds(int64_t num_tokens, int64_t top_k, int64_t num_experts, int64_t ep_size = 1);

  // Profile a single GEMM shape at a given size_m with pre-prepared dummy weights.
  // Tests all matching candidate kernels, returns the fastest KernelMeta*.
  const llm_kernels::nvidia::humming::KernelMeta* ProfileSingleGemm(int64_t size_m, int64_t size_n, int64_t size_k,
                                                                    int64_t num_experts, int64_t top_k, size_t warmup,
                                                                    size_t profile_iters, cudaStream_t stream,
                                                                    const ProfWeights& weights);

  // Profile all token buckets for one GEMM shape, returning size_m→KernelMeta* schedule.
  HummingMoeSearchStatus::ProfiledSchedule ProfileAllBuckets(int64_t n, int64_t k, int64_t e, int64_t top_k,
                                                             size_t warmup, size_t profile_iters,
                                                             int64_t size_m_multiplier);

 protected:
  bool set_workspace_buffer_info_ = true;

  bool has_swiglu_limit_ = false;

  llm_kernels::nvidia::humming::ComputeDtype humming_compute_dtype_;

  // Profiling iteration count (read once from QUANT_PROFILE env in Init).
  // When 0, profiling is disabled and only heuristic kernels are compiled.
  size_t profile_iters_ = 15;
  size_t warmup_iters_ = 5;
  size_t profile_m_thresh1_ = 512;  // rep_m above this → reduced iters (2/3 of profile_iters)

  // --- Workspace layout (all carved from a single contiguous workspace_buffer_) ---
  // Layout order:
  //   [gating.topk_weights] [gating.topk_ids]
  //   [align_gemm1] [align_gemm2]
  //   [cache.gemm1_output / cache.gemm2_output (shared)] [cache.silu_output]
  //   [kernel.locks] [kernel.tma_descriptors]
  //   ← softplus_sqrt slice from tail (time-safe: gating runs before ProcessTokens) →

  struct GatingWorkspace {
    void* topk_weights = nullptr;  // [M*topk] float
    void* topk_ids = nullptr;      // [M*topk] int32
    size_t topk_weights_size = 0;
    size_t topk_ids_size = 0;
  };

  struct AlignWorkspace {
    void* base = nullptr;  // [fill_info(4) + sorted_ids + expert_ids + ntpp + cumsum] int32
    size_t total_size = 0;
  };

  struct ActivationCache {
    void* gemm1_output = nullptr;  // cache1: [M*topk, inter*2] — GEMM1 output
    void* silu_output = nullptr;   // cache2: [M*topk, inter] — SiluAndMul output
    void* gemm2_output = nullptr;  // cache3: [M*topk, hidden] — GEMM2 output (reuses gemm1_output)
    size_t gemm1_size = 0;
    size_t silu_size = 0;
    size_t gemm2_size = 0;
  };

  struct HummingKernelWorkspace {
    int32_t* locks = nullptr;         // stream-K sync locks
    void* tma_descriptors = nullptr;  // TMA descriptor buffer
    size_t locks_size = 0;
    size_t tma_size = 0;
  };

  struct HummingWorkspace {
    GatingWorkspace gating;
    AlignWorkspace align_gemm1;
    AlignWorkspace align_gemm2;
    ActivationCache cache;
    HummingKernelWorkspace kernel;
    size_t softplus_sqrt_size = 0;  // aligned size for tail slice (set in GetWorkspaceSizeT)
    size_t total_size = 0;          // total workspace bytes
  };

  HummingWorkspace ws_;
  size_t softplus_sqrt_workspace_size_ = 0;  // raw size from topk layer (set in InitT)

  // EP / EPLB
  size_t global_expert_para_size_;
  bool using_deepep_ = false;

  std::shared_ptr<TopKSoftplusSqrtLayer> topk_softplus_sqrt_layer_;
  bool use_hash_ = false;

  bool enable_load_eplb_config_ = false;
  bool enable_dump_eplb_data_ = false;
  int eplb_dump_step_ = 0;
  std::string eplb_dump_path_;
  std::shared_ptr<llm_kernels::nvidia::moe::ExpertMap> expert_map_;

  std::shared_ptr<ExpertParallelManager> expert_parallel_manager_;

  // GPU events for per-phase timing (18 events, created lazily, reused across forwards).
  // Indexed by Evt enum; see EnsureProfilingEvents().
  std::vector<cudaEvent_t> events_;

  // Profiling output (nullptr = disabled)
  MoePerfStats* perf_stats_ = nullptr;

  bool defer_perf_flush_ = false;
};
#endif

}  // namespace ksana_llm
