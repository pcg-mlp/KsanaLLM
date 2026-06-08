/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#ifdef ENABLE_FP8
#  include "layers/matmul_base_layer.h"
#  ifdef ENABLE_CUDA
#    include "csrc/kernels/nvidia/gemm/deepgemm/deepgemm_wrapper.h"
#    include "kernels/nvidia/deepseek_deepgemm_wrapper.h"
#  endif

namespace ksana_llm {

class BlockwiseMatMulLayer : public MatMulBaseLayer {
 public:
  enum class Fp8GemmType : int32_t { Dynamic, Cutlass, DeepGemm, DeepGemmSwapAB, BridgeDeepGemm };

  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;

  virtual Status Forward(const Tensor* input, const Tensor* weight, Tensor* output,
                         const LayerExecContext& exec_ctx = {}) override;

  virtual size_t GetWorkspaceSize() override;

  virtual Status Preprocess(const ModelConfig& model_config_, const RuntimeConfig& runtime_config) override;

  // --- test access ---
  void SetGemmType(BlockwiseMatMulLayer::Fp8GemmType gemm_type);
  BlockwiseMatMulLayer::Fp8GemmType PickGemmType(size_t m);

  // Profiling search results: Phase 1 thresholds + Phase 2 point-based lookup
  struct GemmTuningResult {
    size_t deepgemm_max_m_threshold = 0;
    std::vector<size_t> swap_ab_max_m_thresholds;
    std::vector<size_t> phase2_keys;
    std::vector<Fp8GemmType> phase2_types;
  };
  GemmTuningResult search_cache_;
  BlockwiseMatMulLayer::Fp8GemmType gemm_type_ = BlockwiseMatMulLayer::Fp8GemmType::Dynamic;

 private:
  template <typename T>
  Status ForwardT(const Tensor& input, const Tensor& weight, Tensor& output, const LayerExecContext& exec_ctx);

  template <typename T>
  size_t GetWorkspaceSizeT();

  template <typename T>
  size_t GetCachedCutlassBufferSize();

  void BuildDeepGemmKernels();

 private:
  size_t max_m_;
  size_t n_;
  size_t k_;
  size_t block_size_;
  Tensor weight_;

  size_t workspace_size_ = 0;
  size_t input_buffer_size_ = 0;
  size_t cutlass_gemm_workspace_size_ = 0;

  bool deepgemm_enabled_ = false;

  // When true, it indicates that quantization has already been performed elsewhere and stored in the workspace buffer
  bool skip_quant_ = false;

  float swiglu_limit_ = 0.0f;

  static constexpr size_t kAlignSize_ = 4;
  // wgmma on hopper requires `m = 64`
  // See https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-shape
  // So the execution time of DeepGEMM increases in steps of 64
  static constexpr size_t kWgmmaBlockM_ = 64;

  // Phase 1 search: small step size and threshold for adaptive step
  static constexpr size_t kPhase1Step_ = 8;
  static constexpr size_t kPhase1StepThreshold_ = 2048;

  // Phase 2 search step: 512 for m < 8192, 1024 for m >= 8192
  static constexpr size_t kPhase2StepSmall_ = 512;
  static constexpr size_t kPhase2StepLarge_ = 1024;
  static constexpr size_t kPhase2StepThreshold_ = 8192;

#  ifdef ENABLE_CUDA
  std::shared_ptr<llm_kernels::nvidia::DeepGEMMWrapper> trtllm_deepgemm_wrapper_;
  ksana_llm::nvidia::DeepSeekDeepGEMMWrapper* bridge_deepgemm_wrapper_ = nullptr;
#  endif
};

}  // namespace ksana_llm
#endif
