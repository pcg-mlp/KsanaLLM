/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cuda_runtime.h>
#include <torch/torch.h>
#include <memory>

#include "device/nvidia/deepseek_deepgemm_bridge.h"
#include "utils/logger.h"
#include "utils/tensor.h"

namespace ksana_llm {
namespace nvidia {

// fp8_paged_mqa_logits kernel only supports q_seq_len <= 2
constexpr int kDeepGemmMaxQSeqLen = 2;

class DeepSeekDeepGEMMWrapper {
 public:
  static DeepSeekDeepGEMMWrapper* GetInstance();

  DeepSeekDeepGEMMWrapper();
  ~DeepSeekDeepGEMMWrapper() = default;

  // MQA attention functions using DeepSeek's DeepGEMM
  void Fp8MqaLogits(const torch::Tensor& q_fp8, const std::pair<torch::Tensor, torch::Tensor>& kv_fp8,
                    const torch::Tensor& weights, const torch::Tensor& cu_seq_len_k_start,
                    const torch::Tensor& cu_seq_len_k_end, torch::Tensor& logits, int max_seqlen_k,
                    cudaStream_t stream);

  void Fp8PagedMqaLogits(const torch::Tensor& q_fp8, const torch::Tensor& fused_kv_cache, const torch::Tensor& weights,
                         const torch::Tensor& context_lens, const torch::Tensor& block_table,
                         const torch::Tensor& schedule_meta, torch::Tensor& logits, int max_context_len,
                         cudaStream_t stream, bool clean_logits = false);

  // Generate schedule metadata for paged MQA
  void PagedMqaLogitsMetadata(const torch::Tensor& context_lens, torch::Tensor& schedule_metadata, int batch_size,
                              int block_kv, cudaStream_t stream);

  // Fused norm and gemm for mHC
  void Tf32HcPreNormGemm(const Tensor& a_tensor, const Tensor& b_tensor, Tensor& d_tensor, Tensor& sqr_sum_tensor,
                         const int num_splits, const int rank, cudaStream_t stream);

 private:
  int num_device_sms_;
};

}  // namespace nvidia
}  // namespace ksana_llm
