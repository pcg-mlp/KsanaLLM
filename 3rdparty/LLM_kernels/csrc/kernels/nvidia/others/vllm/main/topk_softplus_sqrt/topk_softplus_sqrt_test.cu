/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <tuple>
#include <utility>

#include "csrc/kernels/nvidia/others/vllm/main/topk_softplus_sqrt/topk_softplus_sqrt.h"
#include "csrc/utils/nvidia/cuda_utils.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace nvidia {
namespace test {

class TopkSoftplusSqrtTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }
  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  using NvidiaTestSuitBase::stream;

  // Mirrors _torch_topk_softplus_sqrt from vLLM tests/kernels/moe/test_topk_softplus_sqrt.py.
  // Returns (topk_weights [N, topk] float32, topk_ids [N, topk] int32), both CUDA.
  std::pair<torch::Tensor, torch::Tensor> RefTopkSoftplusSqrt(torch::Tensor gating_output, int topk, bool renormalize,
                                                              double routed_scaling_factor,
                                                              torch::Tensor correction_bias, torch::Tensor input_ids,
                                                              torch::Tensor hash_indices_table) {
    auto scores = torch::sqrt(torch::nn::functional::softplus(gating_output.to(torch::kFloat32)));
    auto original_scores = scores;
    auto scores_for_choice =
        correction_bias.defined() ? (scores + correction_bias.to(torch::kFloat32).unsqueeze(0)) : scores;

    torch::Tensor topk_ids;
    if (hash_indices_table.defined()) {
      topk_ids = hash_indices_table.index_select(0, input_ids.to(torch::kLong));
    } else {
      topk_ids = std::get<1>(torch::topk(scores_for_choice, topk, /*dim=*/-1, /*largest=*/true, /*sorted=*/true));
    }

    auto topk_weights = original_scores.gather(1, topk_ids.to(torch::kLong));
    if (renormalize) {
      topk_weights = topk_weights / topk_weights.sum(-1, /*keepdim=*/true);
    }
    if (routed_scaling_factor != 1.0) {
      topk_weights = topk_weights * routed_scaling_factor;
    }
    return {topk_weights.to(torch::kFloat32).contiguous(), topk_ids.to(torch::kInt32).contiguous()};
  }

  static tensorrt_llm::dev::ScalarType KScalarTypeFor(torch::ScalarType t) {
    if (t == torch::kFloat32) return tensorrt_llm::dev::ScalarType::Float;
    if (t == torch::kFloat16) return tensorrt_llm::dev::ScalarType::Float16;
    if (t == torch::kBFloat16) return tensorrt_llm::dev::ScalarType::BFloat16;
    if (t == torch::kInt32) return tensorrt_llm::dev::ScalarType::Int;
    if (t == torch::kInt64) return tensorrt_llm::dev::ScalarType::Long;
    EXPECT_TRUE(false) << "Unsupported torch dtype";
    return tensorrt_llm::dev::ScalarType::Float;
  }

  // Runs one (dtype, mode) case through the kernel and compares to the torch reference.
  // input_dtype ∈ {torch::kFloat32, torch::kFloat16, torch::kBFloat16}.
  // index_dtype ∈ {torch::kInt32, torch::kInt64}.
  void RunOneCase(int num_tokens, int num_experts, int topk, bool renormalize, double routed_scaling_factor,
                  bool use_hash, bool use_bias, torch::ScalarType input_dtype, torch::ScalarType index_dtype,
                  int64_t seed = 42, float w_atol = 1e-5f, float w_rtol = 1e-5f) {
    ASSERT_TRUE(index_dtype == torch::kInt32 || index_dtype == torch::kInt64);
    ASSERT_FALSE(use_hash && use_bias) << "hash + correction_bias is not a supported combination";

    torch::manual_seed(seed);
    auto input_opts = torch::TensorOptions().device(torch::kCUDA).dtype(input_dtype);
    auto float_opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    auto idx_opts = torch::TensorOptions().device(torch::kCUDA).dtype(index_dtype);

    auto gating_output = torch::randn({num_tokens, num_experts}, input_opts);

    torch::Tensor correction_bias;
    if (use_bias) {
      correction_bias = torch::randn({num_experts}, float_opts);
    }

    torch::Tensor input_ids;
    torch::Tensor hash_indices_table;
    if (use_hash) {
      constexpr int kVocab = 1024;
      std::vector<torch::Tensor> rows;
      rows.reserve(kVocab);
      for (int i = 0; i < kVocab; ++i) {
        rows.push_back(torch::randperm(num_experts).slice(0, 0, topk));
      }
      hash_indices_table = torch::stack(rows).to(idx_opts);
      input_ids = torch::randint(0, kVocab, {num_tokens}, idx_opts);
    }

    // Allocate outputs and scratch buffer for kernel.
    auto topk_weights = torch::empty({num_tokens, topk}, float_opts);
    auto topk_indices = torch::empty({num_tokens, topk}, idx_opts);
    auto token_expert_indices =
        torch::empty({num_tokens, topk}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32));

    InvokeTopkSoftplusSqrt(gating_output.data_ptr(), topk_weights.data_ptr(), topk_indices.data_ptr(),
                           token_expert_indices.data_ptr(), num_tokens, num_experts, topk, renormalize,
                           routed_scaling_factor,
                           correction_bias.defined() ? correction_bias.data_ptr<float>() : nullptr,
                           input_ids.defined() ? input_ids.data_ptr() : nullptr,
                           hash_indices_table.defined() ? hash_indices_table.data_ptr() : nullptr,
                           KScalarTypeFor(input_dtype), KScalarTypeFor(index_dtype), stream);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    auto ref = RefTopkSoftplusSqrt(gating_output, topk, renormalize, routed_scaling_factor, correction_bias, input_ids,
                                   hash_indices_table);
    auto ref_weights = ref.first;
    auto ref_ids_i32 = ref.second;

    // Bring kernel results into a common dtype for comparison (int32 ids).
    auto kernel_ids_i32 = topk_indices.to(torch::kInt32);

    // Different kernels can return selected experts in different orders when scores
    // are close. Sort by expert id before comparing.
    auto ref_sorted = ref_ids_i32.sort(/*dim=*/-1);
    auto kernel_sorted = kernel_ids_i32.sort(/*dim=*/-1);
    auto sorted_ref_ids = std::get<0>(ref_sorted);
    auto perm_ref = std::get<1>(ref_sorted);
    auto sorted_kernel_ids = std::get<0>(kernel_sorted);
    auto perm_kernel = std::get<1>(kernel_sorted);
    ASSERT_TRUE(torch::equal(sorted_ref_ids, sorted_kernel_ids)) << "topk_ids mismatch after sort.\nref:\n"
                                                                 << sorted_ref_ids << "\nkernel:\n"
                                                                 << sorted_kernel_ids;

    auto sorted_ref_w = ref_weights.gather(1, perm_ref);
    auto sorted_kernel_w = topk_weights.gather(1, perm_kernel);
    ASSERT_TRUE(torch::allclose(sorted_ref_w, sorted_kernel_w, /*rtol=*/w_rtol, /*atol=*/w_atol))
        << "topk_weights mismatch.\nref:\n"
        << sorted_ref_w << "\nkernel:\n"
        << sorted_kernel_w;
  }
};

// DeepSeek V4 Flash: 256 experts, topk=6, hash path, float input, renormalize, scale=1.5.
TEST_F(TopkSoftplusSqrtTestSuit, HashFloat_V4Flash) {
  RunOneCase(/*num_tokens=*/128, /*num_experts=*/256, /*topk=*/6, /*renormalize=*/true,
             /*routed_scaling_factor=*/1.5, /*use_hash=*/true, /*use_bias=*/false,
             /*input_dtype=*/torch::kFloat32, /*index_dtype=*/torch::kInt32);
}

// DeepSeek V4 Pro: 384 experts, topk=6, hash, float, renormalize, scale=2.5.
TEST_F(TopkSoftplusSqrtTestSuit, HashFloat_V4Pro) {
  RunOneCase(128, 384, 6, true, 2.5, true, false, torch::kFloat32, torch::kInt32);
}

TEST_F(TopkSoftplusSqrtTestSuit, HashBF16_V4Flash) {
  RunOneCase(128, 256, 6, true, 1.5, true, false, torch::kBFloat16, torch::kInt32,
             /*seed=*/42, /*w_atol=*/2e-2f, /*w_rtol=*/1e-2f);
}

TEST_F(TopkSoftplusSqrtTestSuit, HashBF16_V4Pro) {
  RunOneCase(128, 384, 6, true, 2.5, true, false, torch::kBFloat16, torch::kInt32,
             /*seed=*/42, /*w_atol=*/2e-2f, /*w_rtol=*/1e-2f);
}

TEST_F(TopkSoftplusSqrtTestSuit, HashHalf_V4Flash) {
  RunOneCase(128, 256, 6, true, 1.5, true, false, torch::kFloat16, torch::kInt32,
             /*seed=*/42, /*w_atol=*/5e-3f, /*w_rtol=*/5e-3f);
}

TEST_F(TopkSoftplusSqrtTestSuit, HashHalf_V4Pro) {
  RunOneCase(128, 384, 6, true, 2.5, true, false, torch::kFloat16, torch::kInt32,
             /*seed=*/42, /*w_atol=*/5e-3f, /*w_rtol=*/5e-3f);
}

TEST_F(TopkSoftplusSqrtTestSuit, HashFloat_NoRenormalize) {
  RunOneCase(64, 256, 6, /*renormalize=*/false, 1.5, true, false, torch::kFloat32, torch::kInt32);
}

TEST_F(TopkSoftplusSqrtTestSuit, HashFloat_Scale1) {
  RunOneCase(64, 256, 6, true, /*routed_scaling_factor=*/1.0, true, false, torch::kFloat32, torch::kInt32);
}

TEST_F(TopkSoftplusSqrtTestSuit, HashInt64Indices) {
  RunOneCase(64, 256, 6, true, 1.5, true, false, torch::kFloat32, /*index_dtype=*/torch::kInt64);
}

TEST_F(TopkSoftplusSqrtTestSuit, HashSingleToken) {
  RunOneCase(1, 256, 6, true, 1.5, true, false, torch::kFloat32, torch::kInt32);
}

TEST_F(TopkSoftplusSqrtTestSuit, HashLargeBatch) {
  RunOneCase(1024, 256, 6, true, 1.5, true, false, torch::kFloat32, torch::kInt32);
}

TEST_F(TopkSoftplusSqrtTestSuit, NonHashFloat_256_NoBias) {
  RunOneCase(128, 256, 6, true, 1.5, /*use_hash=*/false, /*use_bias=*/false, torch::kFloat32, torch::kInt32);
}

TEST_F(TopkSoftplusSqrtTestSuit, NonHashFloat_256_WithBias) {
  RunOneCase(128, 256, 6, true, 1.5, false, /*use_bias=*/true, torch::kFloat32, torch::kInt32);
}

TEST_F(TopkSoftplusSqrtTestSuit, NonHashBF16_256_WithBias) {
  RunOneCase(128, 256, 6, true, 1.5, false, true, torch::kBFloat16, torch::kInt32,
             /*seed=*/42, /*w_atol=*/2e-2f, /*w_rtol=*/1e-2f);
}

TEST_F(TopkSoftplusSqrtTestSuit, NonHashHalf_256_WithBias) {
  RunOneCase(128, 256, 6, true, 1.5, false, true, torch::kFloat16, torch::kInt32,
             /*seed=*/42, /*w_atol=*/5e-3f, /*w_rtol=*/5e-3f);
}

TEST_F(TopkSoftplusSqrtTestSuit, NonHashFloat_TopK1) {
  RunOneCase(64, 256, /*topk=*/1, true, 1.5, false, false, torch::kFloat32, torch::kInt32);
}

TEST_F(TopkSoftplusSqrtTestSuit, NonHashFloat_TopK16) {
  RunOneCase(64, 256, /*topk=*/16, true, 1.5, false, false, torch::kFloat32, torch::kInt32);
}

// Power-of-2 num_experts specializations.
TEST_F(TopkSoftplusSqrtTestSuit, NumExperts_128) {
  RunOneCase(64, 128, 6, true, 1.5, true, false, torch::kFloat32, torch::kInt32);
}
TEST_F(TopkSoftplusSqrtTestSuit, NumExperts_512) {
  RunOneCase(64, 512, 6, true, 1.5, true, false, torch::kFloat32, torch::kInt32);
}

// Multiple-of-64 num_experts specializations (CUDA-only path in vLLM).
TEST_F(TopkSoftplusSqrtTestSuit, NumExperts_192) {
  RunOneCase(64, 192, 6, true, 1.5, true, false, torch::kFloat32, torch::kInt32);
}
TEST_F(TopkSoftplusSqrtTestSuit, NumExperts_320) {
  RunOneCase(64, 320, 6, true, 1.5, true, false, torch::kFloat32, torch::kInt32);
}
TEST_F(TopkSoftplusSqrtTestSuit, NumExperts_448) {
  RunOneCase(64, 448, 6, true, 1.5, true, false, torch::kFloat32, torch::kInt32);
}
TEST_F(TopkSoftplusSqrtTestSuit, NumExperts_576) {
  RunOneCase(64, 576, 6, true, 1.5, true, false, torch::kFloat32, torch::kInt32);
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
