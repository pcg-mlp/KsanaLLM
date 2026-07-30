/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================

Test for the single_group_topk kernel ported from vLLM PR #50000.
Uses libtorch for input generation and reference computation.

NOTE: This test's structure (helper functions, torch reference implementation,
comparison logic) serves as the template for Task 3's SGLang cross-validation
test. Task 3 must replicate this test infrastructure, replacing only the kernel
call.

==============================================================================*/

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <utility>

#include "csrc/kernels/nvidia/others/vllm/main/single_group_topk/single_group_topk.h"
#include "csrc/utils/nvidia/cuda_utils.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

namespace llm_kernels {
namespace nvidia {
namespace test {

class SingleGroupTopkTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }
  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  using NvidiaTestSuitBase::stream;

  // torch 参考实现：sigmoid(scores) + bias → topk selection → unbiased weights → 归一化 + scaling
  // The kernel uses biased scores (sigmoid+bias) for expert SELECTION but
  // unbiased scores (sigmoid only) for routing WEIGHTS. This matches vLLM's
  // correction_bias semantics.
  // 返回 (topk_weights [N, topk] fp32, topk_ids [N, topk] int32)
  std::pair<torch::Tensor, torch::Tensor> RefSigmoidTopk(
      torch::Tensor gating_output, int topk, bool renormalize,
      double routed_scaling_factor, torch::Tensor correction_bias) {
    auto scores = torch::sigmoid(gating_output.to(torch::kFloat32));
    auto scores_with_bias = correction_bias.defined()
        ? (scores + correction_bias.to(torch::kFloat32).unsqueeze(0))
        : scores;
    // Select topk experts by biased score
    auto topk_i = std::get<1>(torch::topk(scores_with_bias, topk, -1, true, true));
    // Routing weights use unbiased scores (sigmoid without bias)
    auto topk_w = scores.gather(1, topk_i.to(torch::kLong));
    if (renormalize) {
      topk_w = topk_w / topk_w.sum(-1, true);
    }
    topk_w = topk_w * routed_scaling_factor;
    return {topk_w.to(torch::kFloat32).contiguous(),
            topk_i.to(torch::kInt32).contiguous()};
  }

  // 统一测试 helper：生成输入 → 调用 kernel → 与 torch 参考对比
  void RunOneCase(int num_tokens, int num_experts, int topk,
                  bool renormalize, double routed_scaling_factor,
                  bool use_bias, torch::ScalarType input_dtype,
                  int64_t seed = 42, float w_atol = 1e-5f, float w_rtol = 1e-5f) {
    torch::manual_seed(seed);
    torch::cuda::manual_seed(seed);
    auto input_opts = torch::TensorOptions().device(torch::kCUDA).dtype(input_dtype);
    auto f32_opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);

    auto gating = torch::randn({num_tokens, num_experts}, input_opts);
    // The kernel always reads bias[expert]; pass zeros when no correction bias.
    torch::Tensor bias = torch::zeros({num_experts}, f32_opts);
    if (use_bias) {
      bias = torch::randn({num_experts}, f32_opts);
    }

    // kernel 调用
    auto topk_w = torch::empty({num_tokens, topk}, f32_opts);
    auto topk_i = torch::empty({num_tokens, topk},
        torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32));

    // dtype 分派 → 调用 InvokeVllmSingleGroupTopk
    bool handled = false;
    if (input_dtype == torch::kFloat32) {
      handled = InvokeVllmSingleGroupTopk<float>(
          static_cast<float*>(gating.data_ptr()), static_cast<float*>(topk_w.data_ptr()),
          static_cast<int32_t*>(topk_i.data_ptr()),
          static_cast<float*>(bias.data_ptr()),
          num_tokens, num_experts, topk, renormalize, routed_scaling_factor,
          true, stream);
    } else if (input_dtype == torch::kBFloat16) {
      handled = InvokeVllmSingleGroupTopk<__nv_bfloat16>(
          static_cast<__nv_bfloat16*>(gating.data_ptr()), static_cast<float*>(topk_w.data_ptr()),
          static_cast<int32_t*>(topk_i.data_ptr()),
          static_cast<float*>(bias.data_ptr()),
          num_tokens, num_experts, topk, renormalize, routed_scaling_factor,
          true, stream);
    } else if (input_dtype == torch::kFloat16) {
      handled = InvokeVllmSingleGroupTopk<half>(
          static_cast<half*>(gating.data_ptr()), static_cast<float*>(topk_w.data_ptr()),
          static_cast<int32_t*>(topk_i.data_ptr()),
          static_cast<float*>(bias.data_ptr()),
          num_tokens, num_experts, topk, renormalize, routed_scaling_factor,
          true, stream);
    }
    cudaStreamSynchronize(stream);

    // 未命中 tier 的 case：验证返回 false
    // (100, 24) is truly unmatched: topk=24 exceeds the max topk (22) in any
    // SigmoidBiasTier, so dispatch returns false.
    if (num_experts == 100 && topk == 24) {
      EXPECT_FALSE(handled) << "expected false for unsupported tier (100, 24)";
      return;
    }
    EXPECT_TRUE(handled) << "expected true for supported tier";

    // 与 torch 参考对比：先按 expert id 排序再 allclose
    auto [ref_w, ref_i] = RefSigmoidTopk(gating, topk, renormalize,
                                         routed_scaling_factor, bias);
    auto kernel_sorted = std::get<0>(topk_i.sort(-1));
    auto ref_sorted = std::get<0>(ref_i.sort(-1));
    ASSERT_TRUE(torch::equal(kernel_sorted, ref_sorted))
        << "topk_ids mismatch\nref:\n" << ref_sorted << "\nkernel:\n" << kernel_sorted;

    auto perm_kernel = std::get<1>(topk_i.sort(-1));
    auto perm_ref = std::get<1>(ref_i.sort(-1));
    ASSERT_TRUE(torch::allclose(
        topk_w.gather(1, perm_kernel), ref_w.gather(1, perm_ref),
        /*rtol=*/w_rtol, /*atol=*/w_atol))
        << "topk_weights mismatch";
  }
};

// K3 精确形状
TEST_F(SingleGroupTopkTestSuit, K3_896_16_FP32_WithBias) {
  RunOneCase(128, 896, 16, true, 1.0, true, torch::kFloat32);
}
TEST_F(SingleGroupTopkTestSuit, K3_896_16_BF16_WithBias) {
  RunOneCase(128, 896, 16, true, 1.0, true, torch::kBFloat16,
             42, /*w_atol=*/2e-2f, /*w_rtol=*/1e-2f);
}
// 其他 tier
TEST_F(SingleGroupTopkTestSuit, DSv3_256_8_FP32_WithBias) {
  RunOneCase(128, 256, 8, true, 1.5, true, torch::kFloat32);
}
TEST_F(SingleGroupTopkTestSuit, Tier_512_16_FP32_WithBias) {
  RunOneCase(64, 512, 16, true, 1.0, true, torch::kFloat32);
}
TEST_F(SingleGroupTopkTestSuit, Tier_1024_16_FP32_WithBias) {
  RunOneCase(64, 1024, 16, true, 1.0, true, torch::kFloat32);
}
// 未命中 tier (topk=24 exceeds max tier topk=22)
TEST_F(SingleGroupTopkTestSuit, UnmatchedTier_100_24) {
  RunOneCase(64, 100, 24, true, 1.0, true, torch::kFloat32);
}
// 多 token 数
TEST_F(SingleGroupTopkTestSuit, K3_896_16_SingleToken) {
  RunOneCase(1, 896, 16, true, 1.0, true, torch::kFloat32);
}
TEST_F(SingleGroupTopkTestSuit, K3_896_16_LargeBatch) {
  RunOneCase(1024, 896, 16, true, 1.0, true, torch::kFloat32);
}
// 无 bias / 无 renormalize
TEST_F(SingleGroupTopkTestSuit, K3_896_16_FP32_NoBias) {
  RunOneCase(128, 896, 16, true, 1.0, false, torch::kFloat32);
}
TEST_F(SingleGroupTopkTestSuit, K3_896_16_FP32_NoRenorm) {
  RunOneCase(128, 896, 16, false, 1.0, true, torch::kFloat32);
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
