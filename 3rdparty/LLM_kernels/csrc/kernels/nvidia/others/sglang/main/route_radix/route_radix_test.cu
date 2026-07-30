/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================

Test for the route_radix kernel ported from SGLang PR #32541.
Uses libtorch for input generation and reference computation.

This test replicates the structure of Task 2's single_group_topk_test.cu
(same RefSigmoidTopk reference, same RunOneCase helper signature, same test
cases). The ONLY difference is that RunOneCase calls both
InvokeSglangRouteRadix (SGLang) and InvokeVllmSingleGroupTopk (vLLM) and
cross-validates them.

IMPORTANT: bias cannot be nullptr — both kernels read bias[expert]
unconditionally. Pass a zero-filled tensor when no correction bias is needed.

==============================================================================*/

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <utility>

#include "csrc/kernels/nvidia/others/sglang/main/route_radix/route_radix.h"
#include "csrc/kernels/nvidia/others/vllm/main/single_group_topk/single_group_topk.h"
#include "csrc/utils/nvidia/cuda_utils.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

using llm_kernels::nvidia::sglang_gate::InvokeSglangRouteRadix;
using llm_kernels::nvidia::InvokeVllmSingleGroupTopk;

namespace llm_kernels {
namespace nvidia {
namespace test {

class RouteRadixTestSuit : public NvidiaTestSuitBase {
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

  // 统一测试 helper：生成输入 → 调用两个 kernel → 交叉对比
  // 签名与 Task 2 的 RunOneCase 完全一致，但内部调用 SGLang + vLLM 并交叉对比
  void RunOneCase(int num_tokens, int num_experts, int topk,
                  bool renormalize, double routed_scaling_factor,
                  bool use_bias, torch::ScalarType input_dtype,
                  int64_t seed = 42, float w_atol = 1e-5f, float w_rtol = 1e-5f) {
    torch::manual_seed(seed);
    torch::cuda::manual_seed(seed);
    auto input_opts = torch::TensorOptions().device(torch::kCUDA).dtype(input_dtype);
    auto f32_opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    auto int32_opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32);

    auto gating = torch::randn({num_tokens, num_experts}, input_opts);
    // The kernel always reads bias[expert]; pass zeros when no correction bias.
    torch::Tensor bias = torch::zeros({num_experts}, f32_opts);
    if (use_bias) {
      bias = torch::randn({num_experts}, f32_opts);
    }

    // SGLang kernel 调用 (硬编码 896/16，非此形状时 throw)
    auto sg_w = torch::empty({num_tokens, topk}, f32_opts);
    auto sg_i = torch::empty({num_tokens, topk}, int32_opts);
    bool sglang_launched = false;
    try {
      if (input_dtype == torch::kFloat32) {
        InvokeSglangRouteRadix<float>(
            static_cast<float*>(gating.data_ptr()),
            static_cast<const float*>(bias.data_ptr()),
            static_cast<float*>(sg_w.data_ptr()),
            static_cast<int32_t*>(sg_i.data_ptr()),
            num_tokens, num_experts, topk, renormalize,
            static_cast<float>(routed_scaling_factor), true, stream);
        sglang_launched = true;
      } else if (input_dtype == torch::kBFloat16) {
        InvokeSglangRouteRadix<__nv_bfloat16>(
            static_cast<__nv_bfloat16*>(gating.data_ptr()),
            static_cast<const float*>(bias.data_ptr()),
            static_cast<float*>(sg_w.data_ptr()),
            static_cast<int32_t*>(sg_i.data_ptr()),
            num_tokens, num_experts, topk, renormalize,
            static_cast<float>(routed_scaling_factor), true, stream);
        sglang_launched = true;
      } else if (input_dtype == torch::kFloat16) {
        InvokeSglangRouteRadix<half>(
            static_cast<half*>(gating.data_ptr()),
            static_cast<const float*>(bias.data_ptr()),
            static_cast<float*>(sg_w.data_ptr()),
            static_cast<int32_t*>(sg_i.data_ptr()),
            num_tokens, num_experts, topk, renormalize,
            static_cast<float>(routed_scaling_factor), true, stream);
        sglang_launched = true;
      }
    } catch (const std::runtime_error&) {
      // SGLang kernel 硬编码 896/16，非此形状时 throw — expected
    }

    // vLLM kernel 调用（交叉验证）
    auto vllm_w = torch::empty({num_tokens, topk}, f32_opts);
    auto vllm_i = torch::empty({num_tokens, topk}, int32_opts);
    bool vllm_handled = false;
    if (input_dtype == torch::kFloat32) {
      vllm_handled = InvokeVllmSingleGroupTopk<float>(
          static_cast<float*>(gating.data_ptr()),
          static_cast<float*>(vllm_w.data_ptr()),
          static_cast<int32_t*>(vllm_i.data_ptr()),
          static_cast<float*>(bias.data_ptr()),
          num_tokens, num_experts, topk, renormalize, routed_scaling_factor,
          true, stream);
    } else if (input_dtype == torch::kBFloat16) {
      vllm_handled = InvokeVllmSingleGroupTopk<__nv_bfloat16>(
          static_cast<__nv_bfloat16*>(gating.data_ptr()),
          static_cast<float*>(vllm_w.data_ptr()),
          static_cast<int32_t*>(vllm_i.data_ptr()),
          static_cast<float*>(bias.data_ptr()),
          num_tokens, num_experts, topk, renormalize, routed_scaling_factor,
          true, stream);
    } else if (input_dtype == torch::kFloat16) {
      vllm_handled = InvokeVllmSingleGroupTopk<half>(
          static_cast<half*>(gating.data_ptr()),
          static_cast<float*>(vllm_w.data_ptr()),
          static_cast<int32_t*>(vllm_i.data_ptr()),
          static_cast<float*>(bias.data_ptr()),
          num_tokens, num_experts, topk, renormalize, routed_scaling_factor,
          true, stream);
    }
    cudaStreamSynchronize(stream);

    if (!sglang_launched) {
      // SGLang 不支持的形状：验证 SGLang 确实 throw 了（到达这里说明 catch 成功）
      // 对 vLLM 的验证与 Task 2 一致
      if (num_experts == 100 && topk == 24) {
        EXPECT_FALSE(vllm_handled) << "expected false for unsupported tier (100, 24)";
      } else {
        EXPECT_TRUE(vllm_handled) << "expected true for supported tier";
        // 与 torch 参考对比
        auto [ref_w, ref_i] = RefSigmoidTopk(gating, topk, renormalize,
                                             routed_scaling_factor, bias);
        auto kernel_sorted = std::get<0>(vllm_i.sort(-1));
        auto ref_sorted = std::get<0>(ref_i.sort(-1));
        ASSERT_TRUE(torch::equal(kernel_sorted, ref_sorted))
            << "topk_ids mismatch (vllm vs ref)\nref:\n" << ref_sorted
            << "\nkernel:\n" << kernel_sorted;
        auto perm_kernel = std::get<1>(vllm_i.sort(-1));
        auto perm_ref = std::get<1>(ref_i.sort(-1));
        ASSERT_TRUE(torch::allclose(
            vllm_w.gather(1, perm_kernel), ref_w.gather(1, perm_ref),
            /*rtol=*/w_rtol, /*atol=*/w_atol))
            << "topk_weights mismatch (vllm vs ref)";
      }
      return;
    }

    // SGLang 成功启动：交叉对比 SGLang vs vLLM
    EXPECT_TRUE(vllm_handled) << "vLLM should also handle this shape";

    // 先按 expert id 排序再 allclose
    auto sg_sorted = std::get<0>(sg_i.sort(-1));
    auto vllm_sorted = std::get<0>(vllm_i.sort(-1));
    ASSERT_TRUE(torch::equal(sg_sorted, vllm_sorted))
        << "topk_ids mismatch between sglang and vllm";

    auto perm_sg = std::get<1>(sg_i.sort(-1));
    auto perm_vllm = std::get<1>(vllm_i.sort(-1));
    ASSERT_TRUE(torch::allclose(
        sg_w.gather(1, perm_sg), vllm_w.gather(1, perm_vllm),
        /*rtol=*/w_rtol, /*atol=*/w_atol))
        << "topk_weights mismatch between sglang and vllm";
  }
};

// K3 精确形状 — SGLang + vLLM 交叉验证
TEST_F(RouteRadixTestSuit, K3_896_16_FP32_WithBias) {
  RunOneCase(128, 896, 16, true, 1.0, true, torch::kFloat32);
}
TEST_F(RouteRadixTestSuit, K3_896_16_BF16_WithBias) {
  RunOneCase(128, 896, 16, true, 1.0, true, torch::kBFloat16,
             42, /*w_atol=*/2e-2f, /*w_rtol=*/1e-2f);
}
// 其他 tier — SGLang throw, vLLM vs ref
TEST_F(RouteRadixTestSuit, DSv3_256_8_FP32_WithBias) {
  RunOneCase(128, 256, 8, true, 1.5, true, torch::kFloat32);
}
TEST_F(RouteRadixTestSuit, Tier_512_16_FP32_WithBias) {
  RunOneCase(64, 512, 16, true, 1.0, true, torch::kFloat32);
}
TEST_F(RouteRadixTestSuit, Tier_1024_16_FP32_WithBias) {
  RunOneCase(64, 1024, 16, true, 1.0, true, torch::kFloat32);
}
// 未命中 tier (topk=24 exceeds max tier topk=22)
TEST_F(RouteRadixTestSuit, UnmatchedTier_100_24) {
  RunOneCase(64, 100, 24, true, 1.0, true, torch::kFloat32);
}
// 多 token 数 — SGLang + vLLM 交叉验证
TEST_F(RouteRadixTestSuit, K3_896_16_SingleToken) {
  RunOneCase(1, 896, 16, true, 1.0, true, torch::kFloat32);
}
TEST_F(RouteRadixTestSuit, K3_896_16_LargeBatch) {
  RunOneCase(1024, 896, 16, true, 1.0, true, torch::kFloat32);
}
// 无 bias / 无 renormalize — SGLang + vLLM 交叉验证
TEST_F(RouteRadixTestSuit, K3_896_16_FP32_NoBias) {
  RunOneCase(128, 896, 16, true, 1.0, false, torch::kFloat32);
}
TEST_F(RouteRadixTestSuit, K3_896_16_FP32_NoRenorm) {
  RunOneCase(128, 896, 16, false, 1.0, true, torch::kFloat32);
}
// 边界用例 — SGLang + vLLM 交叉验证
TEST_F(RouteRadixTestSuit, K3_896_16_FP16_WithBias) {
  RunOneCase(128, 896, 16, true, 1.0, true, torch::kFloat16,
             42, /*w_atol=*/2e-2f, /*w_rtol=*/1e-2f);
}
TEST_F(RouteRadixTestSuit, K3_896_16_AllEqualScores) {
  // 全相等 scores: 所有 expert 分数相同 → topk 选择应确定 (id 最小的 topk 个)
  torch::manual_seed(42);
  torch::cuda::manual_seed(42);
  auto f32_opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
  auto int32_opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32);
  int num_tokens = 4, num_experts = 896, topk = 16;
  auto gating = torch::full({num_tokens, num_experts}, 0.5, f32_opts);
  auto bias = torch::zeros({num_experts}, f32_opts);

  auto sg_w = torch::empty({num_tokens, topk}, f32_opts);
  auto sg_i = torch::empty({num_tokens, topk}, int32_opts);
  auto vllm_w = torch::empty({num_tokens, topk}, f32_opts);
  auto vllm_i = torch::empty({num_tokens, topk}, int32_opts);

  InvokeSglangRouteRadix<float>(
      static_cast<float*>(gating.data_ptr()),
      static_cast<const float*>(bias.data_ptr()),
      static_cast<float*>(sg_w.data_ptr()),
      static_cast<int32_t*>(sg_i.data_ptr()),
      num_tokens, num_experts, topk, true, 1.0f, true, stream);
  InvokeVllmSingleGroupTopk<float>(
      static_cast<float*>(gating.data_ptr()),
      static_cast<float*>(vllm_w.data_ptr()),
      static_cast<int32_t*>(vllm_i.data_ptr()),
      static_cast<float*>(bias.data_ptr()),
      num_tokens, num_experts, topk, true, 1.0, true, stream);
  cudaStreamSynchronize(stream);

  auto sg_sorted = std::get<0>(sg_i.sort(-1));
  auto vllm_sorted = std::get<0>(vllm_i.sort(-1));
  ASSERT_TRUE(torch::equal(sg_sorted, vllm_sorted))
      << "topk_ids mismatch (all-equal scores)";
  auto perm_sg = std::get<1>(sg_i.sort(-1));
  auto perm_vllm = std::get<1>(vllm_i.sort(-1));
  ASSERT_TRUE(torch::allclose(
      sg_w.gather(1, perm_sg), vllm_w.gather(1, perm_vllm),
      /*rtol=*/1e-5, /*atol=*/1e-5))
      << "topk_weights mismatch (all-equal scores)";
}
TEST_F(RouteRadixTestSuit, K3_896_16_NegativeBias) {
  // 负 bias: bias 全为负值，验证 sigmoid + bias 仍正确排名
  torch::manual_seed(42);
  torch::cuda::manual_seed(42);
  auto f32_opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
  auto int32_opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32);
  int num_tokens = 64, num_experts = 896, topk = 16;
  auto gating = torch::randn({num_tokens, num_experts}, f32_opts);
  auto bias = -torch::rand({num_experts}, f32_opts) * 2.0;  // negative bias in [-2, 0)

  auto sg_w = torch::empty({num_tokens, topk}, f32_opts);
  auto sg_i = torch::empty({num_tokens, topk}, int32_opts);
  auto vllm_w = torch::empty({num_tokens, topk}, f32_opts);
  auto vllm_i = torch::empty({num_tokens, topk}, int32_opts);

  InvokeSglangRouteRadix<float>(
      static_cast<float*>(gating.data_ptr()),
      static_cast<const float*>(bias.data_ptr()),
      static_cast<float*>(sg_w.data_ptr()),
      static_cast<int32_t*>(sg_i.data_ptr()),
      num_tokens, num_experts, topk, true, 1.0f, true, stream);
  InvokeVllmSingleGroupTopk<float>(
      static_cast<float*>(gating.data_ptr()),
      static_cast<float*>(vllm_w.data_ptr()),
      static_cast<int32_t*>(vllm_i.data_ptr()),
      static_cast<float*>(bias.data_ptr()),
      num_tokens, num_experts, topk, true, 1.0, true, stream);
  cudaStreamSynchronize(stream);

  auto sg_sorted = std::get<0>(sg_i.sort(-1));
  auto vllm_sorted = std::get<0>(vllm_i.sort(-1));
  ASSERT_TRUE(torch::equal(sg_sorted, vllm_sorted))
      << "topk_ids mismatch (negative bias)";
  auto perm_sg = std::get<1>(sg_i.sort(-1));
  auto perm_vllm = std::get<1>(vllm_i.sort(-1));
  ASSERT_TRUE(torch::allclose(
      sg_w.gather(1, perm_sg), vllm_w.gather(1, perm_vllm),
      /*rtol=*/1e-5, /*atol=*/1e-5))
      << "topk_weights mismatch (negative bias)";
}

TEST_F(RouteRadixTestSuit, K3_896_16_AllNanScores) {
  // 全 NaN scores: exercises the nan_floor code path (route_radix_utils.cuh).
  // sigmoid(NaN) = NaN → SGLang nan_floor maps NaN to -1e30f for selection;
  // all experts tie → both kernels select experts 0..15 (lowest-id tie-break).
  //
  // Weight behavior diverges by design (NOT a bug):
  //   SGLang: weights = sigmoid(NaN) = NaN — matches Triton router behavior.
  //           (nan_floor only sanitizes selection keys, not weight sources.)
  //   vLLM:   preprocess_score rejects non-finite input (selection_score =
  //           -INFINITY, unbiased_score = 0.0F); write_outputs then emits the
  //           renormalize fallback 1/topk for each slot.
  // We cross-validate expert IDs only; weights are checked per-kernel.
  torch::manual_seed(42);
  torch::cuda::manual_seed(42);
  auto f32_opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
  auto int32_opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32);
  int num_tokens = 4, num_experts = 896, topk = 16;
  auto gating = torch::full({num_tokens, num_experts}, NAN, f32_opts);
  auto bias = torch::zeros({num_experts}, f32_opts);

  auto sg_w = torch::empty({num_tokens, topk}, f32_opts);
  auto sg_i = torch::empty({num_tokens, topk}, int32_opts);
  auto vllm_w = torch::empty({num_tokens, topk}, f32_opts);
  auto vllm_i = torch::empty({num_tokens, topk}, int32_opts);

  InvokeSglangRouteRadix<float>(
      static_cast<float*>(gating.data_ptr()),
      static_cast<const float*>(bias.data_ptr()),
      static_cast<float*>(sg_w.data_ptr()),
      static_cast<int32_t*>(sg_i.data_ptr()),
      num_tokens, num_experts, topk, true, 1.0f, true, stream);
  InvokeVllmSingleGroupTopk<float>(
      static_cast<float*>(gating.data_ptr()),
      static_cast<float*>(vllm_w.data_ptr()),
      static_cast<int32_t*>(vllm_i.data_ptr()),
      static_cast<float*>(bias.data_ptr()),
      num_tokens, num_experts, topk, true, 1.0, true, stream);
  cudaStreamSynchronize(stream);

  // Both kernels should select experts 0..15 (all-equal scores → lowest-id
  // tie-break, enforced by both radix and warp-topk index encoding).
  auto expected_ids =
      torch::arange(static_cast<int64_t>(topk), int32_opts)
          .unsqueeze(0)
          .expand({num_tokens, topk});
  auto sg_sorted = std::get<0>(sg_i.sort(-1));
  auto vllm_sorted = std::get<0>(vllm_i.sort(-1));
  ASSERT_TRUE(torch::equal(sg_sorted, expected_ids))
      << "SGLang topk_ids should be 0..15 for all-NaN scores\nsg_i:\n" << sg_i;
  ASSERT_TRUE(torch::equal(vllm_sorted, expected_ids))
      << "vLLM topk_ids should be 0..15 for all-NaN scores\nvllm_i:\n" << vllm_i;

  // SGLang weights: sigmoid(NaN) = NaN. Use torch::isnan (not allclose, since
  // NaN != NaN returns false).
  ASSERT_TRUE(torch::isnan(sg_w).all().item<bool>())
      << "SGLang weights should all be NaN for all-NaN scores\nsg_w:\n" << sg_w;

  // vLLM weights: non-finite input sanitized to fallback 1/topk (renormalize).
  auto expected_vllm_w = torch::full({num_tokens, topk}, 1.0f / topk, f32_opts);
  ASSERT_TRUE(torch::allclose(vllm_w, expected_vllm_w))
      << "vLLM weights should be 1/topk for all-NaN scores (renormalize)\n"
      << "vllm_w:\n"
      << vllm_w;
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
