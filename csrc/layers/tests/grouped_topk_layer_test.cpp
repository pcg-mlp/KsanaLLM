/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================

Performance comparison test for GroupedTopkLayer backends (K3 MoE).

Creates two GroupedTopkLayer instances — one with KSANA_MOE_GATE_BACKEND=sglang
(SGLang radix, specialized for 896/16) and one with KSANA_MOE_GATE_BACKEND=vllm
(vLLM single_group_topk, generic 128~1024) — and compares their topk_ids
correctness and latency through the Layer-level dispatch (Task 4).

The env var is read in GroupedTopkLayer::Init, so setenv MUST be called before
Init.  A fixed seed (42) ensures both layers receive identical inputs.

==============================================================================*/

#include <cuda_runtime.h>
#include <torch/torch.h>

#include <cstdlib>
#include <iostream>
#include <set>
#include <utility>
#include <vector>

#include "3rdparty/half/include/half.hpp"
#include "device/common_device.h"
#include "layers/grouped_topk_layer.h"
#include "layers/layer_test_base.h"

namespace ksana_llm {

class GroupedTopkLayerTest : public LayerTest {
 protected:
  // K3 routing config: 896 experts, topk 16, sigmoid + bias + renormalize.
  static constexpr int kNumExperts = 896;
  static constexpr int kTopk = 16;
  static constexpr int kNumTokens = 128;
  // num_expert_group / topk_group are only consumed by the legacy
  // InvokeGroupedTopk fallback; the SGLang/vLLM dispatch paths ignore them.
  // Set to DeepSeek-V3 convention values.
  static constexpr int kNumExpertGroup = 8;
  static constexpr int kTopkGroup = 4;
  static constexpr float kRoutedScalingFactor = 1.0f;
  static constexpr int kWarmupIters = 5;
  static constexpr int kBenchIters = 20;

  GroupedTopkLayerParameters MakeParams() {
    GroupedTopkLayerParameters p;
    p.topk = kTopk;
    p.renormalize = true;
    p.num_expert_group = kNumExpertGroup;
    p.topk_group = kTopkGroup;
    p.scoring_func = "sigmoid";
    p.routed_scaling_factor = kRoutedScalingFactor;
    p.use_e_score_correction_bias = true;
    p.score_threshold = 0.f;
    return p;
  }

  // Execute one Forward call.  The KsanaLLM Tensor wraps the torch tensor's
  // device memory via data_ptr (no copy).  Returns (topk_weights, topk_ids).
  std::pair<torch::Tensor, torch::Tensor> RunOnce(GroupedTopkLayer& layer, const torch::Tensor& gate,
                                                  const torch::Tensor& e_bias) {
    auto f32_opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    auto i32_opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32);

    auto topk_w = torch::empty({kNumTokens, kTopk}, f32_opts);
    auto topk_i = torch::empty({kNumTokens, kTopk}, i32_opts);

    Tensor gate_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
                       {static_cast<size_t>(gate.size(0)), static_cast<size_t>(gate.size(1))}, gate.data_ptr());
    Tensor bias_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {static_cast<size_t>(e_bias.size(0))},
                       e_bias.data_ptr());
    Tensor w_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32,
                    {static_cast<size_t>(topk_w.size(0)), static_cast<size_t>(topk_w.size(1))}, topk_w.data_ptr());
    Tensor i_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32,
                    {static_cast<size_t>(topk_i.size(0)), static_cast<size_t>(topk_i.size(1))}, topk_i.data_ptr());

    EXPECT_TRUE(layer.Forward(&gate_tensor, &w_tensor, &i_tensor, &bias_tensor).OK());
    cudaStreamSynchronize(context_->device->GetPrimaryComputeStream().Get());
    return {topk_w, topk_i};
  }

  // Warmup + timed bench iterations; returns average latency in milliseconds.
  double RunAndTime(GroupedTopkLayer& layer, const torch::Tensor& gate, const torch::Tensor& e_bias) {
    cudaStream_t stream = context_->device->GetPrimaryComputeStream().Get();

    for (int i = 0; i < kWarmupIters; ++i) {
      RunOnce(layer, gate, e_bias);
    }
    cudaStreamSynchronize(stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);
    for (int i = 0; i < kBenchIters; ++i) {
      RunOnce(layer, gate, e_bias);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return static_cast<double>(ms) / kBenchIters;
  }
};

// Migrated from moe_layer_test.cpp — basic GroupedTopkLayer correctness test.
// Tests softmax scoring with grouped topk, e_score_correction_bias, and
// profile mode. Uses FP16 and raw memcpy (original style preserved).
TEST_F(GroupedTopkLayerTest, BasicGroupedTopk) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;
  using dtype = half_float::half;

  // 测试参数
  const int num_tokens = 4;
  const int num_experts = 8;
  const int topk = 2;
  const bool renormalize = true;
  const int num_expert_group = 4;
  const int topk_group = 2;
  const std::string scoring_func = "softmax";
  const float routed_scaling_factor = 1.0f;
  const bool use_e_score_correction_bias = false;

  // 创建 GroupedTopkLayer
  GroupedTopkLayer grouped_topk_layer;

  // 测试初始化
  GroupedTopkLayerParameters params{.topk = topk,
                                    .renormalize = renormalize,
                                    .num_expert_group = num_expert_group,
                                    .topk_group = topk_group,
                                    .scoring_func = scoring_func,
                                    .routed_scaling_factor = routed_scaling_factor,
                                    .use_e_score_correction_bias = use_e_score_correction_bias,
                                    .score_threshold = 0.f};

  EXPECT_TRUE(grouped_topk_layer.Init({params}, runtime_config, context_, kDeviceRank).OK());

  // 准备输入张量 - gating_output
  Tensor gating_output;
  gating_output = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16,
                         {static_cast<size_t>(num_tokens), static_cast<size_t>(num_experts)});

  // 初始化 gating_output 数据 - 使用不同的值来测试 topk 选择
  std::vector<dtype> gating_host(num_tokens * num_experts);
  for (int token = 0; token < num_tokens; ++token) {
    for (int expert = 0; expert < num_experts; ++expert) {
      // 为每个 token 设置不同的专家权重，确保 topk 选择有意义
      gating_host[token * num_experts + expert] = static_cast<dtype>((expert + token * 0.1f + 1.0f) / num_experts);
    }
  }
  MemcpyAsync(gating_output.GetPtr<void>(), gating_host.data(), gating_output.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());

  // 准备输出张量
  Tensor topk_weights, topk_ids;
  topk_weights =
      Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {static_cast<size_t>(num_tokens), static_cast<size_t>(topk)});
  topk_ids =
      Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {static_cast<size_t>(num_tokens), static_cast<size_t>(topk)});

  StreamSynchronize(context_->device->GetMemoryManageStream());

  // 测试前向传播
  EXPECT_TRUE(grouped_topk_layer.Forward(&gating_output, &topk_weights, &topk_ids).OK());
  StreamSynchronize(context_->device->GetPrimaryComputeStream());

  // 验证输出
  std::vector<float> weights_host(num_tokens * topk);
  std::vector<int32_t> ids_host(num_tokens * topk);

  Memcpy(weights_host.data(), topk_weights.GetPtr<void>(), topk_weights.GetTotalBytes(), MEMCPY_DEVICE_TO_HOST);
  Memcpy(ids_host.data(), topk_ids.GetPtr<void>(), topk_ids.GetTotalBytes(), MEMCPY_DEVICE_TO_HOST);

  // 验证 topk_ids 在有效范围内
  for (int i = 0; i < num_tokens * topk; ++i) {
    EXPECT_GE(ids_host[i], 0);
    EXPECT_LT(ids_host[i], num_experts);
  }

  // 验证权重为正数（softmax 输出）
  for (int i = 0; i < num_tokens * topk; ++i) {
    EXPECT_GT(weights_host[i], 0.0f);
    EXPECT_LE(weights_host[i], 1.0f);
  }

  // 验证每个 token 的权重和接近 1.0（如果 renormalize=true）
  if (renormalize) {
    for (int token = 0; token < num_tokens; ++token) {
      float weight_sum = 0.0f;
      for (int k = 0; k < topk; ++k) {
        weight_sum += weights_host[token * topk + k];
      }
      EXPECT_NEAR(weight_sum, 1.0f, 0.01f);  // 允许小的数值误差
    }
  }

  // 验证 topk 选择的正确性 - 检查选中的专家确实是权重最大的
  for (int token = 0; token < num_tokens; ++token) {
    std::vector<std::pair<float, int>> expert_weights;
    for (int expert = 0; expert < num_experts; ++expert) {
      expert_weights.push_back({static_cast<float>(gating_host[token * num_experts + expert]), expert});
    }
    std::sort(expert_weights.rbegin(), expert_weights.rend());  // 降序排列

    // 检查选中的专家是否在前 topk 中
    std::set<int> expected_experts;
    for (int k = 0; k < topk; ++k) {
      expected_experts.insert(expert_weights[k].second);
    }

    std::set<int> actual_experts;
    for (int k = 0; k < topk; ++k) {
      actual_experts.insert(ids_host[token * topk + k]);
    }

    EXPECT_EQ(expected_experts, actual_experts);
  }

  // 测试带 e_bias 的情况
  GroupedTopkLayer grouped_topk_layer_with_bias;
  GroupedTopkLayerParameters params_with_bias{.topk = topk,
                                              .renormalize = renormalize,
                                              .num_expert_group = num_expert_group,
                                              .topk_group = topk_group,
                                              .scoring_func = scoring_func,
                                              .routed_scaling_factor = routed_scaling_factor,
                                              .use_e_score_correction_bias = true,
                                              .score_threshold = 0.f};

  EXPECT_TRUE(grouped_topk_layer_with_bias.Init({params_with_bias}, runtime_config, context_, kDeviceRank).OK());

  // 准备 e_bias 张量
  Tensor e_bias(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {static_cast<size_t>(num_experts)});
  std::vector<float> e_bias_host(num_experts);
  for (int i = 0; i < num_experts; ++i) {
    e_bias_host[i] = 0.1f * i;  // 简单的偏置值
  }
  MemcpyAsync(e_bias.GetPtr<void>(), e_bias_host.data(), e_bias.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());

  StreamSynchronize(context_->device->GetMemoryManageStream());

  // 测试带偏置的前向传播
  EXPECT_TRUE(grouped_topk_layer_with_bias.Forward(&gating_output, &topk_weights, &topk_ids, &e_bias).OK());
  StreamSynchronize(context_->device->GetPrimaryComputeStream());

  // 测试profile模式下输出固定的专家
  GroupedTopkLayer grouped_topk_layer_profile_mode;
  GroupedTopkLayerParameters params_profile{.topk = topk,
                                            .renormalize = renormalize,
                                            .num_expert_group = num_expert_group,
                                            .topk_group = topk_group,
                                            .scoring_func = scoring_func,
                                            .routed_scaling_factor = routed_scaling_factor,
                                            .use_e_score_correction_bias = true,
                                            .score_threshold = 0.f};
  runtime_config.is_profile_mode = true;
  EXPECT_TRUE(grouped_topk_layer_profile_mode.Init({params_profile}, runtime_config, context_, kDeviceRank).OK());
  EXPECT_TRUE(grouped_topk_layer_profile_mode.Forward(&gating_output, &topk_weights, &topk_ids).OK());
  Memcpy(ids_host.data(), topk_ids.GetPtr<void>(), topk_ids.GetTotalBytes(), MEMCPY_DEVICE_TO_HOST);

  // 验证 topk_ids 的值是固定的根据seed选出来的专家
  EXPECT_TRUE(ids_host[0] == 3 && ids_host[1] == 1 && ids_host[2] == 6 && ids_host[3] == 2);
  // set is_profile_mode back to false
  runtime_config.is_profile_mode = false;
#endif
}

TEST_F(GroupedTopkLayerTest, K3_BackendPerfCompare) {
#ifndef ENABLE_CUDA
  GTEST_SKIP();
#endif
  // Fixed seed — both backends receive byte-identical inputs.
  torch::manual_seed(42);
  torch::cuda::manual_seed(42);

  auto bf16_opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kBFloat16);
  auto f32_opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);

  auto gate = torch::randn({kNumTokens, kNumExperts}, bf16_opts);
  auto e_bias = torch::randn({kNumExperts}, f32_opts);

  // inter_data_type must match the gate tensor dtype (BF16) so that
  // DISPATCH_BY_3_DTYPE inside Forward instantiates ForwardT<bfloat16>.
  runtime_config.inter_data_type = TYPE_BF16;
  runtime_config.parallel_basic_config.expert_parallel_size = 1;
  runtime_config.parallel_basic_config.expert_world_size = 1;

  auto params = MakeParams();
  std::vector<std::any> init_params{std::any(params)};

  // --- SGLang backend (default; hits Path 1: radix for 896/16) ---
  setenv("KSANA_MOE_GATE_BACKEND", "sglang", 1);
  GroupedTopkLayer sglang_layer;
  ASSERT_TRUE(sglang_layer.Init(init_params, runtime_config, context_, 0).OK());

  // --- vLLM backend (hits Path 2: single_group_topk) ---
  setenv("KSANA_MOE_GATE_BACKEND", "vllm", 1);
  GroupedTopkLayer vllm_layer;
  ASSERT_TRUE(vllm_layer.Init(init_params, runtime_config, context_, 0).OK());

  // Clean up env to avoid side effects on downstream tests.
  unsetenv("KSANA_MOE_GATE_BACKEND");

  // --- Correctness: topk_ids must match (sorted by expert id) ---
  auto [sg_w, sg_i] = RunOnce(sglang_layer, gate, e_bias);
  auto [vllm_w, vllm_i] = RunOnce(vllm_layer, gate, e_bias);

  auto sg_sorted = std::get<0>(sg_i.sort(-1));
  auto vllm_sorted = std::get<0>(vllm_i.sort(-1));
  EXPECT_TRUE(torch::equal(sg_sorted, vllm_sorted)) << "topk_ids mismatch between sglang and vllm backends";

  // --- Performance ---
  double sg_ms = RunAndTime(sglang_layer, gate, e_bias);
  double vllm_ms = RunAndTime(vllm_layer, gate, e_bias);
  double speedup = (sg_ms > 0.0) ? (vllm_ms / sg_ms) : 0.0;

  std::cout << fmt::format("K3 GroupedTopkLayer backend perf (M={}, E={}, K={}):", kNumTokens, kNumExperts, kTopk)
            << std::endl;
  std::cout << fmt::format("  SGLang radix:            {:.4f} ms/iter", sg_ms) << std::endl;
  std::cout << fmt::format("  vLLM single_group_topk:  {:.4f} ms/iter", vllm_ms) << std::endl;
  std::cout << fmt::format("  Speedup (vLLM/SGLang):   {:.3f}x", speedup) << std::endl;
}

}  // namespace ksana_llm
