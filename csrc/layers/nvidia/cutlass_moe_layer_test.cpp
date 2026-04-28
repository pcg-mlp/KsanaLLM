/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <cuda_runtime.h>

#include "layers/cutlass_moe_layer.h"
#include "test.h"
#include "utils/memory_utils.h"

namespace ksana_llm {

class CutlassMoeLayerWorkspaceTest : public testing::Test {
 protected:
  void SetUp() override {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    sm_ = prop.major * 10 + prop.minor;
  }

  void SetupLayerMembers(CutlassMoeLayer& layer, bool low_latency_mode) {
    layer.max_token_num_ = kMaxTokenNum;
    layer.expert_topk_ = kExpertTopk;
    layer.expert_num_per_node_ = kExpertNumPerNode;
    layer.expert_hidden_size_ = kExpertHiddenSize;
    layer.expert_inter_size_ = kExpertInterSize;
    layer.use_low_latency_mode_ = low_latency_mode;
    layer.rank_ = 0;

    if (low_latency_mode) {
      layer.low_latency_max_tokens_ = kDeepEpMaxNumTokens;
      layer.global_expert_para_size_ = kGlobalExpertParaSize;
    }

    size_t ep_size = low_latency_mode ? kGlobalExpertParaSize : 1;
    layer.cutlass_moe_wrapper_ = std::make_shared<llm_kernels::nvidia::CutlassMoeWrapper>(
        1, 0, ep_size, 0, 1, 0, kExpertTopk, /*enable_alltoall=*/low_latency_mode);
    layer.cutlass_moe_wrapper_->Init<__nv_bfloat16>();
  }

  int sm_ = 0;

  static constexpr size_t kMaxTokenNum = 256;
  static constexpr size_t kExpertTopk = 4;
  static constexpr size_t kExpertNumPerNode = 8;
  static constexpr size_t kExpertHiddenSize = 256;
  static constexpr size_t kExpertInterSize = 512;
  static constexpr size_t kGlobalExpertParaSize = 2;
  static constexpr size_t kDeepEpMaxNumTokens = 256;
};

TEST_F(CutlassMoeLayerWorkspaceTest, NormalModeWorkspaceSize) {
  if (sm_ < 90) GTEST_SKIP() << "Requires SM90+";

  CutlassMoeLayer layer;
  SetupLayerMembers(layer, /*low_latency_mode=*/false);

  size_t ws = layer.GetWorkspaceSize();

  // Normal 模式: total = topk_weights + topk_ids + kernel_workspace
  EXPECT_EQ(ws, layer.topk_weights_ptr_size + layer.topk_ids_ptr_size + layer.kernel_workspace_size);
  EXPECT_GT(layer.kernel_workspace_size, size_t(0));

  // 非 DeepEP 模式下，LL 专用 buffer 应为 0
  EXPECT_EQ(layer.adapted_topk_weights_size_, size_t(0));
  EXPECT_EQ(layer.adapted_topk_ids_size_, size_t(0));
  EXPECT_EQ(layer.recv_expert_count_size_, size_t(0));
}

TEST_F(CutlassMoeLayerWorkspaceTest, LowLatencyModeWorkspaceSize) {
  if (sm_ < 90) GTEST_SKIP() << "Requires SM90+";

  CutlassMoeLayer layer;
  SetupLayerMembers(layer, /*low_latency_mode=*/true);

  size_t ws = layer.GetWorkspaceSize();

  // LL 模式: total = topk_weights + topk_ids + adapted_weights + adapted_ids + recv_count + kernel_workspace
  size_t expected_total = layer.topk_weights_ptr_size + layer.topk_ids_ptr_size + layer.adapted_topk_weights_size_ +
                          layer.adapted_topk_ids_size_ + layer.recv_expert_count_size_ + layer.kernel_workspace_size;
  EXPECT_EQ(ws, expected_total);
  EXPECT_GT(layer.kernel_workspace_size, size_t(0));
  EXPECT_GT(layer.adapted_topk_weights_size_, size_t(0));
  EXPECT_GT(layer.adapted_topk_ids_size_, size_t(0));
  EXPECT_GT(layer.recv_expert_count_size_, size_t(0));
}

}  // namespace ksana_llm
