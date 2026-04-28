/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/moe_layer.h"
#include "test.h"

namespace ksana_llm {

class MoeLayerWorkspaceTest : public testing::Test {
 protected:
  void SetupLayerMembers(MoeLayer& layer, bool low_latency_mode, DataType compute_dtype) {
    // inter_data_type_ 用于 DISPATCH_BY_3_DTYPE 选择模板参数 T，始终用 BF16
    layer.inter_data_type_ = DataType::TYPE_BF16;
    layer.max_token_num_ = kMaxTokenNum;
    layer.expert_topk_ = kExpertTopk;
    layer.expert_num_per_node_ = kExpertNumPerNode;
    layer.expert_hidden_size_ = kExpertHiddenSize;
    layer.expert_inter_size_ = kExpertInterSize;
    layer.use_vllm_moe_ = true;
    layer.low_latency_mode_ = low_latency_mode;
    layer.using_deepep_ = low_latency_mode;
    layer.low_latency_max_tokens_ = low_latency_mode ? kLowLatencyMaxTokens : 0;
    layer.rank_ = 0;
    layer.tp_size_ = 1;
    layer.use_lora_ = false;
    layer.max_ws_bytes_ = 0;
    layer.compute_dtype_ = compute_dtype;
    layer.weight_dtype_ = compute_dtype;
    layer.block_shape_ = {128, 128};
    layer.global_expert_para_size_ = low_latency_mode ? kGlobalExpertParaSize : 1;
    layer.w4afp8_moe_backend_ = W4AFP8_MOE_BACKEND::Default;

    // 初始化所有 size 成员为 0
    layer.topk_weights_ptr_size = 0;
    layer.topk_ids_ptr_size = 0;
    layer.ll_output_size = 0;
    layer.expert_num_tokens_size = 0;
    layer.batched_intermediate_output_size = 0;
    layer.batched_silu_output_size = 0;
    layer.a1_q_size = 0;
    layer.a1_scale_size = 0;
    layer.a2_q_size = 0;
    layer.a2_scale_size = 0;
    layer.a1_and_a2_q_size = 0;
    layer.a1_and_a2_scale_size = 0;
    layer.dequant_workspace_size = 0;
    layer.max_fused_id_buffer_size = 0;
    layer.intermediate_cache1_size = 0;
    layer.intermediate_cache2_size = 0;
    layer.intermediate_cache3_size = 0;
    layer.intermediate_cache1_and_cache3_size = 0;
  }

  static constexpr size_t kMaxTokenNum = 256;
  static constexpr size_t kLowLatencyMaxTokens = 32;
  static constexpr size_t kExpertTopk = 4;
  static constexpr size_t kExpertNumPerNode = 8;
  static constexpr size_t kExpertHiddenSize = 256;
  static constexpr size_t kExpertInterSize = 512;
  static constexpr size_t kGlobalExpertParaSize = 2;
};

TEST_F(MoeLayerWorkspaceTest, LLModeWorkspaceSize_BF16) {
  MoeLayer layer;
  SetupLayerMembers(layer, /*low_latency_mode=*/true, DataType::TYPE_BF16);

  size_t ws = layer.GetWorkspaceSize();

  // LL 模式下基本 buffer 应该 > 0
  EXPECT_GT(layer.topk_weights_ptr_size, size_t(0));
  EXPECT_GT(layer.topk_ids_ptr_size, size_t(0));
  EXPECT_GT(layer.ll_output_size, size_t(0));
  EXPECT_GT(layer.expert_num_tokens_size, size_t(0));
  EXPECT_GT(layer.batched_intermediate_output_size, size_t(0));
  EXPECT_GT(layer.batched_silu_output_size, size_t(0));

  // BF16 不需要 FP8 量化 buffer
  EXPECT_EQ(layer.a1_q_size, size_t(0));
  EXPECT_EQ(layer.a1_scale_size, size_t(0));

  // workspace 应该等于所有 buffer 之和
  size_t expected_total = layer.topk_weights_ptr_size + layer.topk_ids_ptr_size + layer.ll_output_size +
                          layer.expert_num_tokens_size + layer.batched_intermediate_output_size +
                          layer.batched_silu_output_size;
  EXPECT_EQ(ws, expected_total);
}

TEST_F(MoeLayerWorkspaceTest, LLModeWorkspaceSize_FP8) {
  MoeLayer layer;
  SetupLayerMembers(layer, /*low_latency_mode=*/true, DataType::TYPE_BLOCK_FP8_E4M3);

  size_t ws = layer.GetWorkspaceSize();

  // LL 模式下基本 buffer 应该 > 0
  EXPECT_GT(layer.topk_weights_ptr_size, size_t(0));
  EXPECT_GT(layer.topk_ids_ptr_size, size_t(0));
  EXPECT_GT(layer.ll_output_size, size_t(0));
  EXPECT_GT(layer.expert_num_tokens_size, size_t(0));
  EXPECT_GT(layer.batched_intermediate_output_size, size_t(0));
  EXPECT_GT(layer.batched_silu_output_size, size_t(0));

  // FP8 模式需要 a1_q 和 a1_scale
  EXPECT_GT(layer.a1_q_size, size_t(0));
  EXPECT_GT(layer.a1_scale_size, size_t(0));

  // workspace 应该等于所有 buffer 之和（包括 FP8 buffer）
  size_t expected_total = layer.topk_weights_ptr_size + layer.topk_ids_ptr_size + layer.ll_output_size +
                          layer.expert_num_tokens_size + layer.batched_intermediate_output_size +
                          layer.batched_silu_output_size + layer.a1_q_size + layer.a1_scale_size;
  EXPECT_EQ(ws, expected_total);
}

TEST_F(MoeLayerWorkspaceTest, NormalModeWorkspaceSize_BF16) {
  MoeLayer layer;
  SetupLayerMembers(layer, /*low_latency_mode=*/false, DataType::TYPE_BF16);

  size_t ws = layer.GetWorkspaceSize();

  // Normal 模式下基本 buffer 应该 > 0
  EXPECT_GT(layer.topk_weights_ptr_size, size_t(0));
  EXPECT_GT(layer.topk_ids_ptr_size, size_t(0));
  EXPECT_GT(layer.max_fused_id_buffer_size, size_t(0));
  EXPECT_GT(layer.intermediate_cache1_and_cache3_size, size_t(0));

  // BF16 不需要 FP8 量化 buffer
  EXPECT_EQ(layer.a1_and_a2_q_size, size_t(0));
  EXPECT_EQ(layer.a1_and_a2_scale_size, size_t(0));

  // LL 专用 buffer 应为 0
  EXPECT_EQ(layer.ll_output_size, size_t(0));
  EXPECT_EQ(layer.expert_num_tokens_size, size_t(0));
}

TEST_F(MoeLayerWorkspaceTest, NormalModeWorkspaceSize_FP8) {
  MoeLayer layer;
  SetupLayerMembers(layer, /*low_latency_mode=*/false, DataType::TYPE_BLOCK_FP8_E4M3);

  size_t ws = layer.GetWorkspaceSize();

  // Normal FP8 模式: 需要 a1/a2 量化 buffer
  EXPECT_GT(layer.topk_weights_ptr_size, size_t(0));
  EXPECT_GT(layer.topk_ids_ptr_size, size_t(0));
  EXPECT_GT(layer.a1_and_a2_q_size, size_t(0));
  EXPECT_GT(layer.a1_and_a2_scale_size, size_t(0));

  // LL 专用 buffer 应为 0
  EXPECT_EQ(layer.ll_output_size, size_t(0));
  EXPECT_EQ(layer.expert_num_tokens_size, size_t(0));
}

}  // namespace ksana_llm
