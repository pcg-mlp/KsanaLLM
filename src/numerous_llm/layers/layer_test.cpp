/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/layers/activation_layer.h"
#include "numerous_llm/layers/add_layer.h"
#include "numerous_llm/layers/attention_layer.h"
#include "numerous_llm/layers/emb_lookup_layer.h"
#include "numerous_llm/layers/flash_attention_layer.h"
#include "numerous_llm/layers/layernorm_layer.h"
#include "numerous_llm/layers/matmul_layer.h"
#include "numerous_llm/layers/nccl_all_reduce_sum_layer.h"
#include "numerous_llm/layers/paged_attention_layer.h"
#include "numerous_llm/layers/rotary_embedding_layer.h"
#include "numerous_llm/layers/silu_mul_layer.h"
#include "test.h"

namespace numerous_llm {

TEST(AttentionLayerTest, AttentionLayer) {
  FlashAttentionLayer flash_attention_layer;
  EXPECT_TRUE(flash_attention_layer.Init({int(1),int(2048)}, nullptr).OK());
  std::vector<Tensor> input_tensors(2);
  std::vector<Tensor> output_tensors(3);
  EXPECT_TRUE(flash_attention_layer.Forward(input_tensors, output_tensors).OK());

  PagedAttentionLayer attention_layer;
  EXPECT_TRUE(attention_layer.Init({int(1), int(2048)}, nullptr).OK());
}

}  // namespace numerous_llm