/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "models/deepseek_v3/deepseek_v3_test_base.h"

namespace ksana_llm {

TEST_F(DeepSeekV3Test, ForwardFP8BlockWiseTest) {
#ifdef ENABLE_CUDA
  model_config.is_quant = true;
  model_config.weight_data_type = TYPE_BF16;
  runtime_config.inter_data_type = model_config.weight_data_type;
  // deepseek only support fp8 block-wise quantization, don't support fp8 per-tensor quantization
  model_config.quant_config.method = QUANT_BLOCK_FP8_E4M3;
  std::cout << "Test FP8-BlockWise TYPE_BF16 weight_data_type forward." << std::endl;
  TestDeepSeekV3Forward<bfloat16>();
#endif
}

TEST_F(DeepSeekV3Test, ForwardGPTQInt4Test) {
#ifdef ENABLE_CUDA
  const char* skip_env = std::getenv("SKIP_TEST_IF_INVALID_TRITON_VERSION");
  if (skip_env != nullptr && std::string(skip_env) == "1") {
    GTEST_SKIP() << "Skipping test due to Triton version incompatibility (SKIP_TEST_IF_INVALID_TRITON_VERSION=1).";
  }
  model_config.is_quant = true;
  model_config.weight_data_type = TYPE_BF16;
  runtime_config.inter_data_type = model_config.weight_data_type;
  model_config.quant_config.method = QUANT_GPTQ;
  std::cout << "Test GPTQ-Quant TYPE_BF16 weight_data_type forward." << std::endl;
  TestDeepSeekV3Forward<bfloat16>();
#endif
}

TEST_F(DeepSeekV3Test, ForwardMoeInt4Test) {
#ifdef ENABLE_CUDA
  model_config.is_quant = true;
  model_config.weight_data_type = TYPE_BF16;
  runtime_config.inter_data_type = model_config.weight_data_type;
  model_config.quant_config.method = QUANT_BLOCK_FP8_E4M3;
  model_config.quant_config.enable_moe_int4 = true;
  std::cout << "Test MoeInt4 TYPE_BF16 weight_data_type forward." << std::endl;
  TestDeepSeekV3Forward<bfloat16>();
#endif
}

// Test for `num_experts_per_rank_ <= 224`
TEST_F(DeepSeekV3Test, SmallExpertsTest) {
  model_config.is_quant = true;
  model_config.weight_data_type = TYPE_BF16;
  runtime_config.inter_data_type = model_config.weight_data_type;
  model_config.quant_config.method = QUANT_BLOCK_FP8_E4M3;
  TestDeepSeekV3Forward<bfloat16>();
}

TEST_F(DeepSeekV3Test, EnableFullShardExpertTest) {
  model_config.is_quant = true;
  model_config.weight_data_type = TYPE_BF16;
  runtime_config.inter_data_type = model_config.weight_data_type;
  model_config.quant_config.method = QUANT_BLOCK_FP8_E4M3;
  runtime_config.enable_full_shared_expert = true;
  TestDeepSeekV3Forward<bfloat16>();
}

}  // namespace ksana_llm
