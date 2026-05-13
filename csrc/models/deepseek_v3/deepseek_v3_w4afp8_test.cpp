/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "models/deepseek_v3/deepseek_v3_test_base.h"

namespace ksana_llm {

TEST_F(DeepSeekV3Test, ForwardW4AFP8Test) {
#ifdef ENABLE_CUDA
  model_config.is_quant = true;
  model_config.weight_data_type = TYPE_BF16;
  runtime_config.inter_data_type = model_config.weight_data_type;
  QuantConfig gptq_quant_config;
  gptq_quant_config.method = QUANT_GPTQ;
  gptq_quant_config.pattern_layers = {".mlp.experts."};
  gptq_quant_config.ignored_layers = {"model.layers.4.mlp.experts."};
  QuantConfig fp8_quant_config;
  fp8_quant_config.method = QUANT_BLOCK_FP8_E4M3;
  fp8_quant_config.is_fp8_blockwise = true;
  fp8_quant_config.weight_block_size = {128, 128};
  fp8_quant_config.pattern_layers = {};
  fp8_quant_config.ignored_layers = {"model.layers.4.eh_proj"};
  fp8_quant_config.enable_moe_int4 = true;
  model_config.quant_config = fp8_quant_config;
  model_config.sub_quant_configs = {gptq_quant_config};
  std::cout << "Test W4AFP8 TYPE_BF16 weight_data_type forward." << std::endl;
  TestDeepSeekV3Forward<bfloat16>();
#endif
}

}  // namespace ksana_llm
