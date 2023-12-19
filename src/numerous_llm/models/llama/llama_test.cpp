/* Copyright 2023 Tencent Inc.  All rights reserved.
 *
 * ==============================================================================*/

#include "numerous_llm/models/llama/llama.h"
#include "numerous_llm/models/llama/create_test_model.h"
#include "test.h"

using namespace numerous_llm;
// 定义一个 LlamaTest 类,继承自 testing::Test
class LlamaTest : public testing::Test {
 protected:
  void SetUp() override {
    model_config.path = "/model2/llama-ft/13B/2-gpu/";
    model_config.weight_data_type = TYPE_FP16;
    model_config.head_num = 4;
    model_config.size_per_head = 4;
    model_config.inter_size = 14;
    model_config.num_layer = 2;
    model_config.vocab_size = 32;
    model_config.tensor_para_size = 2;
    model_config.layernorm_eps = 0.000987;
  }

 protected:
  ModelConfig model_config;
};

TEST_F(LlamaTest, ContextDecodeTest) {
  // 当环境中不包含该路径时, 下载该模型
  std::filesystem::path ft_path(model_config.path);
  if (!std::filesystem::exists(ft_path)) {
    NLLM_LOG_WARNING << fmt::format("The given model path {} does not exist. Generating a test model",
                                    model_config.path);
    std::filesystem::create_directories(model_config.path);
    create_model(model_config);
  }

  std::shared_ptr<LlamaWeight<half>> llama_weight = std::make_shared<LlamaWeight<half>>(model_config, 0);
  std::shared_ptr<Llama<half>> llama = std::make_shared<Llama<half>>(model_config, 0);
}
