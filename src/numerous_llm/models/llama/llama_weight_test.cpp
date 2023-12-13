/* Copyright 2023 Tencent Inc.  All rights reserved.
 *
 * ==============================================================================*/


#include "numerous_llm/models/llama/llama_weight.h"
#include "test.h"
#include <thread>

using namespace numerous_llm;
// 定义一个 LlamaWeightTest 类,继承自 testing::Test
class LlamaWeightTest : public testing::Test {
 protected:
  void SetUp() override {
    model_config.path = "/model/llama-ft/13B/2-gpu/";
    model_config.weight_data_type = TYPE_FP16;
    model_config.head_num = 40;
    model_config.size_per_head = 128;
    model_config.inter_size = 13824;
    model_config.num_layer = 40;
    model_config.rotary_embedding = 128;
    model_config.vocab_size = 32000;
    model_config.rank = 0;
    model_config.tensor_para_size = 2;
  }

 protected:
  ModelConfig model_config;
};

// 计算数据 hash 值
size_t get_hash_code(short* data, size_t data_size) {
  size_t delta = 0x9e3779b9;
  size_t hash_value = 0;
  size_t mod = 1e9 + 7;
  for (int i = 0; i < data_size; ++i) {
    hash_value ^= (hash_value % mod) + delta + (hash_value << 6) + (hash_value >>  2);
  }
  hash_value %= mod;
  printf("Hash Result = %d\n", hash_value);
  return hash_value;
}

TEST_F(LlamaWeightTest, GetModelWeightsTest) {
  LlamaWeight llama_weight(model_config);
  // 正确的 weight 名称
  std::string weight_name = "lm_head";
  Tensor lm_head = llama_weight.GetModelWeights(weight_name);
  EXPECT_EQ(lm_head.device, MEMORY_GPU);
  EXPECT_EQ(lm_head.storage, STORAGE_CONTIGUOUS);
  EXPECT_EQ(lm_head.shape, std::vector<size_t>({32000, 5120}));

  // 比较数据一致性
  Tensor gpu_tensor = llama_weight.GetModelWeights(weight_name);
  size_t data_size = gpu_tensor.GetTotalBytes() / sizeof(short);
  short* cpu_tensor = new short[data_size];
  cudaMemcpy(cpu_tensor, gpu_tensor.GetPtr<void>(), data_size * sizeof(short), cudaMemcpyDeviceToHost);
  EXPECT_EQ(get_hash_code(cpu_tensor, data_size), 256631325);

  /*
  printf("本地结果\n");
  std::vector<short> local_tensor(data_size);
  std::ifstream file("/model/llama-ft/13B/2-gpu/model.lm_head.weight.bin", std::ios::binary);
  file.read(reinterpret_cast<char*>(local_tensor.data()), data_size * sizeof(short));
  get_hash_code(local_tensor.data(), data_size);
  */

  // 错误的 weight 名称
  weight_name = "wrong_name";
  Tensor wrong_tensor = llama_weight.GetModelWeights(weight_name);
  EXPECT_EQ(wrong_tensor.device, MEMORY_CPU);
  EXPECT_TRUE(wrong_tensor.shape.empty());
}
