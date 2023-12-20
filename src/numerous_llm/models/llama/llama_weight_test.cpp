/* Copyright 2023 Tencent Inc.  All rights reserved.
 *
 * ==============================================================================*/

#include <cstdlib>
#include "numerous_llm/models/llama/llama_weight.h"
#include "test.h"
#include <thread>
#include <random>
#include <filesystem>

using namespace numerous_llm;
// 定义一个 LlamaWeightTest 类,继承自 testing::Test
class LlamaWeightTest : public testing::Test {
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
  NLLM_LOG_INFO << fmt::format("Hash Result = {}", hash_value);
  return hash_value;
}

// 生成测试用小模型
std::vector<uint16_t> get_random_data(int length) {
  std::vector<uint16_t> data(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0, 1.0);
  for (int i = 0; i < length; ++i) {
    float value = dis(gen);
    uint32_t f32 = *reinterpret_cast<uint32_t*>(&value);
    uint32_t f16 = ((f32 & 0x7fffffff) >> 13) - (0x38000000 >> 13);
    data[i] = static_cast<uint16_t>(f16 | (f32 & 0x80000000) >> 16);
  }
  return data;
}

void write_data_to_file(const std::string& saved_dir, const std::string& filename, const std::vector<uint16_t>& data) {
  std::ofstream file(saved_dir + "/" + filename, std::ios::binary);
  if (file) {
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    file.close();
  } else {
    NLLM_LOG_ERROR << fmt::format("Failed to open file: {}/{}", saved_dir, filename);
  }
}
void create_model(ModelConfig& model_config) {
  std::string saved_dir = model_config.path;
  int hidden_units = model_config.head_num * model_config.size_per_head;
  int inter_size = model_config.inter_size;
  int num_layer = model_config.num_layer;
  int vocab_size = model_config.vocab_size;
  int tensor_para_size = model_config.tensor_para_size;
  write_data_to_file(saved_dir, "model.wte.weight.bin", get_random_data(vocab_size * hidden_units));
  write_data_to_file(saved_dir, "model.lm_head.weight.bin", get_random_data(vocab_size * hidden_units));
  write_data_to_file(saved_dir, "model.final_layernorm.weight.bin", get_random_data(hidden_units));
  for (int layer = 0; layer < num_layer; ++layer) {
    write_data_to_file(saved_dir, "model.layers." + std::to_string(layer) + ".input_layernorm.weight.bin",
                       get_random_data(hidden_units));
    write_data_to_file(saved_dir, "model.layers." + std::to_string(layer) + ".post_attention_layernorm.weight.bin",
                       get_random_data(hidden_units));
    for (int rank = 0; rank < tensor_para_size; ++rank) {
      write_data_to_file(saved_dir, "model.layers." + std::to_string(layer) + ".attention.dense.weight."
                         + std::to_string(rank) + ".bin",
                         get_random_data(hidden_units / tensor_para_size * hidden_units));
      write_data_to_file(saved_dir, "model.layers." + std::to_string(layer)
                         + ".attention.query_key_value.weight." + std::to_string(rank) + ".bin",
                         get_random_data(3 * hidden_units / tensor_para_size * hidden_units));
      write_data_to_file(saved_dir, "model.layers." + std::to_string(layer) + ".mlp.gate_proj.weight."
                         + std::to_string(rank) + ".bin",
                         get_random_data(inter_size / tensor_para_size * hidden_units));
      write_data_to_file(saved_dir, "model.layers." + std::to_string(layer) + ".mlp.up_proj.weight."
                         + std::to_string(rank) + ".bin",
                         get_random_data(inter_size / tensor_para_size * hidden_units));
      write_data_to_file(saved_dir, "model.layers." + std::to_string(layer) + ".mlp.down_proj.weight."
                         + std::to_string(rank) + ".bin",
                         get_random_data(inter_size / tensor_para_size * hidden_units));
    }
  }
}

TEST_F(LlamaWeightTest, GetModelWeightsTest) {
  // 当环境中不包含该路径时, 下载该模型
  std::filesystem::path ft_path(model_config.path);
  if (!std::filesystem::exists(ft_path)) {
    NLLM_LOG_WARNING << fmt::format("The given model path {} does not exist. Generating a test model",
                                    model_config.path);
    std::filesystem::create_directories(model_config.path);
    create_model(model_config);
  }

  LlamaWeight llama_weight(model_config, 0);
  // 正确的 weight 名称
  std::string weight_name = "lm_head";
  Tensor lm_head = llama_weight.GetModelWeights(weight_name);
  EXPECT_EQ(lm_head.device, MEMORY_GPU);
  EXPECT_EQ(lm_head.storage, STORAGE_CONTIGUOUS);
  EXPECT_EQ(lm_head.shape, std::vector<size_t>({32, 16}));

  // 比较数据一致性
  //     llama_weight_hash: 使用 GetModelWeights 获取到的 BlockManager 中数据校验值
  //     local_file_hash:   直接读取本地文件,计算得到数据校验值
  Tensor gpu_tensor = llama_weight.GetModelWeights(weight_name);
  size_t data_size = gpu_tensor.GetTotalBytes() / sizeof(short);
  short* cpu_tensor = new short[data_size];
  cudaMemcpy(cpu_tensor, gpu_tensor.GetPtr<void>(), data_size * sizeof(short), cudaMemcpyDeviceToHost);
  size_t llama_weight_hash = get_hash_code(cpu_tensor, data_size);

  std::vector<short> local_tensor(data_size);
  std::ifstream file(model_config.path + "/model.lm_head.weight.bin", std::ios::binary);
  file.read(reinterpret_cast<char*>(local_tensor.data()), data_size * sizeof(short));
  size_t local_file_hash = get_hash_code(local_tensor.data(), data_size);
  EXPECT_EQ(llama_weight_hash, local_file_hash);

  // 错误的 weight 名称
  weight_name = "wrong_name";
  Tensor wrong_tensor = llama_weight.GetModelWeights(weight_name);
  EXPECT_EQ(wrong_tensor.device, MEMORY_CPU);
  EXPECT_TRUE(wrong_tensor.shape.empty());
}
