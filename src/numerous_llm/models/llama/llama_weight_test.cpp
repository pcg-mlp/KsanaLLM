/* Copyright 2023 Tencent Inc.  All rights reserved.
 *
 * ==============================================================================*/

#include "numerous_llm/models/llama/llama_weight.h"
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <random>
#include <thread>
#include "numerous_llm/block_manager/block_manager.h"
#include "numerous_llm/models/llama/create_test_model.h"
#include "numerous_llm/utils/memory_utils.h"
#include "test.h"

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

    BlockManagerConfig block_manager_config;
    block_manager_config.cpu_allocator_config.blocks_num = 2;
    block_manager_config.cpu_allocator_config.block_size = 1024;
    block_manager_config.cpu_allocator_config.device = MEMORY_CPU_PINNED;
    block_manager_config.device_allocator_config.blocks_num = 2;
    block_manager_config.device_allocator_config.block_size = 1024;
    block_manager_config.device_allocator_config.device = MEMORY_GPU;

    context_ = std::make_shared<Context>(2, 1);

    // 使用配置创建一个 BlockManager 对象
    block_manager = new BlockManager(block_manager_config, context_);
    SetBlockManager(block_manager);
  }

  void TearDown() override { delete block_manager; }

 protected:
  ModelConfig model_config;
  BlockManager* block_manager = nullptr;

  std::shared_ptr<Context> context_{nullptr};
};

// 计算数据 hash 值
size_t get_hash_code(short* data, size_t data_size) {
  size_t delta = 0x9e3779b9;
  size_t hash_value = 0;
  size_t mod = 1e9 + 7;
  for (int i = 0; i < data_size; ++i) {
    hash_value ^= (hash_value % mod) + delta + (hash_value << 6) + (hash_value >> 2);
  }
  hash_value %= mod;
  NLLM_LOG_INFO << fmt::format("Hash Result = {}", hash_value);
  return hash_value;
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

  LlamaWeight<half> llama_weight(model_config, 0, context_);
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
