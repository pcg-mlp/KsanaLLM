/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/layers/activation_layer.h"
#include "numerous_llm/layers/add_layer.h"
#include "numerous_llm/layers/attention_layer.h"
#include "numerous_llm/layers/emb_lookup_layer.h"
#include "numerous_llm/layers/flash_attention_layer.h"
#include "numerous_llm/layers/paged_attention_layer.h"
#include "numerous_llm/layers/layernorm_layer.h"
#include "numerous_llm/layers/matmul_layer.h"
#include "numerous_llm/layers/nccl_all_reduce_sum_layer.h"
#include "numerous_llm/layers/paged_attention_layer.h"
#include "numerous_llm/layers/silu_mul_layer.h"
#include "numerous_llm/utils/dtypes.h"
#include "test.h"

namespace numerous_llm {

class LayerTest : public testing::Test {
 protected:
  // 在每个测试用例执行之前调用的函数
  void SetUp() override {
    // 创建一个 BlockManagerConfig 对象，用于配置 BlockManager
    BlockManagerConfig block_manager_config;
    block_manager_config.cpu_allocator_config.blocks_num = 2;
    block_manager_config.cpu_allocator_config.block_size = 1024;
    block_manager_config.cpu_allocator_config.device = MEMORY_CPU_PINNED;
    block_manager_config.device_allocator_config.blocks_num = 2;
    block_manager_config.device_allocator_config.block_size = 1024;
    block_manager_config.device_allocator_config.device = MEMORY_GPU;

    std::shared_ptr<Context> context = std::make_shared<Context>(2, 1);

    // 使用配置创建一个 BlockManager 对象
    block_manager = new BlockManager(block_manager_config, context);

    SetBlockManager(block_manager);
  }

  // 在每个测试用例执行之后调用的函数
  void TearDown() override {
    // 删除 BlockManager 对象
    delete block_manager;
  }

  Status CreateHalfDataTypeTensor(Tensor& tensor, const std::vector<size_t>& shape, const DataType data_type, size_t dtype_size = 2) {
    int idx;
    GetBlockManager()->SetDeviceId(0);
    size_t total_bytes =
        std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1), std::multiplies<size_t>()) * dtype_size;
    GetBlockManager()->AllocateContiguous(total_bytes, idx);
    tensor = Tensor(MEMORY_GPU, STORAGE_CONTIGUOUS, data_type, shape, std::vector<int>{idx});
    return Status();
  }

 protected:
  // 定义一个 BlockManager 指针，用于在测试用例中使用
  BlockManager* block_manager;
};

TEST_F(LayerTest, AttentionLayerTest) {
  std::shared_ptr<Context> context = std::make_shared<Context>(1, 1);
  FlashAttentionLayer flash_attention_layer;
  int head_num = 32;
  int kv_head_num = 32;
  int size_per_head = 128;
  int rotary_embedding = 128;
  float rope_theta = 10000.0f;
  bool is_neox = true;
  EXPECT_TRUE(flash_attention_layer.Init({int(0), int(2048), head_num, kv_head_num, size_per_head, rotary_embedding,
                                          rope_theta, is_neox}, context, 0).OK());

  Tensor qkv, input_len, pos, forward_shape;
  std::vector<size_t> input_shape = {2, 12288};
  CreateHalfDataTypeTensor(qkv, input_shape, GetTensorType<half>());
  CreateHalfDataTypeTensor(input_len, {1}, GetTensorType<int32_t>(), sizeof(int));
  CreateHalfDataTypeTensor(pos, {2}, GetTensorType<uint64_t>(), /*dtype_size*/sizeof(uint64_t));
  forward_shape.shape = {1, 2};
  void* pos_ptr = pos.GetPtr<void>();
  std::vector<uint64_t> pos_cpu({0, 2});
  CUDA_CHECK(cudaMemcpy(pos_ptr, pos_cpu.data(), 2 * sizeof(uint64_t), cudaMemcpyHostToDevice));
  void* input_len_ptr = input_len.GetPtr<void>();
  std::vector<int> input_len_cpu({2});
  CUDA_CHECK(cudaMemcpy(input_len_ptr, input_len_cpu.data(), sizeof(int), cudaMemcpyHostToDevice));
  Tensor output_tensor;
  CreateHalfDataTypeTensor(output_tensor, input_shape, GetTensorType<half>());
  std::vector<Tensor> output_tensors = {output_tensor};
  EXPECT_TRUE(flash_attention_layer.Forward({qkv, input_len, /*kv_list*/ Tensor(), /*kv_cache_buffer*/ Tensor(), pos,
                                             forward_shape}, output_tensors).OK());

  PagedAttentionLayer attention_layer;
  EXPECT_TRUE(
      attention_layer.Init({int(1), int(2048), static_cast<int>(head_num), kv_head_num,
                            static_cast<int>(size_per_head), rotary_embedding, rope_theta, is_neox}, context, 0).OK());
}

}  // namespace numerous_llm
