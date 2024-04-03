/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#ifdef ENABLE_CUDA
#  include "flash_api.h"
#  include "ksana_llm/layers/activation_layer.h"
#  include "ksana_llm/layers/add_layer.h"
#  include "ksana_llm/layers/attention_layer.h"
#  include "ksana_llm/layers/emb_lookup_layer.h"
#  include "ksana_llm/layers/flash_attention_layer.h"
#  include "ksana_llm/layers/layernorm_layer.h"
#  include "ksana_llm/layers/matmul_layer.h"
#  include "ksana_llm/layers/nccl_all_reduce_sum_layer.h"
#  include "ksana_llm/layers/paged_attention_layer.h"
#  include "ksana_llm/layers/silu_mul_layer.h"
#  include "ksana_llm/utils/common_device.h"
#  include "ksana_llm/utils/dtypes.h"
#  include "test.h"

namespace ksana_llm {

class LayerTest : public testing::Test {
 protected:
  // 在每个测试用例执行之前调用的函数
  void SetUp() override {
    model_config.path = "/model/llama-hf/7B/";
    model_config.weight_data_type = TYPE_FP16;
    model_config.head_num = 32;
    model_config.size_per_head = 128;
    model_config.inter_size = 11008;
    model_config.num_layer = 32;
    model_config.vocab_size = 32000;
    model_config.tensor_para_size = 1;
    model_config.layernorm_eps = 1e-6;
    model_config.max_batch_size = 4;
    model_config.max_token_num = 1024;
    model_config.rotary_embedding = 128;
    model_config.max_position_embeddings = 2048;
    model_config.rope_theta = 10000.0f;
    model_config.num_key_value_heads = model_config.head_num;

    BlockManagerConfig block_manager_config;
    block_manager_config.host_allocator_config.blocks_num = 2;
    block_manager_config.host_allocator_config.block_token_num = 16;
    block_manager_config.host_allocator_config.block_size = block_manager_config.host_allocator_config.block_token_num *
                                                            2 * model_config.head_num * model_config.size_per_head *
                                                            model_config.num_layer * sizeof(half);
    block_manager_config.host_allocator_config.device = MEMORY_CPU_PINNED;
    block_manager_config.device_allocator_config.blocks_num = 2;
    block_manager_config.device_allocator_config.block_token_num = 16;
    block_manager_config.device_allocator_config.block_size =
        block_manager_config.host_allocator_config.block_token_num * 2 * model_config.head_num *
        model_config.size_per_head * model_config.num_layer * sizeof(half);
    NLLM_LOG_WARNING << fmt::format("block_size {}", block_manager_config.device_allocator_config.block_size);
    block_manager_config.device_allocator_config.device = MEMORY_GPU;

    context_ = std::make_shared<Context>(1, 1, MEMORY_GPU);

    // 使用配置创建一个 BlockManager 对象
    block_manager = new BlockManager(block_manager_config, context_);
    SetBlockManager(block_manager);
  }

  // 在每个测试用例执行之后调用的函数
  void TearDown() override {
    // 删除 BlockManager 对象
    delete block_manager;
  }

  Status CreateHalfDataTypeTensor(Tensor& tensor, const std::vector<size_t>& shape, const DataType data_type,
                                  size_t dtype_size = 2) {
    int idx;
    GetBlockManager()->SetDeviceId(0);
    size_t total_bytes =
        std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1), std::multiplies<size_t>()) * dtype_size;
    GetBlockManager()->AllocateContiguous(total_bytes, idx);
    tensor = Tensor(MEMORY_GPU, data_type, shape, idx);
    return Status();
  }

 protected:
  ModelConfig model_config;
  BlockManager* block_manager = nullptr;

  std::shared_ptr<Context> context_{nullptr};
};

TEST_F(LayerTest, AttentionLayerTest) {
  std::shared_ptr<Context> context = std::make_shared<Context>(1, 1, MEMORY_GPU);
  FlashAttentionLayer flash_attention_layer;
  int head_num = 32;
  int kv_head_num = 32;
  int size_per_head = 128;
  int rotary_embedding = 128;
  int max_position_embeddings = 2048;
  int stride_size = head_num * size_per_head;
  float rope_theta = 10000.0f;
  bool is_neox = true;
  Tensor cos_sin_cache_tensor;
  RoPEScalingFactor rope_scaling_factor;
  CreateHalfDataTypeTensor(cos_sin_cache_tensor, {rotary_embedding, max_position_embeddings}, GetDataType<half>());
  EXPECT_TRUE(flash_attention_layer
                  .Init({int(0), int(2048), head_num, kv_head_num, size_per_head, stride_size, rotary_embedding,
                         rope_theta, is_neox, std::any(cos_sin_cache_tensor.GetPtr<half>()), rope_scaling_factor},
                        context, 0)
                  .OK());

  Tensor qkv, input_len, pos, forward_shape;
  std::vector<size_t> input_shape = {2, 12288};
  CreateHalfDataTypeTensor(qkv, input_shape, GetDataType<half>());
  CreateHalfDataTypeTensor(input_len, {2}, GetDataType<uint64_t>(), sizeof(uint64_t));
  CreateHalfDataTypeTensor(pos, {2}, GetDataType<uint64_t>(), /*dtype_size*/ sizeof(uint64_t));
  forward_shape.shape = {1, 2, 1};
  void* pos_ptr = pos.GetPtr<void>();
  std::vector<uint64_t> pos_cpu({0, 1});
  Memcpy(pos_ptr, pos_cpu.data(), pos_cpu.size() * sizeof(uint64_t), MEMCPY_HOST_TO_DEVICE);
  void* input_len_ptr = input_len.GetPtr<void>();
  std::vector<uint64_t> input_len_cpu({0, 2});
  Memcpy(input_len_ptr, input_len_cpu.data(), input_len_cpu.size() * sizeof(uint64_t), MEMCPY_HOST_TO_DEVICE);
  Tensor output_tensor;
  CreateHalfDataTypeTensor(output_tensor, input_shape, GetDataType<half>());
  std::vector<Tensor> output_tensors = {output_tensor};

  int block_size = GetBlockManager()->GetBlockSize();
  std::vector<int> h_block_offset = {0, 1};
  Tensor block_offset;
  CreateHalfDataTypeTensor(block_offset, {h_block_offset.size()}, GetDataType<int>(), sizeof(int));
  Memcpy(block_offset.GetPtr<void>(), h_block_offset.data(), h_block_offset.size() * sizeof(int),
         MEMCPY_HOST_TO_DEVICE);
  // 为 kv_list 分配内存并初始化
  Tensor kv_list;
  CreateHalfDataTypeTensor(kv_list, {h_block_offset.back() * 20}, GetDataType<uint64_t>());
  std::vector<void*> h_kv_list_ptrs(h_block_offset.back() * 2);
  for (int i = 0; i < h_kv_list_ptrs.size(); i++) {
    Malloc(&h_kv_list_ptrs[i], block_size);
  }
  Memcpy(kv_list.GetPtr<void>(), h_kv_list_ptrs.data(), h_kv_list_ptrs.size() * sizeof(void*), MEMCPY_HOST_TO_DEVICE);

  EXPECT_TRUE(
      flash_attention_layer.Forward({qkv, input_len, kv_list, block_offset, pos, forward_shape}, output_tensors).OK());

  PagedAttentionLayer attention_layer;
  EXPECT_TRUE(attention_layer
                  .Init({int(1), int(2048), static_cast<int>(head_num), kv_head_num, static_cast<int>(size_per_head),
                         stride_size, rotary_embedding, rope_theta, is_neox,
                         std::any(cos_sin_cache_tensor.GetPtr<half>()), rope_scaling_factor},
                        context, 0)
                  .OK());
}

}  // namespace ksana_llm

#endif
