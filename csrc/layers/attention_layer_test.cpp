/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "layers/layer_test_base.h"

#include "layers/attention_layer.h"
#include "layers/flash_attention_layer.h"
#include "layers/flashinfer_resource_manager.h"
#include "layers/paged_attention_layer.h"

namespace ksana_llm {

TEST_F(LayerTest, AttentionLayerTest) {
#ifndef ENABLE_CUDA
  GTEST_SKIP();
#endif

#ifdef ENABLE_CUDA
  FlashAttentionLayer flash_attention_layer;
  QuantMode quant_mode = QUANT_NONE;
  int head_num = 32;
  int kv_head_num = 32;
  int size_per_head = 128;
  int rotary_embedding = 128;
  int max_position_embeddings = 2048;
  int max_batch_size = 1;
  size_t attn_temperature_tuning = 0;
  float attn_scale = 0;
  size_t floor_scale = 0;
  Tensor mrope_section_ptr;
  CreateHalfDataTypeTensor(mrope_section_ptr, {(size_t)rotary_embedding, (size_t)max_position_embeddings},
                           GetDataType<half>());
  bool enable_qk_pre_norm_before_rotary_pos = false;
  int stride_size = head_num * size_per_head;
  float k_scale = 1.0f;
  float v_scale = 1.0f;
  float rope_theta = 10000.0f;
  bool is_neox = true;
  uint32_t qk_nope_head_dim = 0;
  uint32_t v_head_dim = 0;
  uint32_t qk_rope_head_dim = 0;
  uint32_t q_lora_rank = 0;
  uint32_t kv_lora_rank = 0;
  Tensor cos_sin_cache_tensor;
  RoPEScalingFactor rope_scaling_factor;
  CreateHalfDataTypeTensor(cos_sin_cache_tensor, {(size_t)rotary_embedding, (size_t)max_position_embeddings},
                           GetDataType<half>());
  EXPECT_TRUE(flash_attention_layer
                  .Init({quant_mode,
                         static_cast<float>(0),
                         static_cast<bool>(false),
                         static_cast<int>(0),
                         static_cast<int>(1),
                         static_cast<int>(2048),
                         head_num,
                         kv_head_num,
                         size_per_head,
                         stride_size,
                         static_cast<size_t>(1),
                         TYPE_FP16,
                         k_scale,
                         v_scale,
                         rotary_embedding,
                         rope_theta,
                         v_head_dim,
                         qk_rope_head_dim,
                         qk_nope_head_dim,
                         q_lora_rank,
                         kv_lora_rank,
                         is_neox,
                         PositionEncoding::ROPE,
                         std::any(cos_sin_cache_tensor.GetPtr<void>()),
                         rope_scaling_factor,
                         max_batch_size,
                         attn_temperature_tuning,
                         attn_scale,
                         floor_scale,
                         true,
                         mrope_section_ptr,
                         enable_qk_pre_norm_before_rotary_pos},
                        runtime_config, context_, 0)
                  .OK());

  Tensor qkv, input_len, prefix_offsets, pos, mask, forward_shape, flag_tensor, flexible_rotary_embedding_pos,
      flexible_rotary_embedding_mask, dst_flexible_kv_cache_tensor, src_flexible_kv_cache_tensor,
      dst_flexible_token_idx_tensor, src_flexible_token_idx_tensor;
  std::vector<size_t> input_shape = {2, 12288};
  CreateHalfDataTypeTensor(qkv, input_shape, GetDataType<half>());
  CreateHalfDataTypeTensor(input_len, {2}, GetDataType<uint64_t>(), sizeof(uint64_t));
  CreateHalfDataTypeTensor(prefix_offsets, {2}, GetDataType<int>(), sizeof(int));
  CreateHalfDataTypeTensor(pos, {2}, GetDataType<uint64_t>(), /*dtype_size*/ sizeof(uint64_t));
  CreateHalfDataTypeTensor(mask, {2}, GetDataType<uint64_t>(), /*dtype_size*/ sizeof(uint64_t));

  flag_tensor = Tensor(MemoryLocation::LOCATION_HOST, TYPE_BOOL, {1});

  CreateHalfDataTypeTensor(flexible_rotary_embedding_pos, {0}, GetDataType<int>(), sizeof(int));
  CreateHalfDataTypeTensor(flexible_rotary_embedding_mask, {0}, GetDataType<int>(), sizeof(int));
  CreateHalfDataTypeTensor(dst_flexible_kv_cache_tensor, {0}, GetDataType<int>(), sizeof(int));
  CreateHalfDataTypeTensor(src_flexible_kv_cache_tensor, {0}, GetDataType<int>(), sizeof(int));
  CreateHalfDataTypeTensor(dst_flexible_token_idx_tensor, {0}, GetDataType<int>(), sizeof(int));
  CreateHalfDataTypeTensor(src_flexible_token_idx_tensor, {0}, GetDataType<int>(), sizeof(int));
  forward_shape.shape = {1, 2, 1, 0, 0, 0, 2, 1, 2, 0, 0, 0};
  void* pos_ptr = pos.GetPtr<void>();
  std::vector<uint64_t> pos_cpu({0, 1});
  Memcpy(pos_ptr, pos_cpu.data(), pos_cpu.size() * sizeof(uint64_t), MEMCPY_HOST_TO_DEVICE);
  void* mask_ptr = mask.GetPtr<void>();
  std::vector<uint64_t> mask_cpu({1, 1});
  Memcpy(mask_ptr, mask_cpu.data(), mask_cpu.size() * sizeof(uint64_t), MEMCPY_HOST_TO_DEVICE);
  void* input_len_ptr = input_len.GetPtr<void>();
  std::vector<uint64_t> input_len_cpu({0, 2});
  Memcpy(input_len_ptr, input_len_cpu.data(), input_len_cpu.size() * sizeof(uint64_t), MEMCPY_HOST_TO_DEVICE);
  Memset(prefix_offsets.GetPtr<void>(), 0, 2 * sizeof(int));
  flag_tensor.GetPtr<bool>()[0] = true;  // use_cache
  Tensor output_tensor;
  CreateHalfDataTypeTensor(output_tensor, input_shape, GetDataType<half>());
  std::vector<Tensor> output_tensors = {output_tensor};

  int block_size = runtime_config.attn_backend_config.block_size;
  std::vector<int> h_block_offsets = {0, 1};
  Tensor block_offsets;
  CreateHalfDataTypeTensor(block_offsets, {h_block_offsets.size()}, GetDataType<int>(), sizeof(int));
  Memcpy(block_offsets.GetPtr<void>(), h_block_offsets.data(), h_block_offsets.size() * sizeof(int),
         MEMCPY_HOST_TO_DEVICE);
  // 为 kv_list 分配内存并初始化
  Tensor kv_list;
  CreateHalfDataTypeTensor(kv_list, {static_cast<uint64_t>(h_block_offsets.back() * 20)}, GetDataType<uint64_t>());
  std::vector<void*> h_kv_list_ptrs(h_block_offsets.back() * 2);
  for (size_t i = 0; i < h_kv_list_ptrs.size(); i++) {
    Malloc(&h_kv_list_ptrs[i], block_size);
  }
  Memcpy(kv_list.GetPtr<void>(), h_kv_list_ptrs.data(), h_kv_list_ptrs.size() * sizeof(void*), MEMCPY_HOST_TO_DEVICE);

  // For blocke_prefill.
  Tensor layer_kv_cache_ptr_tensor;
  layer_kv_cache_ptr_tensor = Tensor(MemoryLocation::LOCATION_HOST, TYPE_INT64, {static_cast<uint64_t>(1 + 2)});
  int64_t* kv_cache_block_num = layer_kv_cache_ptr_tensor.GetPtr<int64_t>();
  *kv_cache_block_num = static_cast<uint64_t>(1);
  void** layer_kv_cache_ptr = layer_kv_cache_ptr_tensor.GetPtr<void*>() + 1;
  layer_kv_cache_ptr[0] = h_kv_list_ptrs[0];
  layer_kv_cache_ptr[1] = h_kv_list_ptrs[1];

  std::vector<int32_t> multi_token_request_block_table_host = {0};
  Tensor multi_token_request_block_table;
  multi_token_request_block_table =
      Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {static_cast<uint64_t>(1), static_cast<uint64_t>(1)});
  Memcpy(multi_token_request_block_table.GetPtr<void>(), multi_token_request_block_table_host.data(),
         multi_token_request_block_table_host.size() * sizeof(int32_t), MEMCPY_HOST_TO_DEVICE);

  Tensor empty_q_norm_weight;
  Tensor empty_k_norm_weight;
  EXPECT_TRUE(flash_attention_layer
                  .Forward(
                      {
                          qkv,
                          input_len,
                          kv_list,
                          prefix_offsets,
                          block_offsets,
                          pos,
                          mask,
                          flexible_rotary_embedding_pos,
                          flexible_rotary_embedding_mask,
                          dst_flexible_kv_cache_tensor,
                          src_flexible_kv_cache_tensor,
                          dst_flexible_token_idx_tensor,
                          src_flexible_token_idx_tensor,
                          prefix_offsets,
                          forward_shape,
                          empty_q_norm_weight,
                          empty_k_norm_weight,
                          flag_tensor,
                          layer_kv_cache_ptr_tensor,
                          multi_token_request_block_table,
                          input_len,
                      },
                      output_tensors)
                  .OK());
  PagedAttentionLayer attention_layer;
  EXPECT_TRUE(attention_layer
                  .Init({quant_mode,
                         static_cast<float>(0),
                         static_cast<bool>(false),
                         static_cast<int>(1),
                         static_cast<int>(2),
                         static_cast<int>(2048),
                         static_cast<int>(head_num),
                         kv_head_num,
                         static_cast<int>(size_per_head),
                         stride_size,
                         static_cast<size_t>(1),
                         TYPE_FP16,
                         k_scale,
                         v_scale,
                         rotary_embedding,
                         rope_theta,
                         static_cast<uint32_t>(0),
                         static_cast<uint32_t>(0),
                         static_cast<uint32_t>(0),
                         static_cast<uint32_t>(0),
                         static_cast<uint32_t>(0),
                         is_neox,
                         PositionEncoding::ROPE,
                         std::any(cos_sin_cache_tensor.GetPtr<void>()),
                         rope_scaling_factor,
                         max_batch_size,
                         attn_temperature_tuning,
                         attn_scale,
                         floor_scale,
                         false,
                         nullptr,
                         enable_qk_pre_norm_before_rotary_pos},
                        runtime_config, context_, 0)
                  .OK());
#endif
}

TEST_F(LayerTest, FlashInferResourceManagerTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank0 = 0;
  size_t num_heads = 32;
  size_t num_kv_heads = 4;
  size_t head_dim = 128;

  // Test 1: Get pinned host workspace for rank 0 (first time initialization)
  void* pinned_workspace_rank0 =
      FlashInferResourceManager::GetPinnedHostWorkspace(num_heads, num_kv_heads, head_dim, kDeviceRank0);
  EXPECT_NE(pinned_workspace_rank0, nullptr);
  KLLM_LOG_INFO << "Test 1 passed: Pinned host workspace initialized for rank 0";

  // Test 2: Get pinned host workspace again for rank 0 (should return same pointer)
  void* pinned_workspace_rank0_again =
      FlashInferResourceManager::GetPinnedHostWorkspace(num_heads, num_kv_heads, head_dim, kDeviceRank0);
  EXPECT_EQ(pinned_workspace_rank0, pinned_workspace_rank0_again);
  KLLM_LOG_INFO << "Test 2 passed: Pinned host workspace reused for rank 0";

  // Test 3: Get device workspace for rank 0
  std::shared_ptr<Tensor>& device_workspace_rank0 =
      FlashInferResourceManager::GetDeviceWorkspace(num_heads, num_kv_heads, head_dim, kDeviceRank0);
  EXPECT_NE(device_workspace_rank0, nullptr);
  EXPECT_GT(device_workspace_rank0->GetTotalBytes(), 0);
  EXPECT_EQ(device_workspace_rank0->location, MemoryLocation::LOCATION_DEVICE);
  EXPECT_EQ(device_workspace_rank0->dtype, DataType::TYPE_INT8);
  KLLM_LOG_INFO << "Test 3 passed: Device workspace initialized for rank 0, size: "
                << device_workspace_rank0->GetTotalBytes() << " bytes";

  // Test 4: Get device workspace again for rank 0 (should return same reference)
  std::shared_ptr<Tensor>& device_workspace_rank0_again =
      FlashInferResourceManager::GetDeviceWorkspace(num_heads, num_kv_heads, head_dim, kDeviceRank0);
  EXPECT_EQ(device_workspace_rank0.get(), device_workspace_rank0_again.get());
  KLLM_LOG_INFO << "Test 4 passed: Device workspace reused for rank 0";

  // Test 5: Get prefill helper before initialization (should return nullptr)
  std::shared_ptr<void> helper_before_init = FlashInferResourceManager::GetPrefillHelper(kDeviceRank0);
  EXPECT_EQ(helper_before_init, nullptr);
  KLLM_LOG_INFO << "Test 5 passed: GetPrefillHelper returns nullptr before initialization";

  // Test 6: Set prefill helper for rank 0
  auto mock_helper_rank0 = std::make_shared<int>(42);  // Mock helper object
  FlashInferResourceManager::SetPrefillHelper(kDeviceRank0, mock_helper_rank0);
  KLLM_LOG_INFO << "Test 6 passed: Prefill helper set for rank 0";

  // Test 7: Get prefill helper after initialization
  std::shared_ptr<void> helper_rank0_after = FlashInferResourceManager::GetPrefillHelper(kDeviceRank0);
  EXPECT_NE(helper_rank0_after, nullptr);
  EXPECT_EQ(helper_rank0_after, mock_helper_rank0);
  KLLM_LOG_INFO << "Test 7 passed: GetPrefillHelper returns correct helper after initialization";
  KLLM_LOG_INFO << "All FlashInferResourceManager tests passed successfully! ";
  FlashInferResourceManager::FreeAllResources();
#else
  GTEST_SKIP() << "FlashInferResourceManager requires CUDA support";
#endif
}

}  // namespace ksana_llm
