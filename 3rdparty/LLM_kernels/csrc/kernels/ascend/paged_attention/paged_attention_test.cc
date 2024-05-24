/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include <gtest/gtest.h>

#include <vector>

#include "csrc/kernels/ascend/paged_attention/paged_attention.h"
#include "csrc/utils/ascend/common.h"

#include "tests/kernels/ascend/utils/testsuit_base.h"

namespace llm_kernels {
namespace ascend {
namespace test {

static void* kv_list = nullptr;

static uint32_t head_size = 20;
static uint32_t kv_head_size = 20;
static uint32_t head_dim = 128;
static uint32_t layer_num = 40;
static uint32_t layer_idx = 0;
static uint32_t block_token_num = 16;

static int batch_size = 2;
static int total_block_num = 12;

static void CreateKVCache() {
  std::vector<int> batch_block_num{6, 6};
  std::vector<std::vector<void*>> batch_kv_caches(2);
  size_t block_size = layer_num * head_size * head_dim * block_token_num * 2 * sizeof(aclFloat16);
  for (size_t i = 0; i < batch_block_num.size(); ++i) {
    size_t block_num = batch_block_num[i];
    for (size_t j = 0; j < block_num; ++j) {
      void* kv_cache;
      ACL_CHECK_RET(aclrtMalloc(&kv_cache, block_size, ACL_MEM_MALLOC_HUGE_FIRST));
      batch_kv_caches[i].push_back(kv_cache);
    }
  }

  std::vector<void*> kv_list_host(layer_num * total_block_num * 2);
  for (size_t layer_idx = 0; layer_idx < layer_num; ++layer_idx) {
    int kv_list_index = 0;
    for (size_t idx = 0; idx < batch_size; ++idx) {
      size_t block_num = batch_kv_caches[idx].size();
      for (size_t block_idx = 0; block_idx < block_num; block_idx++) {
        void* kv_cache_ptr = batch_kv_caches[idx][block_idx];
        kv_cache_ptr += layer_idx * block_size / layer_num;
        kv_list_host[layer_idx * total_block_num * 2 + kv_list_index] = kv_cache_ptr;
        kv_list_index++;
      }
    }
    for (size_t idx = 0; idx < batch_size; ++idx) {
      size_t block_num = batch_kv_caches[idx].size();
      for (size_t block_idx = 0; block_idx < block_num; block_idx++) {
        void* kv_cache_ptr = batch_kv_caches[idx][block_idx];
        kv_cache_ptr += layer_idx * block_size / layer_num + block_size / layer_num / 2;
        kv_list_host[layer_idx * total_block_num * 2 + kv_list_index] = kv_cache_ptr;
        kv_list_index++;
      }
    }
  }

  size_t kv_size = (layer_num * total_block_num * 2) * sizeof(void*);
  ACL_CHECK_RET(aclrtMalloc(&kv_list, kv_size, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK_RET(aclrtMemcpy(kv_list, kv_list_host.size() * sizeof(void*), kv_list_host.data(),
                            kv_list_host.size() * sizeof(void*), ACL_MEMCPY_HOST_TO_DEVICE));
}

class AscendPagedAttentionTestSuit : public AscendTestSuitBase {
 public:
  void SetUp() override { AscendTestSuitBase::SetUp(); }

  void TearDown() override { AscendTestSuitBase::TearDown(); }

 protected:
  using AscendTestSuitBase::stream;
};

TEST_F(AscendPagedAttentionTestSuit, PrefillTest) {
  int total_token_num = 163;
  bool is_context_stage = true;

  uint32_t hidden_units = head_size * head_dim;

  PagedAttention<aclFloat16> paged_attention;
  paged_attention.Initialize(head_size, kv_head_size, head_dim, layer_num, layer_idx, block_token_num, stream);

  void* output;
  size_t output_size = total_token_num * hidden_units * sizeof(aclFloat16);
  ACL_CHECK_RET(aclrtMalloc(&output, output_size, ACL_MEM_MALLOC_HUGE_FIRST));

  void* qkv_tensor;
  size_t qkv_size = total_token_num * hidden_units * 3 * sizeof(aclFloat16);
  ACL_CHECK_RET(aclrtMalloc(&qkv_tensor, qkv_size, ACL_MEM_MALLOC_HUGE_FIRST));
  std::vector<size_t> qkv_shape;
  llm_kernels::utils::LoadNpyToPtr<aclFloat16>("/tmp/tests/kernels/data/paged_attention/prefill_qkv_tensor.npy",
                                               (aclFloat16*)qkv_tensor, qkv_shape, false);

  void* seq_offset;
  size_t seq_size = (batch_size + 1) * sizeof(uint64_t);
  ACL_CHECK_RET(aclrtMalloc(&seq_offset, seq_size, ACL_MEM_MALLOC_HUGE_FIRST));
  std::vector<size_t> seq_shape;
  llm_kernels::utils::LoadNpyToPtr<uint64_t>("/tmp/tests/kernels/data/paged_attention/prefill_seq_offset.npy",
                                             (uint64_t*)seq_offset, seq_shape, false);
  if (kv_list == nullptr) {
    CreateKVCache();
  }

  void* block_offset;
  size_t offset_size = (batch_size + 1) * sizeof(int32_t);
  ACL_CHECK_RET(aclrtMalloc(&block_offset, offset_size, ACL_MEM_MALLOC_HUGE_FIRST));
  std::vector<size_t> off_shape;
  llm_kernels::utils::LoadNpyToPtr<int32_t>("/tmp/tests/kernels/data/paged_attention/prefill_block_offset.npy",
                                            (int32_t*)block_offset, off_shape, false);

  void* rope_pos;
  size_t pos_size = (total_token_num) * sizeof(int64_t);
  ACL_CHECK_RET(aclrtMalloc(&rope_pos, pos_size, ACL_MEM_MALLOC_HUGE_FIRST));
  std::vector<size_t> pos_shape;
  llm_kernels::utils::LoadNpyToPtr<int64_t>("/tmp/tests/kernels/data/paged_attention/prefill_rope_pos.npy",
                                            (int64_t*)rope_pos, pos_shape, false);

  ACL_CHECK_RET(aclrtSynchronizeStream(stream));
  paged_attention.Forward(output, qkv_tensor, seq_offset, (void**)kv_list, block_offset, rope_pos, batch_size,
                          total_token_num, total_block_num, is_context_stage, stream);
  ACL_CHECK_RET(aclrtSynchronizeStream(stream));

  std::vector<aclFloat16> result_host(total_token_num * hidden_units, 0);
  ACL_CHECK_RET(aclrtMemcpy(result_host.data(), result_host.size() * sizeof(aclFloat16), output,
                            total_token_num * hidden_units * sizeof(aclFloat16), ACL_MEMCPY_DEVICE_TO_HOST));

  void* ref_result;
  size_t ref_size = (total_token_num * hidden_units) * sizeof(aclFloat16);
  ACL_CHECK_RET(aclrtMalloc(&ref_result, ref_size, ACL_MEM_MALLOC_HUGE_FIRST));
  std::vector<size_t> ref_shape;
  llm_kernels::utils::LoadNpyToPtr<aclFloat16>("/tmp/tests/kernels/data/paged_attention/prefill_attn_output.npy",
                                               (aclFloat16*)ref_result, ref_shape, false);

  std::vector<aclFloat16> ref_host(total_token_num * hidden_units, 0);
  ACL_CHECK_RET(aclrtMemcpy(ref_host.data(), ref_host.size() * sizeof(aclFloat16), output,
                            total_token_num * hidden_units * sizeof(aclFloat16), ACL_MEMCPY_DEVICE_TO_HOST));

  for (size_t i = 0; i < result_host.size(); ++i) {
    EXPECT_NEAR(aclFloat16ToFloat(result_host[i]), aclFloat16ToFloat(ref_host[i]), 1e-4);
  }
}

TEST_F(AscendPagedAttentionTestSuit, DecodeTest) {
  int total_token_num = 2;
  bool is_context_stage = false;

  uint32_t hidden_units = head_size * head_dim;

  PagedAttention<aclFloat16> paged_attention;
  paged_attention.Initialize(head_size, kv_head_size, head_dim, layer_num, layer_idx, block_token_num, stream);

  void* output;
  size_t output_size = batch_size * hidden_units * sizeof(aclFloat16);
  ACL_CHECK_RET(aclrtMalloc(&output, output_size, ACL_MEM_MALLOC_HUGE_FIRST));

  void* qkv_tensor;
  size_t qkv_size = batch_size * hidden_units * 3 * sizeof(aclFloat16);
  ACL_CHECK_RET(aclrtMalloc(&qkv_tensor, qkv_size, ACL_MEM_MALLOC_HUGE_FIRST));
  std::vector<size_t> qkv_shape;
  llm_kernels::utils::LoadNpyToPtr<aclFloat16>("/tmp/tests/kernels/data/paged_attention/decode_qkv_tensor.npy",
                                               (aclFloat16*)qkv_tensor, qkv_shape, false);

  void* seq_offset;
  size_t seq_size = batch_size * sizeof(int32_t);
  ACL_CHECK_RET(aclrtMalloc(&seq_offset, seq_size, ACL_MEM_MALLOC_HUGE_FIRST));
  std::vector<size_t> seq_shape;
  llm_kernels::utils::LoadNpyToPtr<int32_t>("/tmp/tests/kernels/data/paged_attention/decode_seq_offset.npy",
                                            (int32_t*)seq_offset, seq_shape, false);

  if (kv_list == nullptr) {
    CreateKVCache();
  }

  void* block_offset;
  size_t offset_size = (batch_size + 1) * sizeof(int32_t);
  ACL_CHECK_RET(aclrtMalloc(&block_offset, offset_size, ACL_MEM_MALLOC_HUGE_FIRST));
  std::vector<size_t> off_shape;
  llm_kernels::utils::LoadNpyToPtr<int32_t>("/tmp/tests/kernels/data/paged_attention/decode_block_offset.npy",
                                            (int32_t*)block_offset, off_shape, false);

  void* rope_pos;
  size_t pos_size = batch_size * sizeof(int64_t);
  ACL_CHECK_RET(aclrtMalloc(&rope_pos, pos_size, ACL_MEM_MALLOC_HUGE_FIRST));
  std::vector<size_t> pos_shape;
  llm_kernels::utils::LoadNpyToPtr<int64_t>("/tmp/tests/kernels/data/paged_attention/decode_rope_pos.npy",
                                            (int64_t*)rope_pos, pos_shape, false);

  ACL_CHECK_RET(aclrtSynchronizeStream(stream));
  paged_attention.Forward(output, qkv_tensor, seq_offset, (void**)kv_list, block_offset, rope_pos, batch_size,
                          total_token_num, total_block_num, is_context_stage, stream);
  ACL_CHECK_RET(aclrtSynchronizeStream(stream));

  std::vector<aclFloat16> result_host(batch_size * hidden_units, 0);
  ACL_CHECK_RET(aclrtMemcpy(result_host.data(), result_host.size() * sizeof(aclFloat16), output,
                            batch_size * hidden_units * sizeof(aclFloat16), ACL_MEMCPY_DEVICE_TO_HOST));

  void* ref_result;
  size_t ref_size = (batch_size * hidden_units) * sizeof(aclFloat16);
  ACL_CHECK_RET(aclrtMalloc(&ref_result, ref_size, ACL_MEM_MALLOC_HUGE_FIRST));
  std::vector<size_t> ref_shape;
  llm_kernels::utils::LoadNpyToPtr<aclFloat16>("/tmp/tests/kernels/data/paged_attention/decode_attn_output.npy",
                                               (aclFloat16*)ref_result, ref_shape, false);

  std::vector<aclFloat16> ref_host(batch_size * hidden_units, 0);
  ACL_CHECK_RET(aclrtMemcpy(ref_host.data(), ref_host.size() * sizeof(aclFloat16), output,
                            batch_size * hidden_units * sizeof(aclFloat16), ACL_MEMCPY_DEVICE_TO_HOST));

  for (size_t i = 0; i < result_host.size(); ++i) {
    EXPECT_NEAR(aclFloat16ToFloat(result_host[i]), aclFloat16ToFloat(ref_host[i]), 1e-4);
  }
}

}  // namespace test
}  // namespace ascend
}  // namespace llm_kernels
