/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include <gtest/gtest.h>

#include <vector>

#include "csrc/kernels/ascend/attention/attention.h"
#include "csrc/utils/ascend/common.h"

#include "tests/kernels/ascend/utils/testsuit_base.h"

namespace llm_kernels {
namespace ascend {
namespace test {

static void* kv_list = nullptr;

// for atb test
static void* k_cache = nullptr;
static void* v_cache = nullptr;

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

template <typename DTYPE>
static void CreateATBKVCache() {
  size_t cache_size = total_block_num * block_token_num * head_size * head_dim * sizeof(DTYPE);
  ACL_CHECK_RET(aclrtMalloc(&k_cache, cache_size, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK_RET(aclrtMalloc(&v_cache, cache_size, ACL_MEM_MALLOC_HUGE_FIRST));
}

class AscendPagedAttentionTestSuit : public AscendTestSuitBase {
 public:
  void SetUp() override { AscendTestSuitBase::SetUp(); }

  void TearDown() override { AscendTestSuitBase::TearDown(); }
};

TEST_F(AscendPagedAttentionTestSuit, AscendCPagedAttentionMultiTokenForwardTest) {
  int total_token_num = 163;
  bool is_multi_token_forward = true;

  uint32_t hidden_units = head_size * head_dim;

  PagedAttention<aclFloat16> paged_attention;
  paged_attention.Initialize(head_size, kv_head_size, head_dim, layer_num, layer_idx, block_token_num, stream);

  void* output;
  size_t output_size = total_token_num * hidden_units * sizeof(aclFloat16);
  ACL_CHECK_RET(aclrtMalloc(&output, output_size, ACL_MEM_MALLOC_HUGE_FIRST));

  void* qkv_tensor;
  size_t qkv_size = total_token_num * hidden_units * 3 * sizeof(aclFloat16);
  ACL_CHECK_RET(aclrtMalloc(&qkv_tensor, qkv_size, ACL_MEM_MALLOC_HUGE_FIRST));
  std::vector<size_t> qkv_shape;  // shape [163, 7680]
  llm_kernels::utils::LoadNpyToPtr<aclFloat16>("/tmp/tests/kernels/data/paged_attention/prefill_qkv_tensor.npy",
                                               (aclFloat16*)qkv_tensor, qkv_shape, false);

  void* seq_offset;
  size_t seq_size = (batch_size + 1) * sizeof(uint64_t);
  ACL_CHECK_RET(aclrtMalloc(&seq_offset, seq_size, ACL_MEM_MALLOC_HUGE_FIRST));
  std::vector<size_t> seq_shape;  // shape [3,], content: [  0,  80, 163]
  llm_kernels::utils::LoadNpyToPtr<uint64_t>("/tmp/tests/kernels/data/paged_attention/prefill_seq_offset.npy",
                                             (uint64_t*)seq_offset, seq_shape, false);
  if (kv_list == nullptr) {
    CreateKVCache();
  }

  void* block_offset;
  size_t offset_size = (batch_size + 1) * sizeof(int32_t);
  ACL_CHECK_RET(aclrtMalloc(&block_offset, offset_size, ACL_MEM_MALLOC_HUGE_FIRST));
  std::vector<size_t> off_shape;  // shape [3, ], content: [ 0,  6, 12]
  llm_kernels::utils::LoadNpyToPtr<int32_t>("/tmp/tests/kernels/data/paged_attention/prefill_block_offset.npy",
                                            (int32_t*)block_offset, off_shape, false);

  void* rope_pos;
  size_t pos_size = (total_token_num) * sizeof(int64_t);
  ACL_CHECK_RET(aclrtMalloc(&rope_pos, pos_size, ACL_MEM_MALLOC_HUGE_FIRST));
  std::vector<size_t> pos_shape;  // shape [163, ], content: [0, 1,..., 79, 0, 1,.., 81, 82]
  llm_kernels::utils::LoadNpyToPtr<int64_t>("/tmp/tests/kernels/data/paged_attention/prefill_rope_pos.npy",
                                            (int64_t*)rope_pos, pos_shape, false);

  ACL_CHECK_RET(aclrtSynchronizeStream(stream));
  paged_attention.Forward(output, qkv_tensor, seq_offset, (void**)kv_list, block_offset, rope_pos, batch_size,
                          total_token_num, total_block_num, layer_idx, is_multi_token_forward, stream);
  ACL_CHECK_RET(aclrtSynchronizeStream(stream));

  std::vector<aclFloat16> result_host(total_token_num * hidden_units, 0);
  ACL_CHECK_RET(aclrtMemcpy(result_host.data(), result_host.size() * sizeof(aclFloat16), output,
                            total_token_num * hidden_units * sizeof(aclFloat16), ACL_MEMCPY_DEVICE_TO_HOST));

  void* ref_result;
  size_t ref_size = (total_token_num * hidden_units) * sizeof(aclFloat16);
  ACL_CHECK_RET(aclrtMalloc(&ref_result, ref_size, ACL_MEM_MALLOC_HUGE_FIRST));
  std::vector<size_t> ref_shape;  // shape [163, 2560]
  llm_kernels::utils::LoadNpyToPtr<aclFloat16>("/tmp/tests/kernels/data/paged_attention/prefill_attn_output.npy",
                                               (aclFloat16*)ref_result, ref_shape, false);

  std::vector<aclFloat16> ref_host(total_token_num * hidden_units, 0);
  ACL_CHECK_RET(aclrtMemcpy(ref_host.data(), ref_host.size() * sizeof(aclFloat16), output,
                            total_token_num * hidden_units * sizeof(aclFloat16), ACL_MEMCPY_DEVICE_TO_HOST));

  for (size_t i = 0; i < result_host.size(); ++i) {
    EXPECT_NEAR(aclFloat16ToFloat(result_host[i]), aclFloat16ToFloat(ref_host[i]), 1e-4);
  }
  ACL_CHECK_RET(aclrtFree(ref_result));
  ACL_CHECK_RET(aclrtFree(rope_pos));
  ACL_CHECK_RET(aclrtFree(block_offset));
  ACL_CHECK_RET(aclrtFree(seq_offset));
  ACL_CHECK_RET(aclrtFree(qkv_tensor));
  ACL_CHECK_RET(aclrtFree(output));
}

TEST_F(AscendPagedAttentionTestSuit, AscendCPagedAttentionSingleTokenForwardTest) {
  int total_token_num = 2;
  bool is_multi_token_forward = false;

  uint32_t hidden_units = head_size * head_dim;

  PagedAttention<aclFloat16> paged_attention;
  paged_attention.Initialize(head_size, kv_head_size, head_dim, layer_num, layer_idx, block_token_num, stream);

  void* output;
  size_t output_size = batch_size * hidden_units * sizeof(aclFloat16);
  ACL_CHECK_RET(aclrtMalloc(&output, output_size, ACL_MEM_MALLOC_HUGE_FIRST));

  void* qkv_tensor;
  size_t qkv_size = batch_size * hidden_units * 3 * sizeof(aclFloat16);
  ACL_CHECK_RET(aclrtMalloc(&qkv_tensor, qkv_size, ACL_MEM_MALLOC_HUGE_FIRST));
  std::vector<size_t> qkv_shape;  // shape: [2, 7680]
  llm_kernels::utils::LoadNpyToPtr<aclFloat16>("/tmp/tests/kernels/data/paged_attention/decode_qkv_tensor.npy",
                                               (aclFloat16*)qkv_tensor, qkv_shape, false);

  void* seq_offset;
  size_t seq_size = batch_size * sizeof(int32_t);
  ACL_CHECK_RET(aclrtMalloc(&seq_offset, seq_size, ACL_MEM_MALLOC_HUGE_FIRST));
  std::vector<size_t> seq_shape;  // shape: [2], content: [81, 84] ([80 + 1, 83 + 1])
  llm_kernels::utils::LoadNpyToPtr<int32_t>("/tmp/tests/kernels/data/paged_attention/decode_seq_offset.npy",
                                            (int32_t*)seq_offset, seq_shape, false);

  if (kv_list == nullptr) {
    CreateKVCache();
  }

  void* block_offset;
  size_t offset_size = (batch_size + 1) * sizeof(int32_t);
  ACL_CHECK_RET(aclrtMalloc(&block_offset, offset_size, ACL_MEM_MALLOC_HUGE_FIRST));
  std::vector<size_t> off_shape;  // shape: [3], content: [0, 6, 12]
  llm_kernels::utils::LoadNpyToPtr<int32_t>("/tmp/tests/kernels/data/paged_attention/decode_block_offset.npy",
                                            (int32_t*)block_offset, off_shape, false);

  void* rope_pos;
  size_t pos_size = batch_size * sizeof(int64_t);
  ACL_CHECK_RET(aclrtMalloc(&rope_pos, pos_size, ACL_MEM_MALLOC_HUGE_FIRST));
  std::vector<size_t> pos_shape;  // shape: [2], content: [80, 83]
  llm_kernels::utils::LoadNpyToPtr<int64_t>("/tmp/tests/kernels/data/paged_attention/decode_rope_pos.npy",
                                            (int64_t*)rope_pos, pos_shape, false);

  ACL_CHECK_RET(aclrtSynchronizeStream(stream));
  paged_attention.Forward(output, qkv_tensor, seq_offset, (void**)kv_list, block_offset, rope_pos, batch_size,
                          total_token_num, total_block_num, layer_idx, is_multi_token_forward, stream);
  ACL_CHECK_RET(aclrtSynchronizeStream(stream));

  std::vector<aclFloat16> result_host(batch_size * hidden_units, 0);
  ACL_CHECK_RET(aclrtMemcpy(result_host.data(), result_host.size() * sizeof(aclFloat16), output,
                            batch_size * hidden_units * sizeof(aclFloat16), ACL_MEMCPY_DEVICE_TO_HOST));

  void* ref_result;
  size_t ref_size = (batch_size * hidden_units) * sizeof(aclFloat16);
  ACL_CHECK_RET(aclrtMalloc(&ref_result, ref_size, ACL_MEM_MALLOC_HUGE_FIRST));
  std::vector<size_t> ref_shape;  // shape: [2, 2560]
  llm_kernels::utils::LoadNpyToPtr<aclFloat16>("/tmp/tests/kernels/data/paged_attention/decode_attn_output.npy",
                                               (aclFloat16*)ref_result, ref_shape, false);

  std::vector<aclFloat16> ref_host(batch_size * hidden_units, 0);
  ACL_CHECK_RET(aclrtMemcpy(ref_host.data(), ref_host.size() * sizeof(aclFloat16), output,
                            batch_size * hidden_units * sizeof(aclFloat16), ACL_MEMCPY_DEVICE_TO_HOST));

  for (size_t i = 0; i < result_host.size(); ++i) {
    EXPECT_NEAR(aclFloat16ToFloat(result_host[i]), aclFloat16ToFloat(ref_host[i]), 1e-4);
  }
  ACL_CHECK_RET(aclrtFree(ref_result));
  ACL_CHECK_RET(aclrtFree(rope_pos));
  ACL_CHECK_RET(aclrtFree(block_offset));
  ACL_CHECK_RET(aclrtFree(seq_offset));
  ACL_CHECK_RET(aclrtFree(qkv_tensor));
  ACL_CHECK_RET(aclrtFree(output));
}

TEST_F(AscendPagedAttentionTestSuit, ATBPagedAttentionMultiTokenForwardTest) {
  constexpr int total_token_num = 163;
  constexpr bool is_multi_token_forward = true;
  constexpr size_t max_position_embeddings = 2048;
  constexpr float rope_base = 10000.0f;
  uint32_t hidden_units = head_size * head_dim;
  uint32_t max_batch_size = 2;
  ATBAttention<aclFloat16> atb_paged_attention;
  atb_paged_attention.Initialize(max_batch_size, head_size, kv_head_size, head_dim, layer_num, layer_idx,
                                 block_token_num, stream, default_device, is_multi_token_forward,
                                 max_position_embeddings, rope_base);
  void* output;
  size_t output_size = total_token_num * hidden_units * sizeof(aclFloat16);
  ACL_CHECK_RET(aclrtMalloc(&output, output_size, ACL_MEM_MALLOC_HUGE_FIRST));
  void* qkv_tensor;
  size_t qkv_size = total_token_num * hidden_units * 3 * sizeof(aclFloat16);
  ACL_CHECK_RET(aclrtMalloc(&qkv_tensor, qkv_size, ACL_MEM_MALLOC_HUGE_FIRST));
  std::vector<size_t> qkv_shape;  // shape [163, 7680]
  llm_kernels::utils::LoadNpyToPtr<aclFloat16>("/tmp/tests/kernels/data/paged_attention/prefill_qkv_tensor.npy",
                                               (aclFloat16*)qkv_tensor, qkv_shape, false);
  std::vector<int32_t> seq_len_host = {80, 83};
  std::vector<int32_t> batch_block_num{6, 6};
  void* slot_mapping;
  std::vector<int32_t> slot_mapping_host(total_token_num, 0);
  size_t slot_mapping_size = total_token_num * sizeof(int32_t);
  ACL_CHECK_RET(aclrtMalloc(&slot_mapping, slot_mapping_size, ACL_MEM_MALLOC_HUGE_FIRST));
  // NOTE(karlluo): the struct of kcache and vcache, kcache and vcache is isolate.
  // there is 12 blocks each query owns 6 blocks
  // each block has 16 tokens, named variable: block_token_num
  // query_0 owns blocks [0, 1,..., 5]
  // query_1 owns blocks [6, 7,..., 11]
  // query_0 has 80 tokens
  // query_1 has 83 tokens
  // query_0's token_0 store in block_0's token_0 position, so slot_mapping[0] is 0
  // query_0's token_79 store in block_4's token_15 position, so slot_mapping[79] is 79
  // the formula of slot mapping value = block_idx * 16(also named block_token_num) + position_idx
  // thereforce
  // query_1's token_0 store in block_6's token_0 position, slot_mapping[80] is 6 * 16 + 0 = 96
  // query_1's token_44 store in block_8's token_12 position, slot_mapping[124] is 8 * 16 + 12 = 140
  // query_1's token_82 store in block_11's token_2 position, slot_mapping[162] is 11 * 16 + 2 = 178
  int32_t slotmapping_offset = 0;
  int32_t block_idx_offset = 0;
  for (size_t query_idx = 0; query_idx < batch_size; ++query_idx) {
    int32_t query_token_num = seq_len_host[query_idx];
    int32_t query_block_num = batch_block_num[query_idx];
    for (int32_t token_idx = 0; token_idx < query_token_num; ++token_idx) {
      int32_t slotmapping_idx = slotmapping_offset + token_idx;
      int32_t block_idx = int32_t(token_idx / block_token_num) + block_idx_offset;
      int32_t position_idx = token_idx % block_token_num;
      slot_mapping_host[slotmapping_idx] = block_idx * block_token_num + position_idx;
    }
    block_idx_offset += query_block_num;
    slotmapping_offset += query_token_num;
  }
  ACL_CHECK_RET(aclrtMemcpy(slot_mapping, slot_mapping_host.size() * sizeof(int32_t), slot_mapping_host.data(),
                            slot_mapping_host.size() * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE));

  if (k_cache == nullptr || v_cache == nullptr) {
    CreateATBKVCache<aclFloat16>();
  }

  void* rope_pos;
  size_t pos_size = (total_token_num) * sizeof(int64_t);
  ACL_CHECK_RET(aclrtMalloc(&rope_pos, pos_size, ACL_MEM_MALLOC_HUGE_FIRST));
  std::vector<size_t> pos_shape;  // shape [163, ], content: [0, 1,..., 79, 0, 1,.., 81, 82]
  llm_kernels::utils::LoadNpyToPtr<int64_t>("/tmp/tests/kernels/data/paged_attention/prefill_rope_pos.npy",
                                            (int64_t*)rope_pos, pos_shape, false);

  ACL_CHECK_RET(aclrtSynchronizeStream(stream));
  atb_paged_attention.Forward(output, qkv_tensor, rope_pos, slot_mapping, k_cache, v_cache, /*block_tables*/ nullptr,
                              /*max_num_blocks_per_query*/ 0, static_cast<uint32_t>(batch_size),
                              static_cast<uint32_t>(total_token_num), static_cast<uint32_t>(total_block_num),
                              block_token_num, static_cast<uint32_t>(layer_idx), seq_len_host.data(),
                              is_multi_token_forward, atb_context, llm_kernels::utils::GetTestWorkSpaceFunc);
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

  ACL_CHECK_RET(aclrtFree(rope_pos));
  ACL_CHECK_RET(aclrtFree(ref_result));
  ACL_CHECK_RET(aclrtFree(slot_mapping));
  ACL_CHECK_RET(aclrtFree(qkv_tensor));
  ACL_CHECK_RET(aclrtFree(output));
}

TEST_F(AscendPagedAttentionTestSuit, ATBPagedAttentionSingleTokenForwardTest) {
  int total_token_num = 2;
  bool is_multi_token_forward = false;
  constexpr size_t max_position_embeddings = 2048;
  constexpr float rope_base = 10000.0f;
  uint32_t hidden_units = head_size * head_dim;
  uint32_t max_batch_size = 2;

  ATBAttention<aclFloat16> atb_paged_attention;
  atb_paged_attention.Initialize(max_batch_size, head_size, kv_head_size, head_dim, layer_num, layer_idx,
                                 block_token_num, stream, default_device, is_multi_token_forward,
                                 max_position_embeddings, rope_base);

  void* output;
  size_t output_size = batch_size * hidden_units * sizeof(aclFloat16);
  ACL_CHECK_RET(aclrtMalloc(&output, output_size, ACL_MEM_MALLOC_HUGE_FIRST));

  void* qkv_tensor;
  size_t qkv_size = batch_size * hidden_units * 3 * sizeof(aclFloat16);
  ACL_CHECK_RET(aclrtMalloc(&qkv_tensor, qkv_size, ACL_MEM_MALLOC_HUGE_FIRST));
  std::vector<size_t> qkv_shape;  // shape: [2, 7680]
  llm_kernels::utils::LoadNpyToPtr<aclFloat16>("/tmp/tests/kernels/data/paged_attention/decode_qkv_tensor.npy",
                                               (aclFloat16*)qkv_tensor, qkv_shape, false);

  size_t seq_len_size = (batch_size) * sizeof(int32_t);
  std::vector<int32_t> seq_len_host = {81, 84};
  std::vector<int32_t> batch_block_num{6, 6};
  void* slot_mapping;
  std::vector<int32_t> slot_mapping_host(total_token_num, 0);
  size_t slot_mapping_size = total_token_num * sizeof(int32_t);
  ACL_CHECK_RET(aclrtMalloc(&slot_mapping, slot_mapping_size, ACL_MEM_MALLOC_HUGE_FIRST));
  // NOTE(karlluo): the last slot mapping of query_0 is 79
  slot_mapping_host[0] = 80;
  // NOTE(karlluo): the last slot mapping of query_0 is 79
  slot_mapping_host[1] = 179;
  ACL_CHECK_RET(aclrtMemcpy(slot_mapping, slot_mapping_host.size() * sizeof(int32_t), slot_mapping_host.data(),
                            slot_mapping_host.size() * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE));

  if (k_cache == nullptr || v_cache == nullptr) {
    CreateATBKVCache<aclFloat16>();
  }
  void* rope_pos;
  size_t pos_size = batch_size * sizeof(int64_t);
  ACL_CHECK_RET(aclrtMalloc(&rope_pos, pos_size, ACL_MEM_MALLOC_HUGE_FIRST));
  std::vector<size_t> pos_shape;  // shape: [2], content: [80, 83]
  llm_kernels::utils::LoadNpyToPtr<int64_t>("/tmp/tests/kernels/data/paged_attention/decode_rope_pos.npy",
                                            (int64_t*)rope_pos, pos_shape, false);

  // NOTE(karlluo): block_tables is a tensor help paged attention to find the real block
  // batch size: 2
  // query_0 using block: [0, 1, 2, 3, 4]
  // query_1 using block: [6, 7, 8, 9, 10, 11]
  // the block_tables is
  // [[0, 1, 2, 3, 4, 5],
  //  [6, 7, 8, 9, 10, 11]]
  // shape [2, 6], [batch_size, max_num_blocks_per_query]
  // in the meanwhile, seqlen is [81, 84] each block has 16 tokens.
  // it tells kernel that
  // query_0 using ceil(81 / 16) = 6 blocks
  // query_1 using ceil(84 / 16) = 6 blocks
  // kernel will find k/v cache in this 6 blocks.
  uint32_t max_num_blocks_per_query = *(std::max_element(batch_block_num.begin(), batch_block_num.end()));
  std::vector<int32_t> block_tables_host(batch_size * max_num_blocks_per_query, -1);
  int32_t block_idx_offset = 0;
  for (size_t query_idx = 0; query_idx < batch_size; ++query_idx) {
    int32_t query_block_num = batch_block_num[query_idx];
    for (int32_t block_idx = 0; block_idx < query_block_num; ++block_idx) {
      block_tables_host[block_idx + block_idx_offset] = block_idx;
    }
    block_idx_offset += query_block_num;
  }
  void* block_tables_device;
  ACL_CHECK_RET(aclrtMalloc(&block_tables_device, batch_size * max_num_blocks_per_query * sizeof(int32_t),
                            ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK_RET(aclrtMemcpy(block_tables_device, batch_size * max_num_blocks_per_query * sizeof(int32_t),
                            block_tables_host.data(), batch_size * max_num_blocks_per_query * sizeof(int32_t),
                            ACL_MEMCPY_HOST_TO_DEVICE));

  ACL_CHECK_RET(aclrtSynchronizeStream(stream));
  atb_paged_attention.Forward(output, qkv_tensor, rope_pos, slot_mapping, k_cache, v_cache, block_tables_device,
                              max_num_blocks_per_query, static_cast<uint32_t>(batch_size),
                              static_cast<uint32_t>(total_token_num), static_cast<uint32_t>(total_block_num),
                              block_token_num, static_cast<uint32_t>(layer_idx), seq_len_host.data(),
                              is_multi_token_forward, atb_context, llm_kernels::utils::GetTestWorkSpaceFunc);
  ACL_CHECK_RET(aclrtSynchronizeStream(stream));

  std::vector<aclFloat16> result_host(batch_size * hidden_units, 0);
  ACL_CHECK_RET(aclrtMemcpy(result_host.data(), result_host.size() * sizeof(aclFloat16), output,
                            batch_size * hidden_units * sizeof(aclFloat16), ACL_MEMCPY_DEVICE_TO_HOST));

  void* ref_result;
  size_t ref_size = (batch_size * hidden_units) * sizeof(aclFloat16);
  ACL_CHECK_RET(aclrtMalloc(&ref_result, ref_size, ACL_MEM_MALLOC_HUGE_FIRST));
  std::vector<size_t> ref_shape;  // shape: [2, 2560]
  llm_kernels::utils::LoadNpyToPtr<aclFloat16>("/tmp/tests/kernels/data/paged_attention/decode_attn_output.npy",
                                               (aclFloat16*)ref_result, ref_shape, false);

  std::vector<aclFloat16> ref_host(batch_size * hidden_units, 0);
  ACL_CHECK_RET(aclrtMemcpy(ref_host.data(), ref_host.size() * sizeof(aclFloat16), output,
                            batch_size * hidden_units * sizeof(aclFloat16), ACL_MEMCPY_DEVICE_TO_HOST));

  for (size_t i = 0; i < result_host.size(); ++i) {
    EXPECT_NEAR(aclFloat16ToFloat(result_host[i]), aclFloat16ToFloat(ref_host[i]), 1e-4);
  }

  ACL_CHECK_RET(aclrtFree(rope_pos));
  ACL_CHECK_RET(aclrtFree(ref_result));
  ACL_CHECK_RET(aclrtFree(block_tables_device));
  ACL_CHECK_RET(aclrtFree(slot_mapping));
  ACL_CHECK_RET(aclrtFree(qkv_tensor));
  ACL_CHECK_RET(aclrtFree(output));
}

}  // namespace test
}  // namespace ascend
}  // namespace llm_kernels
