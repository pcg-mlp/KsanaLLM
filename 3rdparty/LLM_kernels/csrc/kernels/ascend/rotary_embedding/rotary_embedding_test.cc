/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include <gtest/gtest.h>
#include <cmath>

#include "3rdparty/half/include/half.hpp"
#include "csrc/kernels/ascend/rotary_embedding/rotary_embedding.h"
#include "csrc/utils/ascend/common.h"
#include "tests/kernels/ascend/utils/testsuit_base.h"

#ifdef ENABLE_ACL_ATB
#  include "csrc/utils/ascend/atb_executor.h"
#endif

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace ascend {
namespace test {

class LlamaAscendRotaryEmbeddingTestSuit : public AscendTestSuitBase {
 public:
  void SetUp() override { AscendTestSuitBase::SetUp(); }

  void TearDown() override { AscendTestSuitBase::TearDown(); }

 protected:
  using AscendTestSuitBase::stream;

  template <typename T>
  void TestRotaryEmbeddingAscendC() {
    std::unique_ptr<RotaryEmbeddingAscendC<T>> rope_ptr = std::make_unique<RotaryEmbeddingAscendC<T>>();
    // following config is loaded from llama2-13B
    int rotary_dim{128};
    int max_position_embeddings{2048};
    int num_tokens{512};
    int num_kv_heads{40};
    int num_heads{40};
    int head_size{128};
    float base{10000};
    bool is_neox{true};
    size_t batch_size{1ul};
    RotaryEmbeddingType rotary_embedding_type{RotaryEmbeddingType::DEFAULT};
    float scaling_factor{1.0f};
    int query_stride = num_heads * head_size;
    int key_stride = num_kv_heads * head_size;

    // setting position
    size_t pos_size = num_tokens * sizeof(int64_t);
    uint8_t *pos_host;
    uint8_t *pos_device;
    ACL_CHECK_RET(aclrtMallocHost((void **)(&pos_host), pos_size));
    ACL_CHECK_RET(aclrtMalloc((void **)&pos_device, pos_size, ACL_MEM_MALLOC_HUGE_FIRST));
    int cpu_rotary_pos_idx = 0;
    for (size_t idx = 0; idx < batch_size; ++idx) {
      for (int pos = 0; pos < num_tokens; ++pos) {
        (reinterpret_cast<int64_t *>(pos_host))[cpu_rotary_pos_idx++] = static_cast<int64_t>(pos);
      }
    }
    ACL_CHECK_RET(aclrtMemcpy(pos_device, pos_size, pos_host, pos_size, ACL_MEMCPY_HOST_TO_DEVICE));

    // setting cos sin cache
    size_t cos_sin_cache_size = max_position_embeddings * rotary_dim * sizeof(T);
    uint8_t *cos_sin_cache_host;
    uint8_t *cos_sin_cache_device;
    std::vector<T> cos_sin_cache_ref(max_position_embeddings * rotary_dim);
    ACL_CHECK_RET(aclrtMallocHost((void **)(&cos_sin_cache_host), cos_sin_cache_size));
    ACL_CHECK_RET(aclrtMalloc((void **)&cos_sin_cache_device, cos_sin_cache_size, ACL_MEM_MALLOC_HUGE_FIRST));
    rope_ptr->SetConfig(reinterpret_cast<T *>(cos_sin_cache_device), rotary_dim, max_position_embeddings, base,
                        head_size, num_heads, num_kv_heads, query_stride, is_neox, stream);
    std::vector<size_t> tensor_shape;
    ACL_CHECK_RET(aclrtMemcpy(cos_sin_cache_host, cos_sin_cache_size, cos_sin_cache_device, cos_sin_cache_size,
                              ACL_MEMCPY_DEVICE_TO_HOST));

    // setting query input
    size_t query_size = num_tokens * query_stride * sizeof(T);
    uint8_t *query_host;
    uint8_t *query_device;
    std::vector<T> query_ref(num_tokens * query_stride);
    ACL_CHECK_RET(aclrtMallocHost((void **)(&query_host), query_size));
    ACL_CHECK_RET(aclrtMalloc((void **)&query_device, query_size, ACL_MEM_MALLOC_HUGE_FIRST));
    for (size_t i = 0; i < num_tokens * query_stride; ++i) {
      if (std::is_same<T, aclFloat16>::value) {
        ((T *)query_host)[i] = aclFloatToFloat16(float(std::sin(i)));
      } else if (std::is_same<T, float>::value) {
        ((T *)query_host)[i] = float(std::sin(i));
      } else {
        throw std::invalid_argument("Invalid rope compute type, only support float16 or float32.");
      }
      query_ref[i] = ((T *)query_host)[i];
    }
    ACL_CHECK_RET(aclrtMemcpy(query_device, query_size, query_host, query_size, ACL_MEMCPY_HOST_TO_DEVICE));

    // setting key input
    size_t key_size = num_tokens * key_stride * sizeof(T);
    uint8_t *key_host;
    uint8_t *key_device;
    std::vector<T> key_ref(num_tokens * key_stride);
    ACL_CHECK_RET(aclrtMallocHost((void **)(&key_host), key_size));
    ACL_CHECK_RET(aclrtMalloc((void **)&key_device, key_size, ACL_MEM_MALLOC_HUGE_FIRST));
    for (size_t i = 0; i < num_tokens * key_stride; ++i) {
      if (std::is_same<T, aclFloat16>::value) {
        ((T *)key_host)[i] = aclFloatToFloat16(float(std::cos(i)));
      } else if (std::is_same<T, float>::value) {
        ((T *)key_host)[i] = float(std::cos(i));
      } else {
        throw std::invalid_argument("Invalid rope compute type, only support float16 or float32.");
      }
      key_ref[i] = ((T *)key_host)[i];
    }
    ACL_CHECK_RET(aclrtMemcpy(key_device, key_size, key_host, key_size, ACL_MEMCPY_HOST_TO_DEVICE));

    rope_ptr->SetInput(reinterpret_cast<int64_t *>(pos_device), reinterpret_cast<T *>(query_device),
                       reinterpret_cast<T *>(key_device), num_tokens, stream);
    rope_ptr->Forward();
    ACL_CHECK_RET(aclrtSynchronizeStream(stream));

    if (rotary_embedding_type == RotaryEmbeddingType::DEFAULT) {
      llm_kernels::utils::LoadNpyToPtr<T>("/tmp/tests/kernels/data/rotary_embedding/cos_sin_cache_meta.npy",
                                          cos_sin_cache_ref.data(), tensor_shape, true);
      for (size_t i = 0; i < max_position_embeddings * rotary_dim; ++i) {
        if (std::is_same<T, aclFloat16>::value) {
          EXPECT_NEAR(aclFloat16ToFloat(cos_sin_cache_ref[i]), aclFloat16ToFloat(((T *)cos_sin_cache_host)[i]), 1e-3);
        } else if (std::is_same<T, float>::value) {
          EXPECT_NEAR(cos_sin_cache_ref[i], ((T *)cos_sin_cache_host)[i], 1e-3);
        } else {
          throw std::invalid_argument("Invalid rope compute type, only support float16 or float32.");
        }
      }

      ACL_CHECK_RET(aclrtMemcpy(query_host, query_size, query_device, query_size, ACL_MEMCPY_DEVICE_TO_HOST));
      ACL_CHECK_RET(aclrtMemcpy(key_host, key_size, key_device, key_size, ACL_MEMCPY_DEVICE_TO_HOST));

      llm_kernels::utils::LoadNpyToPtr<T>("/tmp/tests/kernels/data/rotary_embedding/query_meta.npy", query_ref.data(),
                                          tensor_shape, true);
      for (size_t i = 0; i < num_tokens * query_stride; ++i) {
        if (std::is_same<T, aclFloat16>::value) {
          EXPECT_NEAR(aclFloat16ToFloat(query_ref[i]), aclFloat16ToFloat(((T *)query_host)[i]), 1e-2);
        } else if (std::is_same<T, float>::value) {
          EXPECT_NEAR(query_ref[i], ((T *)query_host)[i], 1e-2);
        } else {
          throw std::invalid_argument("Invalid rope compute type, only support float16 or float32.");
        }
      }

      llm_kernels::utils::LoadNpyToPtr<T>("/tmp/tests/kernels/data/rotary_embedding/key_meta.npy", key_ref.data(),
                                          tensor_shape, true);
      for (size_t i = 0; i < num_tokens * key_stride; ++i) {
        if (std::is_same<T, aclFloat16>::value) {
          EXPECT_NEAR(aclFloat16ToFloat(key_ref[i]), aclFloat16ToFloat(((T *)key_host)[i]), 1e-2);
        } else if (std::is_same<T, float>::value) {
          EXPECT_NEAR(key_ref[i], ((T *)key_host)[i], 1e-2);
        } else {
          throw std::invalid_argument("Invalid rope compute type, only support float16 or float32.");
        }
      }
    } else if (rotary_embedding_type == RotaryEmbeddingType::DYNAMIC_NTK_SCALING) {
      throw std::invalid_argument("Invalid rope scaling type, only support default mode.");
    }

    ACL_CHECK_RET(aclrtFree(key_device));
    ACL_CHECK_RET(aclrtFreeHost(key_host));
    ACL_CHECK_RET(aclrtFree(query_device));
    ACL_CHECK_RET(aclrtFreeHost(query_host));
    ACL_CHECK_RET(aclrtFree(cos_sin_cache_device));
    ACL_CHECK_RET(aclrtFreeHost(cos_sin_cache_host));
  }

#ifdef ENABLE_ACL_ATB
  template <typename DTYPE>
  void TestATBRotaryEmbedding() {
    // following config is loaded from llama2-13B
    size_t rotary_dim{128};
    size_t max_position_embeddings{2048};
    size_t num_tokens{512};
    size_t num_kv_heads{40};
    size_t num_heads{40};
    size_t head_dim{128};
    float rope_base{10000};
    bool is_neox{true};
    size_t batch_size{1ul};
    RotaryEmbeddingType scaling_type{RotaryEmbeddingType::DEFAULT};
    float scaling_factor{1.0f};
    size_t query_stride = num_heads * head_dim;
    size_t key_stride = num_kv_heads * head_dim;

    aclDataType aclnn_dtype = aclDataType::ACL_FLOAT16;
    if (std::is_same<DTYPE, float>::value) {
      aclnn_dtype = aclDataType::ACL_FLOAT;
    } else if (std::is_same<DTYPE, aclFloat16>::value || std::is_same<DTYPE, half_float::half>::value) {
      aclnn_dtype = aclDataType::ACL_FLOAT16;
    } else {
      GTEST_SKIP_("This test is just supported float and float16.");
    }

    atb::infer::RopeParam op_param;
    op_param.rotaryCoeff = 2;
    llm_kernels::utils::ATBOperationExecutor atb_op_executor;
    atb_op_executor.Init(default_device, op_param);
    atb_op_executor.ResetVariantPack();

    std::vector<size_t> tensor_shape;
    void *rope_cos_workspace_ptr = nullptr;
    void *rope_sin_workspace_ptr = nullptr;
    std::vector<DTYPE> cos_workspace_host(max_position_embeddings * head_dim);
    std::vector<DTYPE> sin_workspace_host(max_position_embeddings * head_dim);
    ACL_CHECK_RET(aclrtMalloc(&rope_cos_workspace_ptr, max_position_embeddings * head_dim * sizeof(DTYPE),
                              ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK_RET(aclrtMalloc(&rope_sin_workspace_ptr, max_position_embeddings * head_dim * sizeof(DTYPE),
                              ACL_MEM_MALLOC_HUGE_FIRST));
    size_t extend_max_len = max_position_embeddings;
    float new_base = rope_base;
    float scaling = 1.0f;
    // https://github.com/vllm-project/vllm/blob/523e30ea0c5abcb447763dcd9a77b54d5c5f3239/vllm/model_executor/layers/rotary_embedding.py#L219
    if (scaling_type == RotaryEmbeddingType::DYNAMIC_NTK_SCALING) {
      extend_max_len = max_position_embeddings * scaling_factor;
      new_base =
          std::pow(rope_base * ((scaling_factor * extend_max_len / max_position_embeddings) - (scaling_factor - 1)),
                   (head_dim / (head_dim - 2)));
    }
    if (scaling_type == RotaryEmbeddingType::LINEAR_SCALING) {
      extend_max_len = max_position_embeddings * scaling_factor;
      scaling = scaling_factor;
    }
    for (size_t token_idx = 0; token_idx < extend_max_len; ++token_idx) {
      int pos = token_idx;
      for (size_t rid = 0; rid < head_dim / 2; ++rid) {
        float inv_freq = 1.0 / std::pow(new_base, rid * 2 / float(head_dim));
        float freq = pos * inv_freq / scaling;

        if (std::is_same<DTYPE, aclFloat16>::value) {
          cos_workspace_host[pos * head_dim + rid] = aclFloatToFloat16(std::cos(freq));
          cos_workspace_host[pos * head_dim + head_dim / 2 + rid] = cos_workspace_host[pos * head_dim + rid];
          sin_workspace_host[pos * head_dim + rid] = aclFloatToFloat16(std::sin(freq));
          sin_workspace_host[pos * head_dim + head_dim / 2 + rid] = sin_workspace_host[pos * head_dim + rid];
        } else if (std::is_same<DTYPE, float>::value || std::is_same<DTYPE, half_float::half>::value) {
          cos_workspace_host[pos * head_dim + rid] = DTYPE(std::cos(freq));
          cos_workspace_host[pos * head_dim + head_dim / 2 + rid] = cos_workspace_host[pos * head_dim + rid];
          sin_workspace_host[pos * head_dim + rid] = DTYPE(std::sin(freq));
          sin_workspace_host[pos * head_dim + head_dim / 2 + rid] = sin_workspace_host[pos * head_dim + rid];
        } else {
          throw std::invalid_argument("Invalid rope compute type, only support float16 or float32.");
        }
      }
    }
    ACL_CHECK_RET(aclrtMemcpyAsync(rope_cos_workspace_ptr, cos_workspace_host.size() * sizeof(DTYPE),
                                   cos_workspace_host.data(), cos_workspace_host.size() * sizeof(DTYPE),
                                   ACL_MEMCPY_HOST_TO_DEVICE, stream));
    ACL_CHECK_RET(aclrtMemcpyAsync(rope_sin_workspace_ptr, sin_workspace_host.size() * sizeof(DTYPE),
                                   sin_workspace_host.data(), sin_workspace_host.size() * sizeof(DTYPE),
                                   ACL_MEMCPY_HOST_TO_DEVICE, stream));
    ACL_CHECK_RET(aclrtSynchronizeStream(stream));

    // setting query input
    size_t query_size = num_tokens * query_stride * sizeof(DTYPE);
    uint8_t *query_host;
    uint8_t *query_device;
    uint8_t *query_device_result;
    std::vector<DTYPE> query_ref(num_tokens * query_stride);
    ACL_CHECK_RET(aclrtMallocHost((void **)(&query_host), query_size));
    ACL_CHECK_RET(aclrtMalloc((void **)&query_device, query_size, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK_RET(aclrtMalloc((void **)&query_device_result, query_size, ACL_MEM_MALLOC_HUGE_FIRST));
    for (size_t i = 0; i < num_tokens * query_stride; ++i) {
      if (std::is_same<DTYPE, aclFloat16>::value) {
        reinterpret_cast<DTYPE *>(query_host)[i] = aclFloatToFloat16(float(std::sin(i)));
      } else if (std::is_same<DTYPE, float>::value || std::is_same<DTYPE, half_float::half>::value) {
        reinterpret_cast<DTYPE *>(query_host)[i] = DTYPE(std::sin(i));
      } else {
        throw std::invalid_argument("Invalid rope compute type, only support float16 or float32.");
      }
      query_ref[i] = reinterpret_cast<DTYPE *>(query_host)[i];
    }
    ACL_CHECK_RET(aclrtMemcpy(query_device, query_size, query_host, query_size, ACL_MEMCPY_HOST_TO_DEVICE));

    // setting key input
    size_t key_size = num_tokens * key_stride * sizeof(DTYPE);
    uint8_t *key_host;
    uint8_t *key_device;
    uint8_t *key_device_result;
    std::vector<DTYPE> key_ref(num_tokens * key_stride);
    ACL_CHECK_RET(aclrtMallocHost((void **)(&key_host), key_size));
    ACL_CHECK_RET(aclrtMalloc((void **)&key_device, key_size, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK_RET(aclrtMalloc((void **)&key_device_result, key_size, ACL_MEM_MALLOC_HUGE_FIRST));
    for (size_t i = 0; i < num_tokens * key_stride; ++i) {
      if (std::is_same<DTYPE, aclFloat16>::value) {
        reinterpret_cast<DTYPE *>(key_host)[i] = aclFloatToFloat16(float(std::cos(i)));
      } else if (std::is_same<DTYPE, float>::value || std::is_same<DTYPE, half_float::half>::value) {
        reinterpret_cast<DTYPE *>(key_host)[i] = DTYPE(std::cos(i));
      } else {
        throw std::invalid_argument("Invalid rope compute type, only support float16 or float32.");
      }
      key_ref[i] = reinterpret_cast<DTYPE *>(key_host)[i];
    }
    ACL_CHECK_RET(aclrtMemcpy(key_device, key_size, key_host, key_size, ACL_MEMCPY_HOST_TO_DEVICE));

    void *seq_len_device;
    size_t seq_len_size = batch_size * sizeof(uint32_t);
    ACL_CHECK_RET(aclrtMalloc((void **)&seq_len_device, seq_len_size, ACL_MEM_MALLOC_HUGE_FIRST));
    std::vector<uint32_t> seq_len_host(1, num_tokens);
    ACL_CHECK_RET(
        aclrtMemcpy(seq_len_device, seq_len_size, seq_len_host.data(), seq_len_size, ACL_MEMCPY_HOST_TO_DEVICE));

    atb_op_executor.SetInputTensor(query_device, {num_tokens, query_stride}, aclnn_dtype);
    atb_op_executor.SetInputTensor(key_device, {num_tokens, key_stride}, aclnn_dtype);
    atb_op_executor.SetInputTensor(rope_cos_workspace_ptr, {num_tokens, head_dim}, aclnn_dtype);
    atb_op_executor.SetInputTensor(rope_sin_workspace_ptr, {num_tokens, head_dim}, aclnn_dtype);
    atb_op_executor.SetInputTensor(seq_len_device, {batch_size}, aclDataType::ACL_UINT32);

    atb_op_executor.SetOutputTensor(query_device_result, {num_tokens, query_stride}, aclnn_dtype);
    atb_op_executor.SetOutputTensor(key_device_result, {num_tokens, key_stride}, aclnn_dtype);

    atb_op_executor.Run(atb_context, llm_kernels::utils::GetTestWorkSpaceFunc);
    ACL_CHECK_RET(aclrtSynchronizeStream(stream));

    ACL_CHECK_RET(aclrtMemcpy(query_host, query_size, query_device_result, query_size, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK_RET(aclrtMemcpy(key_host, key_size, key_device_result, key_size, ACL_MEMCPY_DEVICE_TO_HOST));

    llm_kernels::utils::LoadNpyToPtr<aclFloat16>("/tmp/tests/kernels/data/rotary_embedding/query_meta.npy",
                                                 reinterpret_cast<aclFloat16 *>(query_ref.data()), tensor_shape, true);
    for (size_t i = 0; i < num_tokens * query_stride; ++i) {
      if (std::is_same<DTYPE, aclFloat16>::value) {
        EXPECT_NEAR(aclFloat16ToFloat(query_ref[i]), aclFloat16ToFloat(reinterpret_cast<DTYPE *>(query_host)[i]), 1e-2);
      } else if (std::is_same<DTYPE, float>::value || std::is_same<DTYPE, half_float::half>::value) {
        EXPECT_NEAR(query_ref[i], reinterpret_cast<DTYPE *>(query_host)[i], 1e-2);
      } else {
        throw std::invalid_argument("Invalid rope compute type, only support float16 or float32.");
      }
    }

    llm_kernels::utils::LoadNpyToPtr<aclFloat16>("/tmp/tests/kernels/data/rotary_embedding/key_meta.npy",
                                                 reinterpret_cast<aclFloat16 *>(key_ref.data()), tensor_shape, true);
    for (size_t i = 0; i < num_tokens * key_stride; ++i) {
      if (std::is_same<DTYPE, aclFloat16>::value) {
        EXPECT_NEAR(aclFloat16ToFloat(key_ref[i]), aclFloat16ToFloat(reinterpret_cast<DTYPE *>(key_host)[i]), 1e-2);
      } else if (std::is_same<DTYPE, float>::value || std::is_same<DTYPE, half_float::half>::value) {
        EXPECT_NEAR(key_ref[i], reinterpret_cast<DTYPE *>(key_host)[i], 1e-2);
      } else {
        throw std::invalid_argument("Invalid rope compute type, only support float16 or float32.");
      }
    }

    ACL_CHECK_RET(aclrtFree(seq_len_device));
    ACL_CHECK_RET(aclrtFree(query_device));
    ACL_CHECK_RET(aclrtFree(key_device));
    ACL_CHECK_RET(aclrtFree(rope_sin_workspace_ptr));
    ACL_CHECK_RET(aclrtFree(rope_cos_workspace_ptr));
  }
#endif
};

TEST_F(LlamaAscendRotaryEmbeddingTestSuit, KernelTest) { TestRotaryEmbeddingAscendC<aclFloat16>(); }

#ifdef ENABLE_ACL_ATB
TEST_F(LlamaAscendRotaryEmbeddingTestSuit, ATBRopeTest) { TestATBRotaryEmbedding<half_float::half>(); }
#endif

}  // namespace test
}  // namespace ascend
}  // namespace llm_kernels
