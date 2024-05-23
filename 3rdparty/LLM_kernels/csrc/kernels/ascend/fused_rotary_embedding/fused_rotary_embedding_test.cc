/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include <gtest/gtest.h>
#include <cmath>

#include "3rdparty/half/include/half.hpp"
#include "csrc/kernels/ascend/fused_rotary_embedding/fused_rotary_embedding.h"
#include "csrc/utils/ascend/common.h"
#include "tests/kernels/ascend/utils/testsuit_base.h"

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
    // following config is loaded from llama-13B
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

    // setting workspace
    uint8_t *workspace_device;
    size_t workspace_size = num_tokens * rotary_dim * sizeof(T) * 2;
    ACL_CHECK_RET(aclrtMalloc((void **)&workspace_device, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST));

    rope_ptr->SetInput(reinterpret_cast<int64_t *>(pos_device), reinterpret_cast<T *>(query_device),
                       reinterpret_cast<T *>(key_device), reinterpret_cast<T *>(workspace_device), num_tokens, stream);
    rope_ptr->Forward();

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
    } else if (rotary_embedding_type == RotaryEmbeddingType::DYNAMIC_NTK_SCALING) {
      throw std::invalid_argument("Invalid rope scaling type, only support default mode.");
    }

    ACL_CHECK_RET(aclrtFree(workspace_device));
    ACL_CHECK_RET(aclrtFree(key_device));
    ACL_CHECK_RET(aclrtFreeHost(key_host));
    ACL_CHECK_RET(aclrtFree(query_device));
    ACL_CHECK_RET(aclrtFreeHost(query_host));
    ACL_CHECK_RET(aclrtFree(cos_sin_cache_device));
    ACL_CHECK_RET(aclrtFreeHost(cos_sin_cache_host));
  }
};

TEST_F(LlamaAscendRotaryEmbeddingTestSuit, KernelTest) { TestRotaryEmbeddingAscendC<aclFloat16>(); }

}  // namespace test
}  // namespace ascend
}  // namespace llm_kernels