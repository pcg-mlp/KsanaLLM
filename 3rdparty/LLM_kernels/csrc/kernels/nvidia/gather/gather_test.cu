/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

#include <gtest/gtest.h>
#include <random>

#include "csrc/kernels/nvidia/gather/gather.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

namespace llm_kernels {
namespace nvidia {
namespace test {

class LlamaNvidiaGatherTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }

  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

  // 1. 随机打乱的索引
  std::vector<int> GenerateRandomIndexes(size_t len) {
    std::vector<int> indices(len);
    for (size_t i = 0; i < len; ++i) {
      indices[i] = i;
    }
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(indices.begin(), indices.end(), gen);
    return indices;
  }

  // 2. 反向索引
  std::vector<int> GenerateReverseIndexes(size_t len) {
    std::vector<int> indices(len);
    for (size_t i = 0; i < len; ++i) {
      indices[i] = len - 1 - i;
    }
    return indices;
  }

  // 3. 交替索引（偶数位置和奇数位置交替）
  std::vector<int> GenerateAlternatingIndexes(size_t len) {
    std::vector<int> indices(len);
    size_t j = 0;
    // 先放偶数索引
    for (size_t i = 0; i < len; i += 2) {
      indices[j++] = i;
    }
    // 再放奇数索引
    for (size_t i = 1; i < len; i += 2) {
      indices[j++] = i;
    }
    return indices;
  }

 protected:
  using NvidiaTestSuitBase::stream;
};

TEST_F(LlamaNvidiaGatherTestSuit, GatherFirstDimAccTest) {
  const std::vector<size_t> test_lens = {1ul, 18ul, 1007ul};
  const std::vector<size_t> test_data_sizes = {34ul, 2048ul};

  for (const size_t len : test_lens) {
    for (const size_t data_size : test_data_sizes) {
      BufferMeta input_device = CreateBuffer<char>(MemoryType::MEMORY_GPU, {len * data_size}, /*is_random_init*/ true);
      BufferMeta input_host = CopyToHost<char>(input_device);

      // 使用不同的索引模式进行测试
      std::vector<std::vector<int>> index_patterns = {GenerateRandomIndexes(len), GenerateReverseIndexes(len),
                                                      GenerateAlternatingIndexes(len)};
      for (const auto& indices : index_patterns) {
        BufferMeta indices_device = CreateBuffer<int>(MemoryType::MEMORY_GPU, {len}, /*is_random_init*/ false);
        CHECK_NVIDIA_CUDA_ERROR(
            cudaMemcpy(indices_device.data_ptr, indices.data(), len * sizeof(int), cudaMemcpyHostToDevice));
        BufferMeta output_device = CreateBuffer<char>(MemoryType::MEMORY_GPU, {len * data_size},
                                                      /*is_random_init*/ false);

        GatherFirstDim(input_device.data_ptr, reinterpret_cast<int*>(indices_device.data_ptr), len, data_size,
                       output_device.data_ptr, stream);

        BufferMeta output_host = CopyToHost<char>(output_device);
        CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

        // 验证结果
        char* input_ptr = reinterpret_cast<char*>(input_host.data_ptr);
        char* output_ptr = reinterpret_cast<char*>(output_host.data_ptr);
        for (size_t i = 0; i < len; ++i) {
          for (size_t j = 0; j < data_size; ++j) {
            size_t src_idx = indices[i] * data_size + j;
            size_t dst_idx = i * data_size + j;
            EXPECT_EQ(output_ptr[dst_idx], input_ptr[src_idx])
                << "Mismatch at len=" << len << ", data_size=" << data_size << ", i=" << i << ", j=" << j
                << ", src_idx=" << indices[i];
          }
        }
      }
    }
  }
}

// Performance test is disabled by default
TEST_F(LlamaNvidiaGatherTestSuit, DISABLED_GatherFirstDimPerfTest) {
  const std::vector<size_t> test_lens = {1ul, 8ul, 16ul, 32ul, 64ul, 128ul, 256ul, 512ul, 1024ul, 2048ul};
  const std::vector<size_t> test_data_sizes = {34ul, 2048ul, 7168ul};

  for (const size_t len : test_lens) {
    for (const size_t data_size : test_data_sizes) {
      BufferMeta input_device = CreateBuffer<char>(MemoryType::MEMORY_GPU, {len * data_size}, /*is_random_init*/ true);
      BufferMeta input_host = CopyToHost<char>(input_device);

      const auto indices = GenerateRandomIndexes(len);
      BufferMeta indices_device = CreateBuffer<int>(MemoryType::MEMORY_GPU, {len}, /*is_random_init*/ false);
      CHECK_NVIDIA_CUDA_ERROR(
          cudaMemcpy(indices_device.data_ptr, indices.data(), len * sizeof(int), cudaMemcpyHostToDevice));
      BufferMeta output_device = CreateBuffer<char>(MemoryType::MEMORY_GPU, {len * data_size},
                                                    /*is_random_init*/ false);

      const int warmups = 5;
      const int iterations = 10;
      auto run = [&]() {
        GatherFirstDim(input_device.data_ptr, reinterpret_cast<int*>(indices_device.data_ptr), len, data_size,
                       output_device.data_ptr, stream);
      };
      const float elapsed_ms = MeasureCudaExecutionTime(run, stream, warmups, iterations);
      std::cout << "Len: " << len << ", Data size: " << data_size << ", Time: " << elapsed_ms << " ms" << std::endl;
    }
  }
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
