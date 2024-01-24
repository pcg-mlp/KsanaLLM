/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <random>
#include <thread>

#include "tests/test.h"

#include "numerous_llm/runtime/context.h"

namespace numerous_llm {
class ContextTest : public testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(ContextTest, ParallelConfigTest) {
  EXPECT_THROW(
      {
        try {
          std::shared_ptr<Context> context =
              std::make_shared<Context>(/*tensor_parallel_size*/ 1, /*pipeline_parallel_size*/ 2);
        } catch (const std::runtime_error& e) {
          EXPECT_STREQ("Only support pipeline_parallel_size == 1", e.what());
          throw;
        }
      },
      std::runtime_error);

  EXPECT_THROW(
      {
        try {
          std::shared_ptr<Context> context =
              std::make_shared<Context>(/*tensor_parallel_size*/ 100, /*pipeline_parallel_size*/ 1);
        } catch (const std::runtime_error& e) {
          EXPECT_THAT(e.what(), testing::HasSubstr("tensor_parallel_size should not bigger than devices num:"));
          throw;
        }
      },
      std::runtime_error);

  const int tensor_parallel_size = 2;
  const int pipeline_parallel_size = 1;
  std::shared_ptr<Context> context = std::make_shared<Context>(tensor_parallel_size, pipeline_parallel_size);
  size_t total_rank_num = tensor_parallel_size * pipeline_parallel_size;

  EXPECT_EQ(context->GetComputeStreams().size(), total_rank_num);
  EXPECT_EQ(context->GetMemoryPools().size(), total_rank_num);
  EXPECT_EQ(context->GetMemoryManageStreams().size(), total_rank_num);
  EXPECT_EQ(context->GetH2DStreams().size(), total_rank_num);
  EXPECT_EQ(context->GetD2HStreams().size(), total_rank_num);
  EXPECT_EQ(context->GetD2DStreams().size(), total_rank_num);
  EXPECT_EQ(context->GetNCCLStreams().size(), total_rank_num);
  EXPECT_EQ(context->GetCublasHandles().size(), total_rank_num);
  EXPECT_EQ(context->GetCublasLtHandles().size(), total_rank_num);
  EXPECT_EQ(context->GetNCCLParam().size(), total_rank_num);

  EXPECT_EQ(context->GetDevice(), MemoryDevice::MEMORY_GPU);
  EXPECT_EQ(context->GetTensorParallelSize(), tensor_parallel_size);
  EXPECT_EQ(context->GetPipeLineParallelSize(), pipeline_parallel_size);

  for (size_t rank_idx = 0; rank_idx < total_rank_num; ++rank_idx) {
    // check stream valid
    CUDA_CHECK(cudaStreamSynchronize(context->GetComputeStreams()[rank_idx]));
    CUDA_CHECK(cudaStreamSynchronize(context->GetMemoryManageStreams()[rank_idx]));
    CUDA_CHECK(cudaStreamSynchronize(context->GetH2DStreams()[rank_idx]));
    CUDA_CHECK(cudaStreamSynchronize(context->GetD2HStreams()[rank_idx]));
    CUDA_CHECK(cudaStreamSynchronize(context->GetD2DStreams()[rank_idx]));
    CUDA_CHECK(cudaStreamSynchronize(context->GetNCCLStreams()[rank_idx]));

    // check mem pool valid
    uint64_t mempool_threshold;
    CUDA_CHECK(cudaMemPoolGetAttribute(context->GetMemoryPools()[rank_idx], cudaMemPoolAttrReleaseThreshold,
                                       &mempool_threshold));
    EXPECT_EQ(mempool_threshold, 0ul);

    // check nccl
    EXPECT_NE(context->GetNCCLParam()[rank_idx].nccl_comm, nullptr);
    EXPECT_GE(context->GetNCCLParam()[rank_idx].rank, 0);
    EXPECT_GE(context->GetNCCLParam()[rank_idx].world_size, 1);

    // cublas
    int cublas_ver = -1;
    CUDA_CHECK(cublasGetVersion(context->GetCublasHandles()[rank_idx], &cublas_ver));
    EXPECT_NE(cublas_ver, -1);
    EXPECT_NE(context->GetCublasLtHandles()[rank_idx], nullptr);
  }
}

}  // namespace numerous_llm