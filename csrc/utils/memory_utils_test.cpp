/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "utils/memory_utils.h"
#include "logger.h"
#include "test.h"

namespace ksana_llm {

TEST(AlignedMemoryQueueTest, ConstructorThrowsWhenAlignmentIsNotPowerOfTwo) {
  EXPECT_THROW({ AlignedMemoryQueue queue(3, [](size_t) -> void* { return nullptr; }); }, std::runtime_error);
}

TEST(AlignedMemoryQueueTest, AllocateAndAlignCorrectlyAllocatesMemory) {
  AlignedMemoryQueue queue(16, [](size_t size) -> void* { return malloc(size); });
  int* ptr1 = nullptr;
  double* ptr2 = nullptr;
  double* ptr3 = nullptr;
  queue.Add(ptr1, 1);
  queue.Add(ptr2, 1);
  queue.Add(ptr3, 0);
  queue.AllocateAndAlign();

  EXPECT_NE(nullptr, ptr1);
  EXPECT_NE(nullptr, ptr2);
  EXPECT_EQ(nullptr, ptr3);
  EXPECT_TRUE(reinterpret_cast<uintptr_t>(ptr1) % 16 == 0);
  EXPECT_TRUE(reinterpret_cast<uintptr_t>(ptr2) % 16 == 0);
}

TEST(MemoryUtilsTest, ComputeIndexerMaxQChunkSize) {
  // chunk = 4GB / (RoundUp(256, 256) * sizeof(float)) ≈ 4M, capped by max_step_token_num.
  EXPECT_EQ(128u, ComputeIndexerMaxQChunkSize(256, 128));
  // Value rounded up to nearest 4 for kSeqLenAlignment.
  EXPECT_EQ(4u, ComputeIndexerMaxQChunkSize(256, 3));
  // computed_q_chunk = 2^30 / (1<<28) = 4, smaller than max_step_token_num, capped by computed value.
  EXPECT_EQ(4u, ComputeIndexerMaxQChunkSize(1u << 28, 128));
}

}  // namespace ksana_llm
