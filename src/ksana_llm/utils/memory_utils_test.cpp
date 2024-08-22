/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/utils/memory_utils.h"
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

}  // namespace ksana_llm
