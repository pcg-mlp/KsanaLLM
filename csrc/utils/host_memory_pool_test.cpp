/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "utils/host_memory_pool.h"

#include <fmt/format.h>

#include <algorithm>
#include <random>
#include <thread>
#include <vector>

#include "logger.h"
#include "test.h"

namespace ksana_llm {

class HostMemoryPoolTest : public ::testing::Test {
 protected:
  void TearDown() override { HostMemoryPool::GetInstance()->Destroy(); }
};

// 测试单例获取
TEST_F(HostMemoryPoolTest, GetInstance) {
  auto* pool1 = HostMemoryPool::GetInstance();
  auto* pool2 = HostMemoryPool::GetInstance();
  EXPECT_NE(pool1, nullptr);
  EXPECT_EQ(pool1, pool2);  // 应该是同一个实例
}

// 测试初始化
TEST_F(HostMemoryPoolTest, Initialize) {
  auto* pool = HostMemoryPool::GetInstance();
  EXPECT_FALSE(pool->IsInitialized());

  pool->Initialize(1);
  EXPECT_TRUE(pool->IsInitialized());
  EXPECT_EQ(pool->GetUsedSize(), 0);
  EXPECT_GT(pool->GetPoolSize(), 0);
}

// 测试重复初始化（应该跳过）
TEST_F(HostMemoryPoolTest, DoubleInitialize) {
  auto* pool = HostMemoryPool::GetInstance();
  pool->Initialize(1);
  size_t first_pool_size = pool->GetPoolSize();

  // 第二次初始化应该被跳过
  pool->Initialize(2);
  EXPECT_EQ(pool->GetPoolSize(), first_pool_size);
}

// 测试基本分配和释放
TEST_F(HostMemoryPoolTest, BasicAllocateAndFree) {
  auto* pool = HostMemoryPool::GetInstance();
  pool->Initialize(1);

  // 测试基本分配
  void* ptr1 = pool->Allocate(1024);
  EXPECT_NE(ptr1, nullptr);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr1) % kHostMemoryAlignment, 0);  // 对齐检查
  EXPECT_GT(pool->GetUsedSize(), 0);

  // 测试释放
  pool->Free(ptr1);
  EXPECT_EQ(pool->GetUsedSize(), 0);
}

// 测试零大小分配
TEST_F(HostMemoryPoolTest, ZeroSizeAllocation) {
  auto* pool = HostMemoryPool::GetInstance();
  pool->Initialize(1);

  void* ptr = pool->Allocate(0);
  EXPECT_EQ(ptr, nullptr);
}

// 测试多次分配
TEST_F(HostMemoryPoolTest, MultipleAllocations) {
  auto* pool = HostMemoryPool::GetInstance();
  pool->Initialize(1);

  std::vector<void*> ptrs;
  for (int i = 0; i < 10; ++i) {
    void* ptr = pool->Allocate(1024);
    EXPECT_NE(ptr, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % kHostMemoryAlignment, 0);
    ptrs.push_back(ptr);
  }

  // 验证所有指针都不同
  for (size_t i = 0; i < ptrs.size(); ++i) {
    for (size_t j = i + 1; j < ptrs.size(); ++j) {
      EXPECT_NE(ptrs[i], ptrs[j]);
    }
  }

  // 释放所有
  for (void* ptr : ptrs) {
    pool->Free(ptr);
  }
  EXPECT_EQ(pool->GetUsedSize(), 0);
}

// 测试相邻块合并
TEST_F(HostMemoryPoolTest, Coalescing) {
  auto* pool = HostMemoryPool::GetInstance();
  pool->Initialize(1);

  void* ptr1 = pool->Allocate(256);
  void* ptr2 = pool->Allocate(256);
  void* ptr3 = pool->Allocate(256);

  EXPECT_NE(ptr1, nullptr);
  EXPECT_NE(ptr2, nullptr);
  EXPECT_NE(ptr3, nullptr);

  // 释放中间块，然后释放相邻块，验证合并
  pool->Free(ptr2);
  pool->Free(ptr1);
  pool->Free(ptr3);

  // 合并后应该能分配更大的块
  void* large_ptr = pool->Allocate(768);
  EXPECT_NE(large_ptr, nullptr);

  pool->Free(large_ptr);
  EXPECT_EQ(pool->GetUsedSize(), 0);
}

// 测试内存池耗尽
TEST_F(HostMemoryPoolTest, PoolExhaustion) {
  auto* pool = HostMemoryPool::GetInstance();
  pool->Initialize(1);

  // 请求超过池大小（1GB + 1）
  void* ptr = pool->Allocate(static_cast<size_t>(1024) * 1024 * 1024 + 1);
  EXPECT_EQ(ptr, nullptr);  // 应返回 nullptr
}

// 测试重置功能
TEST_F(HostMemoryPoolTest, Reset) {
  auto* pool = HostMemoryPool::GetInstance();
  pool->Initialize(1);

  // 分配一些内存
  void* ptr1 = pool->Allocate(1024);
  void* ptr2 = pool->Allocate(2048);
  EXPECT_NE(ptr1, nullptr);
  EXPECT_NE(ptr2, nullptr);
  EXPECT_GT(pool->GetUsedSize(), 0);

  // 重置
  pool->Reset();
  EXPECT_EQ(pool->GetUsedSize(), 0);
  EXPECT_EQ(pool->GetAllocatedBlockCount(), 0);
  EXPECT_EQ(pool->GetFreeBlockCount(), 1);
}

// 测试释放未知指针
TEST_F(HostMemoryPoolTest, FreeUnknownPointer) {
  auto* pool = HostMemoryPool::GetInstance();
  pool->Initialize(1);

  // 释放一个不是从池中分配的指针（应该打印错误日志但不崩溃）
  int dummy = 0;
  pool->Free(&dummy);  // 不应该崩溃

  // 释放 nullptr（应该安全处理）
  pool->Free(nullptr);
}

// 测试统计信息
TEST_F(HostMemoryPoolTest, Statistics) {
  auto* pool = HostMemoryPool::GetInstance();
  pool->Initialize(1);

  EXPECT_EQ(pool->GetUsedSize(), 0);
  EXPECT_EQ(pool->GetAllocatedBlockCount(), 0);

  void* ptr1 = pool->Allocate(1024);
  EXPECT_GT(pool->GetUsedSize(), 0);
  EXPECT_EQ(pool->GetAllocatedBlockCount(), 1);

  void* ptr2 = pool->Allocate(2048);
  EXPECT_EQ(pool->GetAllocatedBlockCount(), 2);

  pool->Free(ptr1);
  EXPECT_EQ(pool->GetAllocatedBlockCount(), 1);

  pool->Free(ptr2);
  EXPECT_EQ(pool->GetAllocatedBlockCount(), 0);
  EXPECT_EQ(pool->GetUsedSize(), 0);
}

// 测试对齐
TEST_F(HostMemoryPoolTest, Alignment) {
  auto* pool = HostMemoryPool::GetInstance();
  pool->Initialize(1);

  // 测试不同大小的分配，验证对齐
  std::vector<size_t> sizes = {1, 7, 15, 33, 100, 255, 257, 500, 1000};
  std::vector<void*> ptrs;

  for (size_t size : sizes) {
    void* ptr = pool->Allocate(size);
    EXPECT_NE(ptr, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % kHostMemoryAlignment, 0)
        << fmt::format("Allocation of size {} is not aligned", size);
    ptrs.push_back(ptr);
  }

  for (void* ptr : ptrs) {
    pool->Free(ptr);
  }
}

// 测试并发访问
TEST_F(HostMemoryPoolTest, ConcurrentAccess) {
  auto* pool = HostMemoryPool::GetInstance();
  pool->Initialize(10);  // 10GB

  const int num_threads = 8;
  const int ops_per_thread = 100;
  std::vector<std::thread> threads;

  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([pool, ops_per_thread]() {
      std::vector<void*> ptrs;
      for (int i = 0; i < ops_per_thread; ++i) {
        void* ptr = pool->Allocate(1024);
        if (ptr) ptrs.push_back(ptr);
      }
      for (void* ptr : ptrs) {
        pool->Free(ptr);
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  EXPECT_EQ(pool->GetUsedSize(), 0);
}

// 测试碎片化压力
TEST_F(HostMemoryPoolTest, FragmentationStress) {
  auto* pool = HostMemoryPool::GetInstance();
  pool->Initialize(10);  // 10GB

  std::vector<void*> ptrs;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> size_dist(256, 4096);

  // 随机分配
  for (int i = 0; i < 100; ++i) {
    void* ptr = pool->Allocate(size_dist(gen));
    if (ptr) ptrs.push_back(ptr);
  }

  // 随机释放一半
  std::shuffle(ptrs.begin(), ptrs.end(), gen);
  size_t half = ptrs.size() / 2;
  for (size_t i = 0; i < half; ++i) {
    pool->Free(ptrs[i]);
  }
  ptrs.erase(ptrs.begin(), ptrs.begin() + half);

  // 再次分配
  for (int i = 0; i < 50; ++i) {
    void* ptr = pool->Allocate(size_dist(gen));
    if (ptr) ptrs.push_back(ptr);
  }

  // 全部释放
  for (void* ptr : ptrs) {
    pool->Free(ptr);
  }

  // 由于相邻块合并，used_size 应该为 0
  EXPECT_EQ(pool->GetUsedSize(), 0);
}

// 测试销毁功能
TEST_F(HostMemoryPoolTest, Destroy) {
  auto* pool = HostMemoryPool::GetInstance();
  pool->Initialize(1);
  EXPECT_TRUE(pool->IsInitialized());

  pool->Destroy();
  EXPECT_FALSE(pool->IsInitialized());

  // 销毁后可以重新初始化
  pool->Initialize(2);
  EXPECT_TRUE(pool->IsInitialized());
}

// 测试未初始化时的操作（内存池大小为 0）
TEST_F(HostMemoryPoolTest, OperationsBeforeInitialize) {
  auto* pool = HostMemoryPool::GetInstance();
  pool->Initialize(0);  // 大小为 0，不会真正初始化
  EXPECT_FALSE(pool->IsInitialized());

  // 未初始化时分配应返回 nullptr
  void* ptr = pool->Allocate(1024);
  EXPECT_EQ(ptr, nullptr);

  // 未初始化时释放应安全处理
  pool->Free(nullptr);

  // 未初始化时重置应安全处理
  pool->Reset();
}

// Test PrintStats (no crash).
TEST_F(HostMemoryPoolTest, PrintStats) {
  auto* pool = HostMemoryPool::GetInstance();

  pool->PrintStats();

  pool->Initialize(1);
  pool->PrintStats();

  void* ptr = pool->Allocate(1024);
  pool->PrintStats();

  pool->Free(ptr);
}

// 测试 PrintStats(true) 详细模式（不崩溃即可）
TEST_F(HostMemoryPoolTest, PrintDetailedStats) {
  auto* pool = HostMemoryPool::GetInstance();

  // 未初始化时调用
  pool->PrintStats(true);

  pool->Initialize(1);

  // 初始化后调用
  pool->PrintStats(true);

  // 分配一些内存后调用
  void* ptr1 = pool->Allocate(1024);
  void* ptr2 = pool->Allocate(2048);
  pool->PrintStats(true);

  // 释放部分内存后调用（产生碎片）
  pool->Free(ptr1);
  pool->PrintStats(true);

  // 全部释放后调用
  pool->Free(ptr2);
  pool->PrintStats(true);
}

// 测试 GetFreeSize 方法
TEST_F(HostMemoryPoolTest, GetFreeSize) {
  auto* pool = HostMemoryPool::GetInstance();
  pool->Initialize(1);

  size_t initial_free = pool->GetFreeSize();
  EXPECT_EQ(initial_free, pool->GetPoolSize());

  void* ptr = pool->Allocate(1024);
  EXPECT_NE(ptr, nullptr);

  // 分配后，空闲空间应该减少
  size_t after_alloc_free = pool->GetFreeSize();
  EXPECT_LT(after_alloc_free, initial_free);

  // GetFreeSize + GetUsedSize 应该等于 GetPoolSize
  EXPECT_EQ(pool->GetFreeSize() + pool->GetUsedSize(), pool->GetPoolSize());

  pool->Free(ptr);

  // 释放后，空闲空间应该恢复
  EXPECT_EQ(pool->GetFreeSize(), initial_free);
}

// 测试内存池禁用（host_pool_size_gb 设置为 0）
TEST_F(HostMemoryPoolTest, PoolDisabledByZero) {
  auto* pool = HostMemoryPool::GetInstance();
  pool->Initialize(0);  // 显式设置为 0
  EXPECT_FALSE(pool->IsInitialized());
}

}  // namespace ksana_llm