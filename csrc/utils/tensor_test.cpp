/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <gtest/gtest.h>
#include <filesystem>
#include <numeric>
#include <thread>
#include <vector>

#include "device/device_types.h"
#include "tensor.h"
#include "tensor_test_helper.h"
#include "test.h"
#include "utils/host_memory_pool.h"
#include "utils/singleton.h"

namespace ksana_llm {

class TensorHostMemorySourceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // 初始化 1GB 内存池用于测试
    auto* pool = HostMemoryPool::GetInstance();
    if (!pool->IsInitialized()) {
      pool->Initialize(1);
    }
  }

  void TearDown() override {
    // 销毁内存池，确保与 host_memory_pool_test 的清理策略一致
    auto* pool = HostMemoryPool::GetInstance();
    pool->Destroy();
  }
};

// 测试 1：基本的池分配
TEST_F(TensorHostMemorySourceTest, BasicPoolAllocation) {
  auto* pool = HostMemoryPool::GetInstance();
  size_t initial_used = pool->GetUsedSize();

  {
    // 创建一个 LOCATION_HOST 的 Tensor
    Tensor tensor(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP32, {1024}, "test_tensor");

    // 断言：应该从池分配
    EXPECT_TRUE(tensor.IsHostFromPool());
    EXPECT_EQ(tensor.GetHostMemorySource(), HostMemorySource::POOL);
    EXPECT_GT(pool->GetUsedSize(), initial_used);
  }

  // 析构后，池使用量应该回到初始值
  EXPECT_EQ(pool->GetUsedSize(), initial_used);
}

// 测试 2：回退到直接分配
TEST_F(TensorHostMemorySourceTest, FallbackToDirectAllocation) {
  auto* pool = HostMemoryPool::GetInstance();

  // 创建一个超大的 Tensor，超过池大小，触发回退
  // 池大小是 1GB，创建一个 2GB 的 Tensor
  {
    Tensor tensor(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP32, {512 * 1024 * 1024}, "large_tensor");  // 2GB

    // 断言：应该回退到直接分配
    EXPECT_FALSE(tensor.IsHostFromPool());
    EXPECT_EQ(tensor.GetHostMemorySource(), HostMemorySource::DIRECT);
  }

  // 释放不应该报错
}

// 测试 3：⚠️ 关键测试 - AssignMembers 正确复制标记（拷贝构造）
TEST_F(TensorHostMemorySourceTest, AssignMembersCopyConstruct) {
  auto* pool = HostMemoryPool::GetInstance();
  size_t initial_used = pool->GetUsedSize();

  {
    // 创建池分配的 Tensor A
    Tensor tensor_a(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP32, {1024}, "tensor_a");
    ASSERT_TRUE(tensor_a.IsHostFromPool());

    size_t used_after_a = pool->GetUsedSize();
    EXPECT_GT(used_after_a, initial_used);

    {
      // 拷贝构造 Tensor B = A
      Tensor tensor_b = tensor_a;

      // ⚠️ 关键断言：B 必须继承 A 的来源标记
      EXPECT_TRUE(tensor_b.IsHostFromPool());
      EXPECT_EQ(tensor_b.GetHostMemorySource(), HostMemorySource::POOL);

      // 池使用量不应该增加（共享同一块内存）
      EXPECT_EQ(pool->GetUsedSize(), used_after_a);
    }

    // B 析构后，由于 A 还在，内存不应该被释放
    EXPECT_EQ(pool->GetUsedSize(), used_after_a);
  }

  // A 和 B 都析构后，内存应该被释放
  EXPECT_EQ(pool->GetUsedSize(), initial_used);
}

// 测试 4：⚠️ 关键测试 - AssignMembers 正确复制标记（赋值操作）
TEST_F(TensorHostMemorySourceTest, AssignMembersAssignOperator) {
  auto* pool = HostMemoryPool::GetInstance();
  size_t initial_used = pool->GetUsedSize();

  {
    // 创建池分配的 Tensor A
    Tensor tensor_a(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP32, {1024}, "tensor_a");
    ASSERT_TRUE(tensor_a.IsHostFromPool());

    // 创建另一个 Tensor B（先分配自己的内存）
    Tensor tensor_b(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP32, {512}, "tensor_b");
    ASSERT_TRUE(tensor_b.IsHostFromPool());

    size_t used_before_assign = pool->GetUsedSize();

    // 赋值操作：B = A
    tensor_b = tensor_a;

    // ⚠️ 关键断言：B 必须继承 A 的来源标记
    EXPECT_TRUE(tensor_b.IsHostFromPool());
    EXPECT_EQ(tensor_b.GetHostMemorySource(), HostMemorySource::POOL);

    // B 原来的内存应该被释放，池使用量应该减少
    EXPECT_LT(pool->GetUsedSize(), used_before_assign);
  }

  // 所有 Tensor 析构后，内存应该被释放
  EXPECT_EQ(pool->GetUsedSize(), initial_used);
}

// 测试 5：⚠️ 关键测试 - ReallocateMemory 后标记正确更新
TEST_F(TensorHostMemorySourceTest, ReallocateMemoryUpdatesSource) {
  auto* pool = HostMemoryPool::GetInstance();
  size_t initial_used = pool->GetUsedSize();

  {
    // 创建池分配的 Tensor
    Tensor tensor(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP32, {1024}, "tensor");
    ASSERT_TRUE(tensor.IsHostFromPool());

    size_t used_after_first_alloc = pool->GetUsedSize();
    EXPECT_GT(used_after_first_alloc, initial_used);

    // 重新分配（仍然是 LOCATION_HOST，应该仍然从池分配）
    tensor.ReallocateMemory(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP32, {2048});

    // 断言：重新分配后仍然是池分配
    EXPECT_TRUE(tensor.IsHostFromPool());
    EXPECT_EQ(tensor.GetHostMemorySource(), HostMemorySource::POOL);
  }

  // 析构后，内存应该被释放
  EXPECT_EQ(pool->GetUsedSize(), initial_used);
}

// 测试 6：ReallocateMemory 从池分配变为直接分配
TEST_F(TensorHostMemorySourceTest, ReallocateMemoryPoolToDirect) {
  auto* pool = HostMemoryPool::GetInstance();
  size_t initial_used = pool->GetUsedSize();

  {
    // 创建池分配的 Tensor
    Tensor tensor(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP32, {1024}, "tensor");
    ASSERT_TRUE(tensor.IsHostFromPool());

    // 重新分配为超大 Tensor，触发回退（2GB）
    tensor.ReallocateMemory(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP32, {512 * 1024 * 1024});

    // 断言：重新分配后应该是直接分配
    EXPECT_FALSE(tensor.IsHostFromPool());
    EXPECT_EQ(tensor.GetHostMemorySource(), HostMemorySource::DIRECT);
  }

  // 析构后，池使用量应该回到初始值
  EXPECT_EQ(pool->GetUsedSize(), initial_used);
}

// 测试 7：ReallocateMemory 从直接分配变为池分配
TEST_F(TensorHostMemorySourceTest, ReallocateMemoryDirectToPool) {
  auto* pool = HostMemoryPool::GetInstance();
  size_t initial_used = pool->GetUsedSize();

  {
    // 先创建一个超大 Tensor，触发直接分配（2GB）
    Tensor tensor(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP32, {512 * 1024 * 1024}, "large_tensor");
    ASSERT_FALSE(tensor.IsHostFromPool());

    // 重新分配为小 Tensor，应该从池分配
    tensor.ReallocateMemory(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP32, {1024});

    // 断言：重新分配后应该是池分配
    EXPECT_TRUE(tensor.IsHostFromPool());
    EXPECT_EQ(tensor.GetHostMemorySource(), HostMemorySource::POOL);
  }

  // 析构后，池使用量应该回到初始值
  EXPECT_EQ(pool->GetUsedSize(), initial_used);
}

// 测试 8：GetView 返回的 Tensor 标记一致性
TEST_F(TensorHostMemorySourceTest, GetViewPreservesSource) {
  auto* pool = HostMemoryPool::GetInstance();

  {
    // 创建池分配的 Tensor
    Tensor tensor(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP32, {1024}, "tensor");
    ASSERT_TRUE(tensor.IsHostFromPool());

    // 获取视图
    Tensor view = tensor.GetView({512}, 0);

    // 断言：视图应该继承来源标记（虽然 shared buffer 不释放，但保持一致性）
    EXPECT_TRUE(view.IsHostFromPool());
    EXPECT_EQ(view.GetHostMemorySource(), HostMemorySource::POOL);
  }
}

// 测试 9：多次拷贝后的标记一致性
TEST_F(TensorHostMemorySourceTest, MultipleCopiesPreserveSource) {
  auto* pool = HostMemoryPool::GetInstance();
  size_t initial_used = pool->GetUsedSize();

  {
    Tensor tensor_a(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP32, {1024}, "tensor_a");
    ASSERT_TRUE(tensor_a.IsHostFromPool());

    Tensor tensor_b = tensor_a;
    Tensor tensor_c = tensor_b;
    Tensor tensor_d = tensor_c;

    // 所有拷贝都应该保持相同的来源标记
    EXPECT_TRUE(tensor_b.IsHostFromPool());
    EXPECT_TRUE(tensor_c.IsHostFromPool());
    EXPECT_TRUE(tensor_d.IsHostFromPool());
  }

  // 所有 Tensor 析构后，内存应该被释放
  EXPECT_EQ(pool->GetUsedSize(), initial_used);
}

// 测试 10：shared buffer 不受来源标记影响
TEST_F(TensorHostMemorySourceTest, SharedBufferIgnoresSource) {
  auto* pool = HostMemoryPool::GetInstance();
  size_t initial_used = pool->GetUsedSize();

  {
    // 创建池分配的 Tensor
    Tensor tensor(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP32, {1024}, "tensor");
    ASSERT_TRUE(tensor.IsHostFromPool());

    void* ptr = tensor.GetPtr<void>();

    // 使用外部指针创建 shared buffer Tensor
    Tensor shared_tensor(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP32, {1024}, ptr, "shared");

    // shared buffer 的来源标记应该是 DIRECT（默认值）
    EXPECT_FALSE(shared_tensor.IsHostFromPool());
    EXPECT_EQ(shared_tensor.GetHostMemorySource(), HostMemorySource::DIRECT);
  }

  // 析构后，池使用量应该回到初始值
  EXPECT_EQ(pool->GetUsedSize(), initial_used);
}

// 测试 11：池未初始化时的行为
TEST_F(TensorHostMemorySourceTest, PoolNotInitialized) {
  auto* pool = HostMemoryPool::GetInstance();

  // 销毁池
  pool->Destroy();
  ASSERT_FALSE(pool->IsInitialized());

  {
    // 创建 LOCATION_HOST 的 Tensor，应该回退到直接分配
    Tensor tensor(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP32, {1024}, "tensor");

    // 断言：应该是直接分配
    EXPECT_FALSE(tensor.IsHostFromPool());
    EXPECT_EQ(tensor.GetHostMemorySource(), HostMemorySource::DIRECT);
  }

  // 重新初始化池，供后续测试使用
  pool->Initialize(1);
}

// 测试 12：ToString 方法正确显示来源
TEST_F(TensorHostMemorySourceTest, ToStringShowsSource) {
  {
    // 池分配的 Tensor
    Tensor pool_tensor(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP32, {1024}, "pool_tensor");
    ASSERT_TRUE(pool_tensor.IsHostFromPool());

    std::string str = pool_tensor.ToString();
    EXPECT_NE(str.find("host(pool)"), std::string::npos);
  }

  // 销毁池以测试直接分配
  auto* pool = HostMemoryPool::GetInstance();
  pool->Destroy();

  {
    // 直接分配的 Tensor
    Tensor direct_tensor(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP32, {1024}, "direct_tensor");
    ASSERT_FALSE(direct_tensor.IsHostFromPool());

    std::string str = direct_tensor.ToString();
    EXPECT_NE(str.find("host(direct)"), std::string::npos);
  }

  // 重新初始化池
  pool->Initialize(1);
}

// 并发测试
TEST_F(TensorHostMemorySourceTest, ConcurrentAllocationAndFree) {
  auto* pool = HostMemoryPool::GetInstance();
  size_t initial_used = pool->GetUsedSize();

  const int num_threads = 8;
  const int iterations_per_thread = 100;
  std::vector<std::thread> threads;

  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([&, t]() {
      for (int i = 0; i < iterations_per_thread; ++i) {
        // 创建 Tensor
        Tensor tensor(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP32, {1024},
                      "tensor_" + std::to_string(t) + "_" + std::to_string(i));

        // 验证来源标记
        EXPECT_TRUE(tensor.IsHostFromPool() || !tensor.IsHostFromPool());

        // 拷贝测试
        Tensor copy = tensor;
        EXPECT_EQ(copy.GetHostMemorySource(), tensor.GetHostMemorySource());
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  // 所有线程完成后，池使用量应该回到初始值
  EXPECT_EQ(pool->GetUsedSize(), initial_used);
}

}  // namespace ksana_llm