/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/block_manager/block_manager.h"

#include <memory>
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/singleton.h"
#include "test.h"

using namespace ksana_llm;

// 定义一个 BlockManagerTest 类，继承自 testing::Test
class BlockManagerTest : public testing::Test {
 protected:
  // 在每个测试用例执行之前调用的函数
  void SetUp() override {
    // 创建一个 BlockManagerConfig 对象，用于配置 BlockManager
    BlockManagerConfig block_manager_config;
    block_manager_config.host_allocator_config.blocks_num = 2;
    block_manager_config.host_allocator_config.block_size = 1024;
    block_manager_config.host_allocator_config.device = MEMORY_HOST;
    block_manager_config.device_allocator_config.blocks_num = 2;
    block_manager_config.device_allocator_config.block_size = 1024;
    block_manager_config.device_allocator_config.device = MEMORY_DEVICE;

    GetDeviceCount(&device_num_);

    std::shared_ptr<Context> context =
        std::make_shared<Context>(/*tensor_parallel_size*/ device_num_, /*pipeline_parallel_size*/ 1);

    // 使用配置创建一个 BlockManager 对象
    block_manager = new BlockManager(block_manager_config, context);
    block_manager->PreAllocateBlocks();
  }

  // 在每个测试用例执行之后调用的函数
  void TearDown() override {
    // 删除 BlockManager 对象
    delete block_manager;
  }

 protected:
  // 定义一个 BlockManager 指针，用于在测试用例中使用
  BlockManager* block_manager;

  int device_num_ = -1;
};

// 定义一个测试用例，继承自 BlockManagerTest
TEST_F(BlockManagerTest, AllocateAndFree) {
  if (device_num_ != 2) {
    GTEST_SKIP_("This test is just for 2 GPUs");
  }

  // 创建一个整数向量，用于存储分配的内存块
  std::vector<int> blocks;

  // 尝试分配 2 个内存块
  Status status = block_manager->AllocateBlocks(2, blocks);

  // 检查分配是否成功
  EXPECT_TRUE(status.OK());

  // 检查分配的内存块数量是否正确
  EXPECT_EQ(blocks.size(), 2);

  // 尝试释放分配的内存块
  status = block_manager->FreeBlocks(blocks);

  // 检查释放是否成功
  EXPECT_TRUE(status.OK());

  // 检查释放后的空闲内存块数量是否正确
  EXPECT_EQ(block_manager->GetDeviceFreeBlockNumber(), 2);
}

// 测试 BlockManager 类的 AllocateAndFreeContiguousMemory 方法
TEST_F(BlockManagerTest, AllocateAndFreeContiguousMemory) {
  int block_id;
  int64_t size = 1024;

  // 分配指定大小的设备存储空间
  Status status = block_manager->AllocateContiguous(size, block_id);

  // 分配成功的情况下，状态应该是 OK
  EXPECT_TRUE(status.OK());
  // 分配成功后，block_id 应该大于等于 0
  EXPECT_GE(block_id, 0);

  // 获取分配的设备存储空间指针
  void* addr;
  status = block_manager->GetContiguousPtr(block_id, addr);
  // 未释放的情况下，状态应该是 OK
  EXPECT_TRUE(status.OK());
  EXPECT_NE(addr, nullptr);

  // 释放分配的设备存储空间
  status = block_manager->FreeContiguous(block_id);

  // 释放成功的情况下，状态应该是 OK
  EXPECT_TRUE(status.OK());

  // 获取分配的设备存储空间指针
  status = block_manager->GetContiguousPtr(block_id, addr);
  // 释放的情况下，状态应该不是 OK
  EXPECT_FALSE(status.OK());

  // 再次尝试释放已释放的设备存储空间
  status = block_manager->FreeContiguous(block_id);

  // 再次尝试释放已释放的设备存储空间时，状态应该不是 OK
  EXPECT_FALSE(status.OK());
}

TEST_F(BlockManagerTest, SwapInAndSwapOut) {
  if (device_num_ != 2) {
    GTEST_SKIP_("This test is just for 2 GPUs");
  }
  // 在 device 上分配两个 block
  std::vector<int> device_blocks;
  Status status = block_manager->AllocateBlocks(2, device_blocks);
  EXPECT_TRUE(status.OK());
  EXPECT_EQ(device_blocks.size(), 2);

  // 获取 block 的指针
  std::vector<void*> addrs;
  block_manager->GetBlockPtrs(device_blocks, addrs);

  // 将数据从 host 复制到 block 中
  std::string string_a = "string_a";
  std::string string_b = "string_b";
  Memcpy(addrs[0], string_a.data(), string_a.size(), MEMCPY_HOST_TO_DEVICE);
  Memcpy(addrs[1], string_b.data(), string_b.size(), MEMCPY_HOST_TO_DEVICE);

  // 在 host 上分配两个 block
  std::vector<int> host_blocks;
  status = block_manager->AllocateHostBlocks(2, host_blocks);
  EXPECT_TRUE(status.OK());
  EXPECT_EQ(host_blocks.size(), 2);

  // 将 block 从 device 交换到 host
  status = block_manager->SwapOut(host_blocks[0], device_blocks[0]);
  status = block_manager->SwapOut(host_blocks[1], device_blocks[1]);
  EXPECT_TRUE(status.OK());

  // 修改 block 中的数据
  string_a = "string_x";
  string_b = "string_x";
  Memcpy(addrs[0], string_a.data(), string_a.size(), MEMCPY_HOST_TO_DEVICE);
  Memcpy(addrs[1], string_b.data(), string_b.size(), MEMCPY_HOST_TO_DEVICE);

  // 将 block 从 host 交换回 device
  status = block_manager->SwapIn(device_blocks[0], host_blocks[0]);
  status = block_manager->SwapIn(device_blocks[1], host_blocks[1]);
  EXPECT_TRUE(status.OK());

  // 获取 block 的指针
  block_manager->GetBlockPtrs(device_blocks, addrs);

  // 将 block 中的数据从 GPU 复制回 CPU
  Memcpy(string_a.data(), addrs[0], string_a.size(), MEMCPY_DEVICE_TO_HOST);
  Memcpy(string_b.data(), addrs[1], string_b.size(), MEMCPY_DEVICE_TO_HOST);

  // 检查数据是否正确
  EXPECT_EQ(string_a, "string_a");
  EXPECT_EQ(string_b, "string_b");

  EXPECT_TRUE(block_manager->FreeBlocks(device_blocks).OK());
}

// 定义一个测试用例，继承自 BlockManagerTest
TEST_F(BlockManagerTest, GetFreeBlockNumber) {
  if (device_num_ != 2) {
    GTEST_SKIP_("This test is just for 2 GPUs");
  }
  // 检查 CPU 和 GPU 的空闲内存块数量是否正确
  EXPECT_EQ(block_manager->GetHostFreeBlockNumber(), 1);
  EXPECT_EQ(block_manager->GetDeviceFreeBlockNumber(), 2);

  // 创建一个整数向量，用于存储分配的内存块
  std::vector<int> blocks;

  // 尝试分配 2 个内存块
  Status status = block_manager->AllocateBlocks(2, blocks);

  // 检查分配是否成功
  EXPECT_TRUE(status.OK());

  // 检查分配的内存块数量是否正确
  EXPECT_EQ(blocks.size(), 2);

  // 检查分配后的空闲内存块数量是否正确
  EXPECT_EQ(block_manager->GetHostFreeBlockNumber(), 1);
  EXPECT_EQ(block_manager->GetDeviceFreeBlockNumber(), 0);

  EXPECT_TRUE(block_manager->FreeBlocks(blocks).OK());
}
