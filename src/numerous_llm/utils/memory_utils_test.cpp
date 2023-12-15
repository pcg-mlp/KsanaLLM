/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <thread>
#include <vector>

#include "test.h"
#include "numerous_llm/utils/memory_utils.h"

namespace numerous_llm {

// 测试UniqueIDGenerator类的单线程唯一ID生成功能
TEST(UniqueIDGeneratorTest, SingleThreadedUniqueID) {
  // 创建一个UniqueIDGenerator实例
  UniqueIDGenerator generator;

  // 获取两个唯一ID
  int id1 = generator.GetUniqueID();
  int id2 = generator.GetUniqueID();

  // 检查两个ID是否不相等
  EXPECT_NE(id1, id2);
}

// 测试UniqueIDGenerator类的ID回收功能
TEST(UniqueIDGeneratorTest, IDRecycling) {
  // 创建一个UniqueIDGenerator实例
  UniqueIDGenerator generator;

  // 获取一个唯一ID
  int id1 = generator.GetUniqueID();

  // 回收ID
  generator.RecycleID(id1);

  // 再次获取一个唯一ID
  int id3 = generator.GetUniqueID();

  // 检查回收的ID和新获取的ID是否相等
  EXPECT_EQ(id1, id3);
}

// 测试UniqueIDGenerator类的多线程唯一ID生成功能
TEST(UniqueIDGeneratorTest, MultiThreadedUniqueID) {
  // 创建一个UniqueIDGenerator实例
  UniqueIDGenerator generator;

  // 创建1000个线程
  std::vector<std::thread> threads;
  std::vector<int> ids(1000);

  // 每个线程获取一个唯一ID，回收它，然后再次获取一个唯一ID
  for (int i = 0; i < 1000; i++) {
    threads.push_back(std::thread([&generator, &ids, i]() {
      ids[i] = generator.GetUniqueID();
      generator.RecycleID(ids[i]);
      ids[i] = generator.GetUniqueID();
    }));
  }

  // 等待所有线程完成
  for (auto& thread : threads) {
    thread.join();
  }

  // 对ID进行排序
  std::sort(ids.begin(), ids.end());

  // 检查排序后的ID是否互不相同
  for (int i = 0; i < 999; i++) {
    EXPECT_NE(ids[i], ids[i + 1]);
  }
}

}  // namespace numerous_llm
