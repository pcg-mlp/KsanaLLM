/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <vector>
#include <queue>

#include "numerous_llm/block_manager/block_manager.h"
#include "numerous_llm/utils/singleton.h"

namespace numerous_llm {

// Get block pointer.
template <typename T>
std::vector<T*> GetBlockPtrs(const std::vector<int>& blocks) {
  std::vector<void*> addrs;
  Singleton<BlockManager>::GetInstance()->GetBlockPtrs(blocks, addrs);
  return addrs;
}

// 定义一个唯一ID生成器类
class UniqueIDGenerator {
 public:
  // 构造函数，初始化唯一ID生成器
  UniqueIDGenerator();

  // 获取一个唯一ID
  int64_t GetUniqueID();

  // 回收一个ID，将其重新加入ID池
  void RecycleID(int64_t id);

 private:
  // 当前的唯一ID
  int64_t current_id;

  // 已回收的ID队列，用于复用
  std::queue<int64_t> recycled_ids;

  // 当前活动的ID集合，用于快速查找
  std::unordered_set<int64_t> active_ids;

  // 互斥锁，用于多线程安全
  std::mutex mutex_;
};

}  // namespace numerous_llm
