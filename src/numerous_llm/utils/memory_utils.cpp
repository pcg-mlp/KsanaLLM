/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/utils/memory_utils.h"

namespace numerous_llm {
// 构造函数，初始化唯一ID生成器
UniqueIDGenerator::UniqueIDGenerator() : current_id(0) {}

// 获取一个唯一ID
int64_t UniqueIDGenerator::GetUniqueID() {
  // 使用互斥锁保证线程安全
  std::lock_guard<std::mutex> lock(mutex_);

  int64_t id;
  // 如果有回收的ID，则复用
  if (!recycled_ids.empty()) {
    id = recycled_ids.front();
    recycled_ids.pop();
  }
  // 没有回收的ID，生成新的ID
  else {
    // 检查是否达到整数最大值，如果是则抛出异常
    if (current_id == std::numeric_limits<int64_t>::max()) {
      throw std::runtime_error("No more unique IDs available.");
    }
    id = current_id++;
  }

  // 将新生成的ID加入活动ID集合
  active_ids.insert(id);

  // 返回新生成的ID
  return id;
}

// 回收一个ID，将其重新加入ID池
void UniqueIDGenerator::RecycleID(int64_t id) {
  // 使用互斥锁保证线程安全
  std::lock_guard<std::mutex> lock(mutex_);

  // 从活动ID集合中移除
  if (active_ids.find(id) != active_ids.end()) {
    active_ids.erase(id);

    // 加入回收ID队列
    recycled_ids.push(id);
  } else {
    // 如果ID不存在于活动ID集合中，则抛出异常
    throw std::runtime_error("ID not found in active IDs set.");
  }
}

}  // namespace numerous_llm
