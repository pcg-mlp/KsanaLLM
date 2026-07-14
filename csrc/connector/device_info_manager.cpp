/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "connector/device_info_manager.h"
#include <oneapi/tbb/concurrent_hash_map.h>
#include <chrono>
#include "utils/logger.h"

namespace ksana_llm {

// Pimpl 实现类，避免向上暴露TBB
class DeviceInfoManagerImpl {
 public:
  using const_accessor = tbb::concurrent_hash_map<std::string, std::pair<int, int>>::const_accessor;

  // key 为 kv_comm_group_key, value 为 <decode_dp_num, decode_total_device_num>
  tbb::concurrent_hash_map<std::string, std::pair<int, int>> decode_dev_config_map_;

  std::mutex wait_mutex_;
  std::condition_variable cv_;
};

DeviceInfoManager::DeviceInfoManager() : impl_(std::make_unique<DeviceInfoManagerImpl>()) {}

DeviceInfoManager::~DeviceInfoManager() = default;

DeviceInfoManager::DeviceInfoManager(DeviceInfoManager&&) noexcept = default;

DeviceInfoManager& DeviceInfoManager::operator=(DeviceInfoManager&&) noexcept = default;

bool DeviceInfoManager::Find(const std::string& group_key, std::pair<int, int>& device_info_pair) const {
  DeviceInfoManagerImpl::const_accessor accessor;
  if (impl_->decode_dev_config_map_.find(accessor, group_key)) {
    device_info_pair = accessor->second;
    return true;
  }
  return false;
}

void DeviceInfoManager::FindAndInsert(const std::string& group_key, int adp_num, int dev_total_num) {
  typename decltype(impl_->decode_dev_config_map_)::accessor accessor;
  bool is_new_insert = impl_->decode_dev_config_map_.insert(accessor, group_key);
  auto new_value = std::make_pair(adp_num, dev_total_num);

  if (is_new_insert) {
    accessor->second = new_value;
    impl_->cv_.notify_all();
    return;
  }

  if (accessor->second != new_value) {
    accessor->second = new_value;
  }

  impl_->cv_.notify_all();
}

void DeviceInfoManager::FindAndWait(const std::string& group_key, std::pair<int, int>& device_info_pair) {
  DeviceInfoManagerImpl::const_accessor accessor;
  if (impl_->decode_dev_config_map_.find(accessor, group_key)) {
    device_info_pair = accessor->second;
    return;
  }

  // 查找不到时加锁等待
  std::unique_lock<std::mutex> lock(impl_->wait_mutex_);
  impl_->cv_.wait(lock, [this, &group_key]() {
    DeviceInfoManagerImpl::const_accessor tmp_accessor;
    return impl_->decode_dev_config_map_.find(tmp_accessor, group_key);
  });

  // 等待返回后，再次查找获取数据
  DeviceInfoManagerImpl::const_accessor final_accessor;
  if (impl_->decode_dev_config_map_.find(final_accessor, group_key)) {
    device_info_pair = final_accessor->second;
  }
}

}  // namespace ksana_llm
