/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace ksana_llm {

// 前向声明，隐藏实现细节
class DeviceInfoManagerImpl;

/**
 * @brief 存储 Decode 节点的DP组和总设备数信息
 *
 * 对于Decode节点，负责保存和管理该节点的DP组和总设备数信息，
 * 避免设备信息重复传输。
 * 对于Prefill节点，负责接收通过ZMQ传输的Decode节点的DP组和总设备数信息，
 * 以便在Prefill节点上正确地分配transfer task。
 */

class DeviceInfoManager {
 public:
  DeviceInfoManager();
  ~DeviceInfoManager();

  DeviceInfoManager(const DeviceInfoManager&) = delete;
  DeviceInfoManager& operator=(const DeviceInfoManager&) = delete;

  DeviceInfoManager(DeviceInfoManager&&) noexcept;
  DeviceInfoManager& operator=(DeviceInfoManager&&) noexcept;

  /**
   * @brief Prefill端专用：无锁查找设备信息
   * @param group_key 通信组键
   * @param device_info_pair 输出参数，返回找到的设备信息
   * @return true 如果找到，false 如果未找到
   */
  bool Find(const std::string& group_key, std::pair<int, int>& device_info_pair) const;

  /**
   * @brief Prefill端专用：查找设备信息，不存在则插入，存在且值不同则更新
   * @param group_key 通信组键
   * @param adp_num ADP数量
   * @param dev_total_num 总设备数量
   */
  void FindAndInsert(const std::string& group_key, int adp_num, int dev_total_num);

  /**
   * @brief Prefill端专用：查找设备信息，不存在则等待，存在则直接返回
   * @param group_key 通信组键
   * @param device_info_pair 输出参数，返回找到的设备信息
   */
  void FindAndWait(const std::string& group_key, std::pair<int, int>& device_info_pair);

 private:
  // 使用 Pimpl 模式隐藏实现细节
  std::unique_ptr<DeviceInfoManagerImpl> impl_;
};

}  // namespace ksana_llm
