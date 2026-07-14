/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "device/device_utils.h"
#include "utils/singleton.h"

namespace ksana_llm {

// Forward declaration
class IpcFuncWrapper;

// 用于跟踪单个device执行到的层位置
class LayerProgressTracker {
 public:
  struct EventInfo {
    Event event;
    int layer_index;
    bool processed;
  };

  // 初始化 LayerProgressTracker - 只初始化单个 device
  void Initialize(int device_id, int max_layers);

  // 设置 IPC 包装器（用于发送回调）
  void SetIpcFuncWrapper(std::shared_ptr<IpcFuncWrapper> ipc_func_wrapper);

  // 清理资源
  void Cleanup();

  // 记录层执行进度（由 attention 层调用）
  void RecordLayerProgress(int device_id, int layer_index, Stream& stream);

  // 重置当前状态，清除所有进度记录，但保留 CUDA event 以便复用
  void ResetState();

  // 使用 Singleton 类实现单例模式，需要将构造函数和析构函数设为公有
  LayerProgressTracker();
  ~LayerProgressTracker();

  // 声明 Singleton 为友元类，使其能够访问私有构造函数
  friend class Singleton<LayerProgressTracker>;

  // 监控线程函数，用于等待 event 完成并通知
  void MonitorEvents();

 private:
  std::mutex mutex_;
  int current_layer_progress_ = -1;  // 当前完成的层索引
  int max_recorded_layer_ = -1;      // 记录的最大层索引

  // 当前管理的 device_id
  int device_id_ = -1;

  // IPC 包装器，用于发送层进度回调到 engine
  std::shared_ptr<IpcFuncWrapper> ipc_func_wrapper_;

  // 持久的 CUDA event 管理 - 使用 vector 存储，索引即为 layer_index
  std::vector<EventInfo> events_vector_;

  // 监控线程
  std::thread monitor_thread_;
  bool monitor_thread_running_;
};

}  // namespace ksana_llm
