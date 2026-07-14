/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "executor/layer_progress_tracker.h"

#include <string.h>

#include "ipc/ipc_func_wrapper.h"

namespace ksana_llm {

// The current LayerProgressTracker may cause deadlocks. It will be temporarily disabled via an environment variable
// until the bug is resolved in a subsequent update.
const char* const enable_tracker_env = std::getenv("ENABLE_LAYER_TRACKER");
const bool kEnableTracker = (enable_tracker_env != nullptr && strcmp(enable_tracker_env, "1") == 0);

LayerProgressTracker::LayerProgressTracker() : monitor_thread_running_(false) {}

LayerProgressTracker::~LayerProgressTracker() { Cleanup(); }

void LayerProgressTracker::Initialize(int device_id, int max_layers) {
  if (!kEnableTracker) {
    return;
  }

  Cleanup();
  std::lock_guard<std::mutex> lock(mutex_);

  // 保存当前管理的 device_id
  device_id_ = device_id;

  // 只为当前 executor 管理的 device 创建 event
  SetDevice(device_id);
  events_vector_.resize(max_layers);
  for (int layer_index = 0; layer_index < max_layers; ++layer_index) {
    EventInfo& event_info = events_vector_[layer_index];
    EventCreateWithFlags(&event_info.event, EVENT_DISABLE_TIMING);
    event_info.layer_index = layer_index;
    event_info.processed = true;  // 初始状态为已处理
  }

  // 启动监控线程
  if (!monitor_thread_running_) {
    monitor_thread_running_ = true;
    monitor_thread_ = std::thread(&LayerProgressTracker::MonitorEvents, this);
  }
  KLLM_LOG_INFO << "LayerProgressTracker initialized for device " << device_id;
}

void LayerProgressTracker::SetIpcFuncWrapper(std::shared_ptr<IpcFuncWrapper> ipc_func_wrapper) {
  ipc_func_wrapper_ = ipc_func_wrapper;
}

void LayerProgressTracker::Cleanup() {
  if (!kEnableTracker) {
    return;
  }

  // 先停止监控线程，避免死锁
  if (monitor_thread_running_) {
    monitor_thread_running_ = false;  // 这是一个简单的布尔值赋值，是原子操作
    if (monitor_thread_.joinable()) {
      monitor_thread_.join();  // 等待线程结束
    }
  }

  // 然后获取锁进行其他清理操作
  std::lock_guard<std::mutex> lock(mutex_);

  // 销毁所有 event
  for (auto& event_info : events_vector_) {
    EventDestroy(event_info.event);
  }
  events_vector_.clear();
}

void LayerProgressTracker::RecordLayerProgress(int device_id, int layer_index, Stream& stream) {
  if (!kEnableTracker) {
    return;
  }
  KLLM_LOG_DEBUG << "RecordLayerProgress device_id: " << device_id << " layer_index: " << layer_index;
  // 验证 device_id 是否匹配
  if (device_id != device_id_) {
    KLLM_LOG_ERROR << "Device ID mismatch: expected " << device_id_ << ", got " << device_id;
    return;
  }

  std::lock_guard<std::mutex> lock(mutex_);

  if (layer_index >= 0 && layer_index < static_cast<int>(events_vector_.size())) {
    EventInfo& event_info = events_vector_[layer_index];
    // 记录 event
    EventRecord(event_info.event, stream);
    event_info.processed = false;  // 标记为未处理

    // 更新最大记录层索引
    if (layer_index > max_recorded_layer_) {
      max_recorded_layer_ = layer_index;
    }
  }
}

void LayerProgressTracker::MonitorEvents() {
  if (!kEnableTracker) {
    return;
  }

  while (monitor_thread_running_) {
    {
      std::lock_guard<std::mutex> lock(mutex_);

      // 从当前完成的层数+1开始遍历到最大记录层
      int start_layer = current_layer_progress_ + 1;
      int end_layer = max_recorded_layer_;

      for (int layer_index = start_layer; layer_index <= end_layer; ++layer_index) {
        if (layer_index < 0 || layer_index >= static_cast<int>(events_vector_.size())) {
          break;
        }

        EventInfo& event_info = events_vector_[layer_index];

        // 如果事件未处理，查询事件状态
        if (!event_info.processed) {
          bool event_completed = EventQuery(event_info.event);
          KLLM_LOG_DEBUG << "MonitorEvents: device_id=" << device_id_ << ", layer_index=" << layer_index
                         << ", event_completed=" << event_completed;
          if (event_completed) {
            // event 已完成，更新进度并通知
            current_layer_progress_ = layer_index;
            event_info.processed = true;

            // 通过 IPC 发送回调通知到 engine 进程
            if (ipc_func_wrapper_) {
              KLLM_LOG_DEBUG << "ReportLayerProgress: device_id=" << device_id_ << ", layer_index=" << layer_index;
              ipc_func_wrapper_->ReportLayerProgress(device_id_, layer_index);
            }
          } else {
            // 事件未完成，由于是按顺序查询，后续层也不会完成，直接退出循环
            break;
          }
        }
      }
    }

    // 让出当前线程的时间片，避免过度消耗 CPU，同时提高响应及时性
    std::this_thread::yield();
  }
}

void LayerProgressTracker::ResetState() {
  if (!kEnableTracker) {
    return;
  }

  std::lock_guard<std::mutex> lock(mutex_);

  // 在重置前，将所有未上报的层触发上报
  if (ipc_func_wrapper_ && max_recorded_layer_ >= 0) {
    for (int layer_index = current_layer_progress_ + 1; layer_index <= max_recorded_layer_; ++layer_index) {
      if (layer_index >= 0 && layer_index < static_cast<int>(events_vector_.size())) {
        KLLM_LOG_DEBUG << "ResetState: Force report layer_index=" << layer_index;
        ipc_func_wrapper_->ReportLayerProgress(device_id_, layer_index);
      }
    }
  }

  // 重置层进度记录
  current_layer_progress_ = -1;
  max_recorded_layer_ = -1;

  // 重置所有事件的处理状态为已处理（不需要监控）
  for (auto& event_info : events_vector_) {
    event_info.processed = true;
  }
}

}  // namespace ksana_llm