/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "executor/layer_progress_tracker.h"

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>
#include "gtest/gtest.h"

#include "device/device_utils.h"

// Namespace alias for convenience
namespace ksana = ksana_llm;

class LayerProgressTrackerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // 在每个测试前初始化设备环境
    int device_id = 0;
    ksana_llm::SetDevice(device_id);
  }

  void TearDown() override {
    // 测试结束后同步设备
    ksana_llm::DeviceSynchronize();
  }

  // 辅助函数：创建一个流，指定设备ID
  ksana_llm::Stream CreateStream(int device_id) {
    return ksana_llm::Stream();  // 在指定设备上创建流
  }

  // 辅助函数：销毁流
  void DestroyStream(ksana_llm::Stream& stream) { stream.Destroy(); }
};

// 测试初始化和清理
TEST_F(LayerProgressTrackerTest, InitializeAndCleanup) {
  ksana::LayerProgressTracker tracker;

  // 测试初始化 - 现在只需要device_id和max_layers
  tracker.Initialize(0, 3);

  // 测试清理
  tracker.Cleanup();

  // 再次初始化，确保可以重复初始化
  tracker.Initialize(1, 2);

  // 再次初始化，确保可以多次重复初始化
  tracker.Initialize(1, 3);

  // 最后清理
  tracker.Cleanup();
}

// 测试单设备初始化
TEST_F(LayerProgressTrackerTest, SingleDeviceInitialization) {
  ksana::LayerProgressTracker tracker;

  // 测试为device 0初始化
  tracker.Initialize(0, 10);

  // 验证可以正常清理
  tracker.Cleanup();

  // 测试为device 1初始化
  tracker.Initialize(1, 20);

  tracker.Cleanup();
}

// 测试RecordLayerProgress基本功能
TEST_F(LayerProgressTrackerTest, RecordLayerProgressBasic) {
  ksana::LayerProgressTracker tracker;
  int device_id = 0;
  int max_layers = 5;

  tracker.Initialize(device_id, max_layers);

  // 创建流
  ksana::Stream stream = CreateStream(device_id);

  // 记录多个层的进度
  for (int layer_idx = 0; layer_idx < max_layers; ++layer_idx) {
    tracker.RecordLayerProgress(device_id, layer_idx, stream);
    ksana_llm::StreamSynchronize(stream);
  }

  // 等待一段时间确保事件被处理
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  // 清理
  DestroyStream(stream);
  tracker.Cleanup();
}

// 测试ResetState功能
TEST_F(LayerProgressTrackerTest, ResetStateBasic) {
  ksana::LayerProgressTracker tracker;
  int device_id = 0;
  int max_layers = 5;

  tracker.Initialize(device_id, max_layers);

  // 创建流
  ksana::Stream stream = CreateStream(device_id);

  // 记录一些层的进度
  tracker.RecordLayerProgress(device_id, 0, stream);
  tracker.RecordLayerProgress(device_id, 1, stream);
  tracker.RecordLayerProgress(device_id, 2, stream);
  ksana_llm::StreamSynchronize(stream);

  // 等待事件处理
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  // 重置状态
  tracker.ResetState();

  // 重置后应该可以继续记录
  tracker.RecordLayerProgress(device_id, 0, stream);
  ksana_llm::StreamSynchronize(stream);

  // 清理
  DestroyStream(stream);
  tracker.Cleanup();
}

// 测试SetIpcFuncWrapper
TEST_F(LayerProgressTrackerTest, SetIpcFuncWrapper) {
  ksana::LayerProgressTracker tracker;
  int device_id = 0;

  tracker.Initialize(device_id, 5);

  // 设置nullptr的IPC包装器（测试不崩溃）
  tracker.SetIpcFuncWrapper(nullptr);

  // 创建流并记录进度
  ksana::Stream stream = CreateStream(device_id);
  tracker.RecordLayerProgress(device_id, 0, stream);
  ksana_llm::StreamSynchronize(stream);

  // 清理
  DestroyStream(stream);
  tracker.Cleanup();
}

// 测试跨层记录
TEST_F(LayerProgressTrackerTest, RecordNonSequentialLayers) {
  ksana::LayerProgressTracker tracker;
  int device_id = 0;
  int max_layers = 10;

  tracker.Initialize(device_id, max_layers);

  // 创建流
  ksana::Stream stream = CreateStream(device_id);

  // 非顺序记录层进度
  tracker.RecordLayerProgress(device_id, 0, stream);
  tracker.RecordLayerProgress(device_id, 2, stream);
  tracker.RecordLayerProgress(device_id, 1, stream);
  tracker.RecordLayerProgress(device_id, 5, stream);
  tracker.RecordLayerProgress(device_id, 3, stream);

  ksana_llm::StreamSynchronize(stream);

  // 等待事件处理
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  // 清理
  DestroyStream(stream);
  tracker.Cleanup();
}

// 测试重复初始化同一设备
TEST_F(LayerProgressTrackerTest, RepeatedInitializationSameDevice) {
  ksana::LayerProgressTracker tracker;
  int device_id = 0;

  // 第一次初始化
  tracker.Initialize(device_id, 5);
  tracker.Cleanup();

  // 第二次初始化相同设备
  tracker.Initialize(device_id, 10);
  tracker.Cleanup();

  // 第三次初始化相同设备，不同层数
  tracker.Initialize(device_id, 3);
  tracker.Cleanup();
}

// 测试边界条件：layer_index为0
TEST_F(LayerProgressTrackerTest, RecordLayerZero) {
  ksana::LayerProgressTracker tracker;
  int device_id = 0;

  tracker.Initialize(device_id, 5);

  ksana::Stream stream = CreateStream(device_id);

  // 记录第0层
  tracker.RecordLayerProgress(device_id, 0, stream);
  ksana_llm::StreamSynchronize(stream);

  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  DestroyStream(stream);
  tracker.Cleanup();
}

// 测试边界条件：layer_index为最大值
TEST_F(LayerProgressTrackerTest, RecordMaxLayer) {
  ksana::LayerProgressTracker tracker;
  int device_id = 0;
  int max_layers = 5;

  tracker.Initialize(device_id, max_layers);

  ksana::Stream stream = CreateStream(device_id);

  // 记录最后一层
  tracker.RecordLayerProgress(device_id, max_layers - 1, stream);
  ksana_llm::StreamSynchronize(stream);

  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  DestroyStream(stream);
  tracker.Cleanup();
}

// 测试多次ResetState
TEST_F(LayerProgressTrackerTest, MultipleResetState) {
  ksana::LayerProgressTracker tracker;
  int device_id = 0;

  tracker.Initialize(device_id, 5);

  ksana::Stream stream = CreateStream(device_id);

  // 第一轮记录
  tracker.RecordLayerProgress(device_id, 0, stream);
  tracker.RecordLayerProgress(device_id, 1, stream);
  ksana_llm::StreamSynchronize(stream);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  // 第一次重置
  tracker.ResetState();

  // 第二轮记录
  tracker.RecordLayerProgress(device_id, 0, stream);
  ksana_llm::StreamSynchronize(stream);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  // 第二次重置
  tracker.ResetState();

  // 第三轮记录
  tracker.RecordLayerProgress(device_id, 0, stream);
  tracker.RecordLayerProgress(device_id, 1, stream);
  tracker.RecordLayerProgress(device_id, 2, stream);
  ksana_llm::StreamSynchronize(stream);

  DestroyStream(stream);
  tracker.Cleanup();
}
