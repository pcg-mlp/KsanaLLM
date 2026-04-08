/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ksana_llm {

struct GlobalCacheConnectorConfig {
  // 后端类型，用于选择具体实现:
  //   "mooncake-mock" — 基于文件系统的 mock backend，用于本地测试
  //   "mooncake"      — 真实 Mooncake Store backend，基于 mooncake::Client
  std::string backend_type = "mooncake-mock";

  // 元数据服务器地址:
  //   - HTTP metadata server: "http://127.0.0.1:8080/metadata"
  //   - etcd metadata backend: "etcd://127.0.0.1:2379"
  std::string service_endpoint;

  int device_id = 0;
  int device_count = 1;

  int timeout_ms = 5000;
  int max_retry = 3;

  int fetch_thread_num = 4;

  // 角色配置:
  //   "prefill" — prefill 节点：计算完首 token(s) 后保存到全局 cache，仅卡 0 保存
  //   "decode"  — decode 节点：从全局 cache 拉取 KV + token(s)，仅卡 0 查询 token，不做 prefill 计算
  //   ""        — 单机模式：正常查询逻辑，不查询/保存 token
  std::string role;

  // ========== 真实 Mooncake Store 配置 (backend_type = "mooncake") ==========

  // 传输协议: "tcp" 或 "rdma"
  std::string protocol = "tcp";

  // RDMA 设备列表，多个设备用逗号分隔，例如 "mlx5_0,mlx5_1"
  // 首版使用 TCP 时可为空
  std::string rdma_devices;

  // Mooncake master server 地址
  std::string master_server_addr = "127.0.0.1:50051";

  // RealClient 全局 segment 大小 (bytes)
  size_t global_segment_size = 16 * 1024 * 1024;

  // 本地 buffer 大小 (bytes)
  size_t local_buffer_size = 16 * 1024 * 1024;

  bool IsDecodeRole() const { return role == "decode"; }
  bool IsPrefillRole() const { return role == "prefill"; }
  bool IsStandalone() const { return role.empty(); }

  std::string ToString() const {
    std::string result = "GlobalCacheConnectorConfig {\n";
    result += "  backend_type: " + backend_type + "\n";
    result += "  service_endpoint: " + service_endpoint + "\n";
    result += "  device_id: " + std::to_string(device_id) + "\n";
    result += "  device_count: " + std::to_string(device_count) + "\n";
    result += "  timeout_ms: " + std::to_string(timeout_ms) + "\n";
    result += "  max_retry: " + std::to_string(max_retry) + "\n";
    result += "  fetch_thread_num: " + std::to_string(fetch_thread_num) + "\n";
    result += "  role: " + (role.empty() ? "(standalone)" : role) + "\n";
    result += "  protocol: " + protocol + "\n";
    result += "  rdma_devices: " + (rdma_devices.empty() ? "(none)" : rdma_devices) + "\n";
    result += "  master_server_addr: " + master_server_addr + "\n";
    result += "  global_segment_size: " + std::to_string(global_segment_size) + "\n";
    result += "  local_buffer_size: " + std::to_string(local_buffer_size) + "\n";
    result += "}";
    return result;
  }
};

}  // namespace ksana_llm
