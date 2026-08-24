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

  // batch scheduler 最大 step tokens，用于确定 staging pool 大小
  size_t max_step_tokens = 0;

  // block 配置，用于在 Initialize() 阶段提前初始化 staging pool（避免首次请求延迟）。
  // 若 block_byte_size > 0，Initialize() 会立即分配 staging buffer；否则回退到首次请求时延迟初始化。
  int block_token_num = 0;
  size_t block_byte_size = 0;

  // Put 热 key 时间窗（秒）：仅当同一 block key 在窗口内再次出现才允许 Put；首次一律跳过。
  // 0 = 关闭频率过滤（保持原行为）。例如 3600=1 小时，1800=30 分钟。
  int64_t put_hotkey_window_sec = 0;

  // ========== SWA (sliding window attention) 全局 cache 配置 ==========
  // SWA block 未压缩，字节数远大于主 KV block（DSv4 实测约 7.6 倍），全量存储会让
  // 全局 cache 占用翻倍，因此按「尾部 N 密存 + 前段等分中位点稀存」的策略选点。

  // N：尾部密集存储的 block 个数。
  int swa_tail_block_num = 4;

  // 前段（去掉尾部 N 之后）等分成多少份；默认 5。首尾两段不上传，只取中间 parts-2 份的中位点。
  int swa_front_equal_parts = 5;

  // SWA block 字节数，与主 KV 的 block_byte_size 不同，需要独立的 staging pool。
  // 由 executor 创建 connector 时按 SWA block allocator 的配置填充；0 表示未启用 SWA。
  size_t swa_block_byte_size = 0;

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
    result += "  max_step_tokens: " + std::to_string(max_step_tokens) + "\n";
    result += "  block_token_num: " + std::to_string(block_token_num) + "\n";
    result += "  block_byte_size: " + std::to_string(block_byte_size) + "\n";
    result += "  put_hotkey_window_sec: " + std::to_string(put_hotkey_window_sec) + "\n";
    result += "  swa_tail_block_num: " + std::to_string(swa_tail_block_num) + "\n";
    result += "  swa_front_equal_parts: " + std::to_string(swa_front_equal_parts) + "\n";
    result += "  swa_block_byte_size: " + std::to_string(swa_block_byte_size) + "\n";
    result += "}";
    return result;
  }
};

/**
 * @brief QueryAndFetch 的 SWA 侧输入上下文；nullptr 表示该请求不涉及 SWA。
 *
 * SWA block 存在与否决定了「命中点是否合法」：恢复计算时 attention 需要窗口内的
 * SWA 内容，且 DSv4 里 compress_ratio == 0 的层的全部持久状态只在 SWA block 里。
 * 因此命中边界必须落在「末 swa_window_block_num 个位置都有 SWA」的地方。
 */
struct GlobalCacheSwaContext {
  // 本地已分配的 SWA block 在设备上的指针，与 local_swa_block_idxs 一一对应。
  // 拉取到的全局 SWA 内容只能写进这些位置，因此它同时是「命中点有落点」的判据。
  std::vector<void*> swa_block_ptrs;

  // swa_block_ptrs[i] 对应的 FA block 全局索引，升序。
  std::vector<size_t> local_swa_block_idxs;

  size_t swa_block_byte_size = 0;

  // W = ceil(sliding_window_size / block_token_num)，与本地 prefix 匹配用的同一个公式。
  size_t swa_window_block_num = 1;

  // 命中长度上限（token 数），由 engine 按 seq_len - 1 传入。
  // 必须在求边界之前参与约束：事后再截断会把命中点推到没有拉 SWA 的位置上。
  size_t max_hit_token_num = 0;

  int64_t req_id = -1;
};

/**
 * @brief QueryAndFetch 的 SWA 侧输出。
 */
struct GlobalCacheSwaResult {
  // 对齐后的命中 token 数（block 对齐）。0 表示没有可用的全局命中。
  size_t swa_aligned_hit_len = 0;

  // 实际写入了全局 SWA 内容的 FA block 索引。
  std::vector<size_t> filled_block_idxs;
};

}  // namespace ksana_llm
