/* Copyright 2026 Tencent Inc. All rights reserved.

==============================================================================*/

#pragma once

#include <atomic>
#include <cstddef>
#include <string>

#include "configure/model_config.h"
#include "configure/schedule_config_parser.h"

namespace ksana_llm {

// =============================================================================
// FastPathController — build-ahead 启动期 gate
// -----------------------------------------------------------------------------
// Executor 端提前构建 + device 侧 token 替换的总开关判定。
// Engine 进程配置加载完成后一次性做硬条件 check，缓存全局是否启用：
//   启用 → max_depth=2，所有 batch 走 build-ahead；Replace/WriteRing kernel 每步无条件 enqueue，
//         kernel 自检 placeholder / ring count=0 自动 no-op，行为幂等。
//   禁用 → 退化到原 enable_async 深度 1 路径。
//
// 不做 batch-level / per-req 运行时拒绝：schema 差异、req 缺席、混合 batch 等由 device kernel 自检处理；
// swap / MTP / draft 等安全约束在 InitializeAtStartup 启动期排除。
// TP>1 时 rank 0 WriteRing 后经 NCCL 将 ring slot 广播到组内各卡，下一步各卡本地 Replace。
//
// Engine 与 Executor 均在配置加载后调用 InitializeAtStartup；运行时统一读 IsEnabledAtStartup()。
// =============================================================================
class FastPathController {
 public:
  // 单例访问。
  static FastPathController& GetInstance();

  // 启动期 check，依据各项配置硬性筛选；任一不满足则禁用并写入 disable_reason。
  // 重复调用会覆盖之前的结果（一般只在 Engine 启动时调一次）。
  void InitializeAtStartup(const BatchSchedulerConfig& sched_cfg, const RuntimeConfig& runtime_cfg,
                           const ModelConfig& model_cfg);

  // 启动期 check 是否通过（所有硬条件均满足）。
  bool IsEnabledAtStartup() const { return enabled_at_startup_.load(std::memory_order_relaxed); }

  // 启动期被禁用时的原因（便于日志和诊断）；启用时为空字符串。
  const std::string& DisableReason() const { return disable_reason_; }

  // 强制重置（仅用于测试）。
  void ResetForTesting();

 private:
  FastPathController() = default;
  FastPathController(const FastPathController&) = delete;
  FastPathController& operator=(const FastPathController&) = delete;

  // 启动期 check 标志，原子访问以便多线程读取。
  std::atomic<bool> enabled_at_startup_{false};

  // 启动期 check 失败原因（仅 disabled 时非空）。
  std::string disable_reason_;
};

}  // namespace ksana_llm
