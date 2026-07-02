/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <chrono>
#include <cstdint>
#include <map>
#include <unordered_map>
#include <vector>

#include "device/device_utils.h"

namespace ksana_llm {

// ──────────────────────────────────────────────────────────────────────────────
// CPU 时间戳工具（纳秒）
// ──────────────────────────────────────────────────────────────────────────────

inline int64_t ModelPerfNowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::high_resolution_clock::now().time_since_epoch())
      .count();
}

// ──────────────────────────────────────────────────────────────────────────────
// Profile 采集项 id 与注册表
// ──────────────────────────────────────────────────────────────────────────────

using ForwardProfileOpId = int32_t;

namespace ForwardProfileOp {
inline constexpr ForwardProfileOpId kInvalid = -1;
inline constexpr ForwardProfileOpId kAttn = 0;
inline constexpr ForwardProfileOpId kFfn = 1;
// 预留 2–15 给通用采集项；16+ 给各模型 Register()
}  // namespace ForwardProfileOp

class ForwardProfileOpRegistry {
 public:
  // 首次注册返回唯一 id；重复注册同名采集项返回 kInvalid 并打 ERROR 日志。
  static ForwardProfileOpId Register(const char* name);
  static const char* Name(ForwardProfileOpId id);
  static size_t RegisteredCount();
  static bool IsValid(ForwardProfileOpId id) { return id >= 0; }
};

// ──────────────────────────────────────────────────────────────────────────────
// 单类算子的累积计时结构
// ──────────────────────────────────────────────────────────────────────────────

struct LayerPerfMetrics {
  int64_t cumulative_time_ns = 0;  // 累积耗时（纳秒）
  int32_t count = 0;                 // 执行次数（= 层数）
};

// forward 类型标识（与 RunMode 一一对应，可扩展）
enum class ForwardMetricsKind : int {
  kMain = 0,         // RunMode::kMain — 正常 forward
  kSpeculative = 1,  // RunMode::kNextN — MTP / 投机解码 forward
};

// 单次 forward 的子 metrics
struct ForwardPerfMetrics {
  int64_t forward_time_ns = 0;
  std::unordered_map<ForwardProfileOpId, LayerPerfMetrics> ops;

  LayerPerfMetrics& GetOrCreateOp(ForwardProfileOpId id);
  const LayerPerfMetrics* GetOp(ForwardProfileOpId id) const;
  void Accumulate(ForwardProfileOpId id, int64_t delta_ns);
};

// ──────────────────────────────────────────────────────────────────────────────
// 单次 decode step 的完整性能指标
// ──────────────────────────────────────────────────────────────────────────────

struct ProfileMetrics {
  // 调度 dispatch 到首次 forward 开始的间隔（纳秒），仅首次 forward 写入。
  int64_t schedule_forward_interval_ns = 0;

  // 各 forward 类型的独立子 metrics（懒创建：仅实际执行的 forward 才写入 map）
  std::map<ForwardMetricsKind, ForwardPerfMetrics> forward_metrics;

  ForwardPerfMetrics& GetOrCreate(ForwardMetricsKind kind);
  const ForwardPerfMetrics* Get(ForwardMetricsKind kind) const;

  static const char* KindToString(ForwardMetricsKind kind);

  void SerializeTo(std::vector<int>& out) const;
  static size_t DeserializeFrom(const std::vector<int>& in, size_t idx, ProfileMetrics& m);

 private:
  static void SerializeI64(int64_t v, std::vector<int>& out);
  static size_t DeserializeI64(const std::vector<int>& in, size_t idx, int64_t& v);
  static void SerializeForwardPerfMetrics(const ForwardPerfMetrics& sub, std::vector<int>& out);
  static size_t DeserializeForwardPerfMetrics(const std::vector<int>& in, size_t idx, ForwardPerfMetrics& sub);
};

// ──────────────────────────────────────────────────────────────────────────────
// Profile 计时 helper 与宏
//
// metrics_ptr == nullptr 时宏完全空展开，零运行时开销。
// ──────────────────────────────────────────────────────────────────────────────

inline void ModelPerfOpEnd(ForwardPerfMetrics* metrics, ForwardProfileOpId profile_op_id, int64_t begin_ns) {
  if (metrics != nullptr && ForwardProfileOpRegistry::IsValid(profile_op_id)) {
    DeviceSynchronize();
    metrics->Accumulate(profile_op_id, ModelPerfNowNs() - begin_ns);
  }
}

inline void ModelPerfOpEndStream(ForwardPerfMetrics* metrics, ForwardProfileOpId profile_op_id, int64_t begin_ns,
                                 Stream& stream) {
  if (metrics != nullptr && ForwardProfileOpRegistry::IsValid(profile_op_id)) {
    StreamSynchronize(stream);
    metrics->Accumulate(profile_op_id, ModelPerfNowNs() - begin_ns);
  }
}

inline void ModelPerfOpEndNoSync(ForwardPerfMetrics* metrics, ForwardProfileOpId profile_op_id, int64_t begin_ns) {
  if (metrics != nullptr && ForwardProfileOpRegistry::IsValid(profile_op_id)) {
    metrics->Accumulate(profile_op_id, ModelPerfNowNs() - begin_ns);
  }
}

// NOLINTBEGIN(bugprone-macro-parentheses)
#define MODEL_PERF_OP_BEGIN(metrics_ptr, timer_var) \
  int64_t timer_var = 0;                            \
  if ((metrics_ptr) != nullptr) {                   \
    (timer_var) = ::ksana_llm::ModelPerfNowNs();    \
  }

#define MODEL_PERF_OP_END(metrics_ptr, profile_op_id, timer_var) \
  ::ksana_llm::ModelPerfOpEnd((metrics_ptr), (profile_op_id), (timer_var))

#define MODEL_PERF_OP_END_STREAM(metrics_ptr, profile_op_id, timer_var, stream) \
  ::ksana_llm::ModelPerfOpEndStream((metrics_ptr), (profile_op_id), (timer_var), (stream))

#define MODEL_PERF_OP_END_NOSYNC(metrics_ptr, profile_op_id, timer_var) \
  ::ksana_llm::ModelPerfOpEndNoSync((metrics_ptr), (profile_op_id), (timer_var))
// NOLINTEND(bugprone-macro-parentheses)

}  // namespace ksana_llm
