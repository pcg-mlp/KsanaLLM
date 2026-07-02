/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "runtime/profile_metrics.h"

#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

#include "utils/logger.h"

namespace ksana_llm {

namespace {

std::vector<const char*>& ProfileOpNames() {
  static std::vector<const char*> names;
  return names;
}

std::mutex& ProfileOpNamesMutex() {
  static std::mutex mu;
  return mu;
}

// deque 保证 push_back 不使已有元素的 c_str() 失效；vector 扩容会悬空 names 中的指针。
std::deque<std::string>& ProfileOpOwnedNames() {
  static std::deque<std::string> owned;
  return owned;
}

void EnsureBuiltinProfileOpsRegistered() {
  auto& names = ProfileOpNames();
  if (names.empty()) {
    names.push_back("attn");
    names.push_back("ffn");
  }
}

constexpr char kPlaceholderOpName[] = "unknown";

// 反序列化时按 wire 上的 (op_id, name) 对齐 Engine 侧注册表，避免跨进程静态初始化顺序不一致。
void EnsureOpRegistered(ForwardProfileOpId op_id, const char* name) {
  if (!ForwardProfileOpRegistry::IsValid(op_id)) {
    return;
  }

  std::lock_guard<std::mutex> lock(ProfileOpNamesMutex());
  EnsureBuiltinProfileOpsRegistered();
  auto& names = ProfileOpNames();
  auto& owned = ProfileOpOwnedNames();

  const size_t target = static_cast<size_t>(op_id);
  if (target < names.size()) {
    if (std::strcmp(names[target], name) == 0) {
      return;
    }
    if (std::strcmp(names[target], kPlaceholderOpName) == 0) {
      owned.emplace_back(name);
      names[target] = owned.back().c_str();
      return;
    }
    KLLM_LOG_ERROR << "ForwardProfileOpRegistry id/name mismatch at id " << op_id << ": registry has \""
                   << names[target] << "\", wire has \"" << name << "\"";
    std::cout << "[Error] ForwardProfileOpRegistry id/name mismatch at id " << op_id << ": registry has \""
              << names[target] << "\", wire has \"" << name << "\"" << std::endl;
    return;
  }

  while (names.size() < target) {
    KLLM_LOG_ERROR << "ForwardProfileOpRegistry gap: expected id " << names.size() << ", got " << op_id
                   << " for name \"" << name << "\"";
    owned.emplace_back(kPlaceholderOpName);
    names.push_back(owned.back().c_str());
  }

  owned.emplace_back(name);
  names.push_back(owned.back().c_str());
}

void SerializeOpName(const char* name, std::vector<int>& out) {
  const int name_len = static_cast<int>(std::strlen(name));
  out.push_back(name_len);
  for (int i = 0; i < name_len; ++i) {
    out.push_back(static_cast<int>(static_cast<unsigned char>(name[i])));
  }
}

size_t DeserializeOpName(const std::vector<int>& in, size_t idx, std::string& name) {
  const int name_len = in[idx++];
  name.resize(static_cast<size_t>(name_len));
  for (int i = 0; i < name_len; ++i) {
    name[static_cast<size_t>(i)] = static_cast<char>(in[idx++]);
  }
  return idx;
}

}  // namespace

ForwardProfileOpId ForwardProfileOpRegistry::Register(const char* name) {
  std::lock_guard<std::mutex> lock(ProfileOpNamesMutex());
  EnsureBuiltinProfileOpsRegistered();
  auto& names = ProfileOpNames();
  for (size_t i = 0; i < names.size(); ++i) {
    if (std::strcmp(names[i], name) == 0) {
      KLLM_LOG_ERROR << "ForwardProfileOpRegistry::Register duplicate profile op name: " << name;
      return ForwardProfileOp::kInvalid;
    }
  }
  const ForwardProfileOpId id = static_cast<ForwardProfileOpId>(names.size());
  names.push_back(name);
  return id;
}

const char* ForwardProfileOpRegistry::Name(ForwardProfileOpId id) {
  static const char kUnknown[] = "unknown";
  if (!IsValid(id)) {
    return kUnknown;
  }
  std::lock_guard<std::mutex> lock(ProfileOpNamesMutex());
  EnsureBuiltinProfileOpsRegistered();
  const auto& names = ProfileOpNames();
  if (id >= names.size()) {
    return kUnknown;
  }
  return names[id];
}

size_t ForwardProfileOpRegistry::RegisteredCount() {
  std::lock_guard<std::mutex> lock(ProfileOpNamesMutex());
  EnsureBuiltinProfileOpsRegistered();
  return ProfileOpNames().size();
}

LayerPerfMetrics& ForwardPerfMetrics::GetOrCreateOp(ForwardProfileOpId id) {
  if (!ForwardProfileOpRegistry::IsValid(id)) {
    static LayerPerfMetrics discarded;
    return discarded;
  }
  return ops[id];
}

const LayerPerfMetrics* ForwardPerfMetrics::GetOp(ForwardProfileOpId id) const {
  const auto it = ops.find(id);
  if (it == ops.end()) {
    return nullptr;
  }
  return &it->second;
}

void ForwardPerfMetrics::Accumulate(ForwardProfileOpId id, int64_t delta_ns) {
  if (!ForwardProfileOpRegistry::IsValid(id)) {
    return;
  }
  LayerPerfMetrics& metrics = ops[id];
  metrics.cumulative_time_ns += delta_ns;
  metrics.count++;
}

ForwardPerfMetrics& ProfileMetrics::GetOrCreate(ForwardMetricsKind kind) {
  return forward_metrics[kind];
}

const ForwardPerfMetrics* ProfileMetrics::Get(ForwardMetricsKind kind) const {
  const auto it = forward_metrics.find(kind);
  if (it == forward_metrics.end()) {
    return nullptr;
  }
  return &it->second;
}

const char* ProfileMetrics::KindToString(ForwardMetricsKind kind) {
  switch (kind) {
    case ForwardMetricsKind::kMain:
      return "main";
    case ForwardMetricsKind::kSpeculative:
      return "speculative";
    default:
      return "unknown";
  }
}

void ProfileMetrics::SerializeI64(int64_t v, std::vector<int>& out) {
  out.push_back(static_cast<int>(static_cast<uint64_t>(v) >> 32));
  out.push_back(static_cast<int>(static_cast<uint32_t>(v)));
}

size_t ProfileMetrics::DeserializeI64(const std::vector<int>& in, size_t idx, int64_t& v) {
  const uint64_t hi = static_cast<uint32_t>(in[idx]);
  const uint64_t lo = static_cast<uint32_t>(in[idx + 1]);
  v = static_cast<int64_t>((hi << 32) | lo);
  return idx + 2;
}

void ProfileMetrics::SerializeForwardPerfMetrics(const ForwardPerfMetrics& sub, std::vector<int>& out) {
  SerializeI64(sub.forward_time_ns, out);
  out.push_back(static_cast<int>(sub.ops.size()));
  for (const auto& [op_id, metrics] : sub.ops) {
    out.push_back(static_cast<int>(op_id));
    SerializeOpName(ForwardProfileOpRegistry::Name(op_id), out);
    SerializeI64(metrics.cumulative_time_ns, out);
    out.push_back(metrics.count);
  }
}

size_t ProfileMetrics::DeserializeForwardPerfMetrics(const std::vector<int>& in, size_t idx, ForwardPerfMetrics& sub) {
  idx = DeserializeI64(in, idx, sub.forward_time_ns);
  sub.ops.clear();
  const int op_count = in[idx++];
  for (int i = 0; i < op_count; ++i) {
    const auto op_id = static_cast<ForwardProfileOpId>(in[idx++]);
    std::string name;
    idx = DeserializeOpName(in, idx, name);
    EnsureOpRegistered(op_id, name.c_str());
    LayerPerfMetrics metrics;
    idx = DeserializeI64(in, idx, metrics.cumulative_time_ns);
    metrics.count = in[idx++];
    sub.ops[op_id] = metrics;
  }
  return idx;
}

void ProfileMetrics::SerializeTo(std::vector<int>& out) const {
  SerializeI64(schedule_forward_interval_ns, out);
  out.push_back(static_cast<int>(forward_metrics.size()));
  for (const auto& [kind, sub] : forward_metrics) {
    out.push_back(static_cast<int>(kind));
    SerializeForwardPerfMetrics(sub, out);
  }
}

size_t ProfileMetrics::DeserializeFrom(const std::vector<int>& in, size_t idx, ProfileMetrics& m) {
  idx = DeserializeI64(in, idx, m.schedule_forward_interval_ns);
  m.forward_metrics.clear();
  const int map_size = in[idx++];
  for (int i = 0; i < map_size; ++i) {
    const auto kind = static_cast<ForwardMetricsKind>(in[idx++]);
    ForwardPerfMetrics sub;
    idx = DeserializeForwardPerfMetrics(in, idx, sub);
    m.forward_metrics[kind] = sub;
  }
  return idx;
}

}  // namespace ksana_llm
