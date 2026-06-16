/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstdint>
#include <unordered_map>

#include "runtime/model_instance.h"
#include "runtime/schedule_output.h"

namespace ksana_llm {

class NewModelInput {
 public:
  RunMode run_mode;

  // The schedule output.
  std::shared_ptr<ScheduleOutput> schedule_output;

  std::vector<ForwardRequest> reqs;

  // ModelInput device buffer 槽位编号（慢路径恒为 0；build-ahead 在 0/1 间 ping-pong，每步 cur_ring_slot_ ^= 1u）。
  // ForwardingContext 按 slot_index 选用对应 ModelInput 实例，避免相邻 step 的 device tensor 竞争；
  // slot 之间物理隔离，不复用 device buffer。
  size_t slot_index = 0;

  // 上一步 sampling 写入的 (req_id, token) 配对 ring slot 起始指针（含 count 头部）。
  // ModelRunner::HandleExecute 从 sampler->GetRingSlot(cur_slot ^ 1u) 注入；
  // CommonModel::Forward 见 prev_ring_dev != nullptr 即 launch ReplaceLastTokenKernel（按 req_id 命中回填
  // placeholder）。 新 req / 首步 ring count==0 时 kernel 自动 no-op。慢路径恒为 nullptr；生命周期由 Sampler 管理。
  const int64_t* prev_ring_dev = nullptr;

  // true 表示 ParseFromRequests 已在 Preprocess 线程完成 (双 H2D stream 流水线); Execute 仅做 metadata 更新。
  bool gpu_input_ready = false;
};

}  // namespace ksana_llm
