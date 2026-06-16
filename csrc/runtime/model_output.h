/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "runtime/schedule_output.h"

namespace ksana_llm {

class NewModelOutput {
 public:
  RunMode run_mode;

  // The schedule output.
  std::shared_ptr<ScheduleOutput> schedule_output;

  // hidden buffer for MTP, only used in sampling node
  HiddenUnitDeviceBuffer* hidden_unit;

  // 与对应 NewModelInput::slot_index 同义, 由 ModelExecutor::Forward 复制过来。
  // HandlePostprocess 调用 Sampler::Sampling(slot) 时透传, 保证 sampling 落到正确的 device buffer slot。
  // slow path 恒为 0。
  size_t slot_index = 0;

  // 本步 (req_id, last_offset) 配对 tensor 的 device 指针 + batch_size。
  // ModelExecutor::Forward 完成后从 ModelInstance::GetCurBatchPairs(slot) 固化；sampling 后调 WriteRing 写入 ring，
  // 供下一步 ReplaceLastTokenKernel 按 req_id 命中。慢路径保持 nullptr/0，WriteRing 自检 no-op。
  const int64_t* cur_pairs_dev = nullptr;
  int B_cur = 0;

  // BuildModelInput 时 ReorderInferRequests 写入的 logits_offset 快照，与 model_input->reqs 顺序一一对应。
  // ExecutorInferRequest 在 req cache 中跨 schedule 共享，后续 schedule 会覆写 req->logits_offset；
  // build-ahead 下本步 Sampling 可能晚于下一步 Preprocess，若直接读 req 字段会错位。
  // Forward 时从 model_input->reqs 固化到此，Sampling 只读此快照。
  std::vector<size_t> frozen_logits_offsets;
};

}  // namespace ksana_llm
