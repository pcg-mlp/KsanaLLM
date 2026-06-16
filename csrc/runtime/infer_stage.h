/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

namespace ksana_llm {

enum class InferStage {
  kContext,
  kDecode,
};

// build-ahead 占位 sentinel：在 step N 采样完成前提前 launch step N+1 时，尚未知的末位 token 用此值占位。
// Engine 侧写入 forwarding_tokens，Executor 侧 ReplaceLastTokenKernel 用上一步采样结果回填 input_ids。
// Engine（InferRequest）与 Executor（ModelBuilder）共用此常量。
inline constexpr int kFastPathPlaceholderTokenId = -1;

}  // namespace ksana_llm