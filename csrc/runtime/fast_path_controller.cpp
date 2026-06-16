/* Copyright 2026 Tencent Inc. All rights reserved.

==============================================================================*/

#include "runtime/fast_path_controller.h"

#include "utils/logger.h"
#include "utils/string_utils.h"

namespace ksana_llm {

FastPathController& FastPathController::GetInstance() {
  static FastPathController instance;
  return instance;
}

namespace {

// Set disable_reason 并打日志。返回 false 便于 InitializeAtStartup 链式调用。
inline bool MarkDisabled(std::atomic<bool>& flag, std::string& slot, std::string reason) {
  flag.store(false, std::memory_order_relaxed);
  slot = std::move(reason);
  KLLM_LOG_INFO << "[FastPath] disabled at startup: " << slot;
  return false;
}

}  // namespace

void FastPathController::InitializeAtStartup(const BatchSchedulerConfig& sched_cfg, const RuntimeConfig& runtime_cfg,
                                             const ModelConfig& model_cfg) {
  // 默认置为禁用，所有 check 通过后再置为启用。
  enabled_at_startup_.store(false, std::memory_order_relaxed);
  disable_reason_.clear();

  // 0. 总开关
  if (!sched_cfg.enable_executor_build_ahead) {
    MarkDisabled(enabled_at_startup_, disable_reason_, "enable_executor_build_ahead=false (default off)");
    return;
  }

  // 1. enable_async 是 build-ahead 的前置依赖
  if (!sched_cfg.enable_async) {
    MarkDisabled(enabled_at_startup_, disable_reason_, "enable_async must be true");
    return;
  }

  // 2. 只适配 BatchScheduler + ScheduleProcessor，不动 EventDrivenScheduleProcessor
  if (sched_cfg.scheduler_type != SchedulerType::DEFAULT) {
    MarkDisabled(enabled_at_startup_, disable_reason_, "scheduler_type must be DEFAULT (BatchScheduler)");
    return;
  }

  // 3. TP 已支持（WriteRing 后 NCCL broadcast ring）；DP / EP / PP 仍排除。
  const auto& parallel = runtime_cfg.parallel_basic_config;
  if (parallel.attn_data_parallel_size != 1) {
    MarkDisabled(enabled_at_startup_, disable_reason_,
                 FormatStr("attn_data_parallel_size must be 1, got %zu", parallel.attn_data_parallel_size));
    return;
  }
  if (parallel.expert_parallel_size != 1) {
    MarkDisabled(enabled_at_startup_, disable_reason_,
                 FormatStr("expert_parallel_size must be 1, got %zu", parallel.expert_parallel_size));
    return;
  }
  if (runtime_cfg.max_pp_batch_num != 1) {
    MarkDisabled(enabled_at_startup_, disable_reason_,
                 FormatStr("max_pp_batch_num must be 1, got %zu", runtime_cfg.max_pp_batch_num));
    return;
  }

  // 4. 各类高级功能必须关闭（当前版本不支持与 build-ahead 叠加）
  if (sched_cfg.enable_speculative_decoding) {
    MarkDisabled(enabled_at_startup_, disable_reason_, "enable_speculative_decoding must be false");
    return;
  }
  if (sched_cfg.mtp_step_num != 0) {
    MarkDisabled(enabled_at_startup_, disable_reason_,
                 FormatStr("mtp_step_num must be 0, got %zu", sched_cfg.mtp_step_num));
    return;
  }
  if (sched_cfg.enable_xgrammar) {
    MarkDisabled(enabled_at_startup_, disable_reason_, "enable_xgrammar must be false");
    return;
  }
  if (sched_cfg.split_fuse_token_num != 0) {
    MarkDisabled(enabled_at_startup_, disable_reason_,
                 FormatStr("split_fuse_token_num must be 0, got %zu", sched_cfg.split_fuse_token_num));
    return;
  }
  if (runtime_cfg.enable_prefix_caching) {
    MarkDisabled(enabled_at_startup_, disable_reason_, "enable_prefix_caching must be false");
    return;
  }
  if (runtime_cfg.enable_flexible_caching) {
    MarkDisabled(enabled_at_startup_, disable_reason_, "enable_flexible_caching must be false");
    return;
  }
  if (runtime_cfg.separate_prefill_decode) {
    MarkDisabled(enabled_at_startup_, disable_reason_, "PD separation (separate_prefill_decode) not supported");
    return;
  }

  // 全部 check 通过 → 启用 (模型类型 / MoE / MLA / 多模态 不做启动期限制)
  enabled_at_startup_.store(true, std::memory_order_relaxed);
  disable_reason_.clear();
  KLLM_LOG_INFO << "[FastPath] enabled at startup (model_type=" << model_cfg.type
                << ", tp=" << parallel.tensor_parallel_size << ", dp=" << parallel.attn_data_parallel_size
                << ", max_depth=2 — all batches go through fast path)";
}

void FastPathController::ResetForTesting() {
  enabled_at_startup_.store(false, std::memory_order_relaxed);
  disable_reason_.clear();
}

}  // namespace ksana_llm
