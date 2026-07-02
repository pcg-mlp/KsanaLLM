/* Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * ModelPerfBackend：基于配置文件驱动固定 batch 执行的性能采集调度后端。
 *
 * 激活条件：ENABLE_MODEL_PERF=1 + MODEL_PERF_CONFIG=<path>
 * 架构：直接实现 ScheduleBackendInterface，与 ContinuousBatchingBackend 无任何继承或组合关系。
 *
 * 调度流程：
 *   inference_engine::Start() → ModelPerfBackend::StartPerf()（阻塞）→ 写出结果后进程退出
 *     for each batch:
 *       1. 构造 InferRequest 列表（含共享 batch_waiter），一次性提交给调度器
 *       2. Schedule() 负责多轮（warmup + stat）调度：
 *            - 第 0 轮：分配 KV blocks，保存快照，加入 running_reqs
 *            - 后续轮（needs_reset_=true）：恢复快照，加入 running_reqs
 *       3. UpdateWithGenerationResult() 在每轮结束后：
 *            - 清理 inflight 状态，更新 KV cache token 追踪
 *            - 若未完成所有轮：设置 needs_reset_=true，等待下轮 Schedule
 *            - 若已完成所有轮：收集 stat_metrics，发出 kStopRequest
 *       4. batch_waiter->Wait() 返回（kStopRequest → Notify()）
 *       5. 写出结果 JSON
 *
 * ==============================================================================*/
#pragma once

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "batch_scheduler/event_driven_scheduler/schedule_backend_interface.h"
#include "runtime/profile_metrics.h"

namespace ksana_llm {

// ─────────────────────────────────────────────────────────────────────────────
// 配置结构
// ─────────────────────────────────────────────────────────────────────────────

enum class ModelPerfOutputLevel {
  kSummary,  // 仅输出多轮汇总（均值/标准差）
  kDetail,   // 输出汇总 + 每轮原始数据
};

struct MetricMeanStddev {
  double mean = 0.0;
  double stddev = 0.0;  // 总体标准差；round_count <= 1 时为 0
};

struct OpPerfMetricsSummary {
  MetricMeanStddev cumulative_time_ns;
  MetricMeanStddev count;
};

struct ForwardPerfMetricsSummary {
  MetricMeanStddev forward_time_ns;
  std::map<std::string, OpPerfMetricsSummary> ops;
};

struct ProfileMetricsSummary {
  size_t round_count = 0;
  MetricMeanStddev schedule_forward_interval_ns;
  std::map<ForwardMetricsKind, ForwardPerfMetricsSummary> forward_metrics;
};

struct ModelPerfConfig {
  std::string output_file;
  std::string metrics_granularity = "batch";  // 预留，当前仅支持 "batch"
  ModelPerfOutputLevel output_level = ModelPerfOutputLevel::kDetail;
  // input_tokens 生成的全局种子；各请求在此基础叠加偏移，保证可复现且互不重复。
  uint32_t token_seed = 42;
};

struct PerfRequestSpec {
  size_t input_token_len = 0;
  size_t query_token_num = 0;
  int req_num = 1;
};

struct PerfDpBatchSpec {
  size_t dp_id = 0;
  std::vector<PerfRequestSpec> requests;
};

struct PerfBatchSpec {
  std::string batch_id;
  int warmup_rounds = 0;
  int stat_rounds = 1;
  std::vector<PerfDpBatchSpec> dp_batches;
};

// ─────────────────────────────────────────────────────────────────────────────
// 单个 perf 请求的状态跟踪（Schedule/UpdateWithGenerationResult 共享）
// ─────────────────────────────────────────────────────────────────────────────

struct PerfReqState {
  // 原始 spec
  size_t input_token_len = 0;
  size_t query_token_num = 0;
  size_t dp_idx = 0;

  // 第 0 轮 KV 分配完毕后保存的快照，用于后续轮恢复
  bool kv_allocated = false;
  size_t snap_kv_cached_token_num = 0;
  ScheduleTaskWorkload snap_planning_workload;
  ScheduleTaskWorkload snap_remaining_workload;
};

// ─────────────────────────────────────────────────────────────────────────────
// ModelPerfBackend
// ─────────────────────────────────────────────────────────────────────────────

class EventDrivenScheduleProcessor;  // 前向声明

class ModelPerfBackend : public ScheduleBackendInterface {
 public:
  ModelPerfBackend(const BatchSchedulerConfig& batch_scheduler_config, const std::string& config_path,
                   uint32_t vocab_size, size_t block_token_num);
  ~ModelPerfBackend() override = default;

  // 由 inference_engine 在 SetScheduleBackend 之后调用，供 StartPerf 使用。
  void SetScheduleProcessor(std::shared_ptr<EventDrivenScheduleProcessor> processor);

  // 性能测试主入口：顺序执行所有 batch，阻塞直到所有 batch 完成并写出结果。
  // 由 inference_engine::Start() 替代 DoWarmupRun() 调用。
  void StartPerf();

  // 禁用 PreSchedule 中的 timeout / abort 扫描（ModelPerfBackend 自行控制请求生命周期）
  void PreSchedule(const ScheduleContext& context,
                   const std::vector<std::shared_ptr<PlanningBatch>>& planning_batches,
                   std::vector<ScheduleBackendEvent>& events) override {}

  // 从 prefilling_reqs 取 perf 请求：
  //   第 0 轮：分配 KV blocks，保存快照，加入 running_reqs
  //   后续轮（needs_reset_=true）：恢复快照，加入 running_reqs
  void Schedule(const ScheduleContext& context, std::vector<std::shared_ptr<PlanningBatch>>& planning_batches,
                std::vector<ScheduleBackendEvent>& events) override;

  // 每轮执行完毕后：清理 inflight 状态，决定是否进入下一轮或发出 kStopRequest
  void UpdateWithGenerationResult(size_t multi_batch_id, const GenerationOutputGroup& generation_output,
                                  const ScheduleContext& context,
                                  std::vector<ScheduleBackendEvent>& events) override;

  void Stop() override;

  // 计算 batch 内所有请求的 query_token_num 总和（含 req_num 展开）。
  static size_t ComputeBatchQueryTokenSum(const PerfBatchSpec& batch_spec);

  // 对多轮 stat 指标计算均值与总体标准差。
  static ProfileMetricsSummary ComputeProfileMetricsSummary(const std::vector<ProfileMetrics>& metrics);

 private:
  // 解析 config_path 中的 JSON 配置（失败则 KLLM_THROW）
  void LoadConfig(const std::string& config_path);

  // 为单个请求分配 KV blocks（两步：prefix + query）。
  // prefix_len = input_token_len，total_token_len = input_token_len + query_token_num。
  // 返回 false 表示 OOM 或其他分配失败（已打印日志）。
  bool AllocateKvBlocksForReq(const std::shared_ptr<InferRequest>& req, size_t prefix_len, size_t total_token_len,
                              CacheManagerInterface& cache_manager);

  // 写出聚合结果 JSON 文件
  void Finalize();

  // ── 配置 ───────────────────────────────────────────────────────────────────
  ModelPerfConfig config_;
  std::vector<PerfBatchSpec> batches_;
  uint32_t vocab_size_ = 0;
  size_t block_token_num_ = 0;

  // ── 调度器引用 ─────────────────────────────────────────────────────────────
  std::weak_ptr<EventDrivenScheduleProcessor> processor_;
  std::atomic<bool> stopped_{false};

  // ── 当前 batch 的多轮状态（调度器线程与 StartPerf 线程协同）──────────────
  // 当前 batch 的总轮数（warmup + stat）
  int total_rounds_ = 0;
  // 当前已执行的轮次（每次 UpdateWithGenerationResult 完成后递增）
  int round_idx_ = 0;
  // 当前 batch 的 stat 轮数起始索引（= warmup_rounds）
  int stat_start_round_ = 0;
  // stat 阶段耗时统计（受 req_spec_mutex_ 保护）
  int64_t stat_start_ms_ = 0;
  int64_t stat_end_ms_ = 0;
  // 是否需要在下次 Schedule 中恢复快照（非首轮时由 UpdateWithGenerationResult 设置）
  bool needs_reset_ = false;

  // req_id → PerfReqState（Schedule 写入，UpdateWithGenerationResult 读取）
  // 受 req_spec_mutex_ 保护（StartPerf 线程写，Schedule/Update 在调度器线程读写）
  std::mutex req_spec_mutex_;
  std::unordered_map<int64_t, PerfReqState> perf_req_states_;

  // 当前 batch 在 stat 轮中采集到的 ProfileMetrics
  std::vector<ProfileMetrics> current_stat_metrics_;

  // ── 最终结果 ────────────────────────────────────────────────────────────────
  struct BatchResult {
    std::string batch_id;
    std::vector<ProfileMetrics> stat_metrics;
  };
  std::vector<BatchResult> results_;
};

}  // namespace ksana_llm
