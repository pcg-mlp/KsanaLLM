/* Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * ModelPerfBackend 实现：基于配置文件驱动固定 batch 执行的性能采集调度后端。
 * ==============================================================================*/

#include "batch_scheduler/event_driven_scheduler/model_perf/model_perf_backend.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <set>

#include "nlohmann/json.hpp"

#include "batch_scheduler/event_driven_scheduler/event_driven_schedule_processor.h"
#include "batch_scheduler/event_driven_scheduler/model_perf/model_perf_token_utils.h"
#include "profiler/timer.h"
#include "utils/logger.h"
#include "utils/request.h"
#include "utils/status.h"
#include "utils/waiter.h"

namespace ksana_llm {

namespace {

void PrintPerfStdout(const std::string& msg) {
  std::cout << msg << std::endl;
  std::cout.flush();
}

void PrintPerfBatchStart(const std::string& msg) {
  std::cout << msg;
  std::cout.flush();
}

void PrintPerfBatchDone(const std::string& suffix) {
  std::cout << suffix << std::endl;
  std::cout.flush();
}

MetricMeanStddev ComputeMeanStddev(const std::vector<double>& values) {
  MetricMeanStddev result;
  const size_t n = values.size();
  if (n == 0) {
    return result;
  }
  double sum = 0.0;
  for (double v : values) {
    sum += v;
  }
  result.mean = sum / static_cast<double>(n);
  if (n > 1) {
    double sq_sum = 0.0;
    for (double v : values) {
      const double diff = v - result.mean;
      sq_sum += diff * diff;
    }
    result.stddev = std::sqrt(sq_sum / static_cast<double>(n));
  }
  return result;
}

MetricMeanStddev ComputeMeanStddevI64(const std::vector<int64_t>& values) {
  std::vector<double> as_double;
  as_double.reserve(values.size());
  for (int64_t v : values) {
    as_double.push_back(static_cast<double>(v));
  }
  return ComputeMeanStddev(as_double);
}

MetricMeanStddev ComputeMeanStddevInt(const std::vector<int>& values) {
  std::vector<double> as_double;
  as_double.reserve(values.size());
  for (int v : values) {
    as_double.push_back(static_cast<double>(v));
  }
  return ComputeMeanStddev(as_double);
}

nlohmann::json MetricMeanStddevToJson(const MetricMeanStddev& mv) {
  // 时间单位为纳秒、计数为整数，汇总结果输出为整数即可。
  return nlohmann::json{{"mean", static_cast<int64_t>(std::llround(mv.mean))},
                        {"stddev", static_cast<int64_t>(std::llround(mv.stddev))}};
}

nlohmann::json LayerPerfMetricsToJson(const LayerPerfMetrics& metrics) {
  return nlohmann::json{{"cumulative_time_ns", metrics.cumulative_time_ns}, {"count", metrics.count}};
}

nlohmann::json ForwardPerfMetricsToJson(const ForwardPerfMetrics& sub) {
  nlohmann::json sub_json;
  sub_json["forward_time_ns"] = sub.forward_time_ns;
  nlohmann::json ops_json = nlohmann::json::object();
  for (const auto& [op_id, metrics] : sub.ops) {
    ops_json[ForwardProfileOpRegistry::Name(op_id)] = LayerPerfMetricsToJson(metrics);
  }
  sub_json["ops"] = ops_json;
  return sub_json;
}

nlohmann::json ProfileMetricsToRoundJson(const ProfileMetrics& m) {
  nlohmann::json round_json;
  round_json["schedule_forward_interval_ns"] = m.schedule_forward_interval_ns;
  nlohmann::json forward_metrics_json = nlohmann::json::object();
  for (const auto& [kind, sub] : m.forward_metrics) {
    forward_metrics_json[ProfileMetrics::KindToString(kind)] = ForwardPerfMetricsToJson(sub);
  }
  round_json["forward_metrics"] = forward_metrics_json;
  return round_json;
}

nlohmann::json ForwardPerfMetricsSummaryToJson(const ForwardPerfMetricsSummary& summary) {
  nlohmann::json summary_json;
  summary_json["forward_time_ns"] = MetricMeanStddevToJson(summary.forward_time_ns);
  nlohmann::json ops_json = nlohmann::json::object();
  for (const auto& [name, op_summary] : summary.ops) {
    nlohmann::json op_json;
    op_json["cumulative_time_ns"] = MetricMeanStddevToJson(op_summary.cumulative_time_ns);
    op_json["count"] = MetricMeanStddevToJson(op_summary.count);
    ops_json[name] = op_json;
  }
  summary_json["ops"] = ops_json;
  return summary_json;
}

nlohmann::json ProfileMetricsSummaryToJson(const ProfileMetricsSummary& summary) {
  nlohmann::json summary_json;
  summary_json["round_count"] = summary.round_count;
  summary_json["schedule_forward_interval_ns"] = MetricMeanStddevToJson(summary.schedule_forward_interval_ns);
  nlohmann::json forward_metrics_json = nlohmann::json::object();
  for (const auto& [kind, sub_summary] : summary.forward_metrics) {
    forward_metrics_json[ProfileMetrics::KindToString(kind)] = ForwardPerfMetricsSummaryToJson(sub_summary);
  }
  summary_json["forward_metrics"] = forward_metrics_json;
  return summary_json;
}

ModelPerfOutputLevel ParseOutputLevel(const std::string& output_level) {
  if (output_level == "summary") {
    return ModelPerfOutputLevel::kSummary;
  }
  if (output_level == "detail") {
    return ModelPerfOutputLevel::kDetail;
  }
  KLLM_THROW(fmt::format("ModelPerfBackend: invalid output_level '{}', expected 'summary' or 'detail'", output_level));
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// 构造
// ─────────────────────────────────────────────────────────────────────────────
ModelPerfBackend::ModelPerfBackend(const BatchSchedulerConfig& batch_scheduler_config, const std::string& config_path,
                                   uint32_t vocab_size, size_t block_token_num)
    : ScheduleBackendInterface(batch_scheduler_config), vocab_size_(vocab_size), block_token_num_(block_token_num) {
  if (block_token_num_ == 0) {
    KLLM_THROW("ModelPerfBackend: block_token_num must be > 0");
  }
  if (vocab_size_ == 0) {
    KLLM_THROW("ModelPerfBackend: vocab_size from ModelConfig must be > 0");
  }
  LoadConfig(config_path);
  KLLM_LOG_INFO << "ModelPerfBackend initialized, batches=" << batches_.size()
                << ", output_file=" << config_.output_file
                << ", output_level=" << (config_.output_level == ModelPerfOutputLevel::kSummary ? "summary" : "detail")
                << ", vocab_size=" << vocab_size_;
}

// ─────────────────────────────────────────────────────────────────────────────
// LoadConfig：解析 model_perf_config.json 和 batch_config.json
// ─────────────────────────────────────────────────────────────────────────────
void ModelPerfBackend::LoadConfig(const std::string& config_path) {
  std::ifstream cfg_file(config_path);
  if (!cfg_file.is_open()) {
    KLLM_THROW(fmt::format("ModelPerfBackend: cannot open config '{}'", config_path));
  }
  nlohmann::json cfg_json;
  cfg_file >> cfg_json;

  config_.output_file = cfg_json.value("output_file", "model_perf_result.json");
  config_.metrics_granularity = cfg_json.value("metrics_granularity", "batch");
  config_.output_level = ParseOutputLevel(cfg_json.value("output_level", "detail"));
  config_.token_seed = cfg_json.value("token_seed", static_cast<uint32_t>(42));

  if (!cfg_json.contains("batch_config")) {
    KLLM_THROW("ModelPerfBackend: model_perf_config.json missing 'batch_config' field");
  }
  std::string batch_config_path = cfg_json["batch_config"].get<std::string>();

  std::ifstream batch_file(batch_config_path);
  if (!batch_file.is_open()) {
    KLLM_THROW(fmt::format("ModelPerfBackend: cannot open batch_config '{}'", batch_config_path));
  }
  nlohmann::json batch_json;
  batch_file >> batch_json;

  if (!batch_json.contains("batches")) {
    KLLM_THROW("ModelPerfBackend: batch_config.json missing 'batches' field");
  }

  const size_t max_step_token_num = batch_scheduler_config_.max_step_token_num;
  size_t skipped_batch_count = 0;

  for (const auto& b : batch_json["batches"]) {
    PerfBatchSpec spec;
    spec.batch_id = b.value("batch_id", "");
    spec.warmup_rounds = b.value("warmup_rounds", 0);
    spec.stat_rounds = b.value("stat_rounds", 1);

    for (const auto& dp : b["dp_batches"]) {
      PerfDpBatchSpec dp_spec;
      dp_spec.dp_id = dp.value("dp_id", static_cast<size_t>(0));
      for (const auto& req : dp["requests"]) {
        PerfRequestSpec rs;
        rs.input_token_len = req.value("input_token_len", static_cast<size_t>(1));
        rs.query_token_num = req.value("query_token_num", rs.input_token_len);
        rs.req_num = req.value("req_num", 1);
        dp_spec.requests.push_back(rs);
      }
      spec.dp_batches.push_back(dp_spec);
    }

    const size_t query_token_sum = ComputeBatchQueryTokenSum(spec);
    if (query_token_sum > max_step_token_num) {
      KLLM_LOG_WARNING << "ModelPerfBackend: skip batch '" << spec.batch_id
                       << "': total query_token_num=" << query_token_sum
                       << " exceeds max_step_token_num=" << max_step_token_num;
      ++skipped_batch_count;
      continue;
    }

    batches_.push_back(spec);
  }

  if (batches_.empty()) {
    if (skipped_batch_count > 0) {
      KLLM_THROW(fmt::format(
          "ModelPerfBackend: all {} batch(es) skipped because query_token_num sum exceeds max_step_token_num ({})",
          skipped_batch_count, max_step_token_num));
    }
    KLLM_THROW("ModelPerfBackend: batch_config.json has empty 'batches' list");
  }
  if (skipped_batch_count > 0) {
    KLLM_LOG_WARNING << "ModelPerfBackend: skipped " << skipped_batch_count
                     << " batch(es) due to max_step_token_num limit=" << max_step_token_num;
  }
  KLLM_LOG_INFO << "ModelPerfBackend: loaded " << batches_.size() << " batch spec(s)";
}

size_t ModelPerfBackend::ComputeBatchQueryTokenSum(const PerfBatchSpec& batch_spec) {
  size_t sum = 0;
  for (const auto& dp_spec : batch_spec.dp_batches) {
    for (const auto& req_spec : dp_spec.requests) {
      sum += req_spec.query_token_num * static_cast<size_t>(req_spec.req_num);
    }
  }
  return sum;
}

ProfileMetricsSummary ModelPerfBackend::ComputeProfileMetricsSummary(const std::vector<ProfileMetrics>& metrics) {
  ProfileMetricsSummary summary;
  summary.round_count = metrics.size();
  if (metrics.empty()) {
    return summary;
  }

  std::set<ForwardMetricsKind> kinds;
  std::set<ForwardProfileOpId> op_ids;
  for (const auto& metric : metrics) {
    for (const auto& [kind, sub] : metric.forward_metrics) {
      kinds.insert(kind);
      for (const auto& [op_id, _] : sub.ops) {
        op_ids.insert(op_id);
      }
    }
  }

  {
    std::vector<int64_t> schedule_forward_interval_ns;
    schedule_forward_interval_ns.reserve(metrics.size());
    for (const auto& metric : metrics) {
      schedule_forward_interval_ns.push_back(metric.schedule_forward_interval_ns);
    }
    summary.schedule_forward_interval_ns = ComputeMeanStddevI64(schedule_forward_interval_ns);
  }

  for (const ForwardMetricsKind kind : kinds) {
    std::vector<int64_t> forward_time_ns;

    for (const auto& metric : metrics) {
      const ForwardPerfMetrics* sub = metric.Get(kind);
      if (sub == nullptr) {
        continue;
      }
      forward_time_ns.push_back(sub->forward_time_ns);
    }

    ForwardPerfMetricsSummary& sub_summary = summary.forward_metrics[kind];
    sub_summary.forward_time_ns = ComputeMeanStddevI64(forward_time_ns);

    for (const ForwardProfileOpId op_id : op_ids) {
      std::vector<int64_t> cumulative_time_ns;
      std::vector<int> count;
      for (const auto& metric : metrics) {
        const ForwardPerfMetrics* sub = metric.Get(kind);
        if (sub == nullptr) {
          continue;
        }
        const LayerPerfMetrics* op_metrics = sub->GetOp(op_id);
        if (op_metrics == nullptr) {
          continue;
        }
        cumulative_time_ns.push_back(op_metrics->cumulative_time_ns);
        count.push_back(op_metrics->count);
      }
      if (cumulative_time_ns.empty()) {
        continue;
      }
      OpPerfMetricsSummary& op_summary = sub_summary.ops[ForwardProfileOpRegistry::Name(op_id)];
      op_summary.cumulative_time_ns = ComputeMeanStddevI64(cumulative_time_ns);
      op_summary.count = ComputeMeanStddevInt(count);
    }
  }
  return summary;
}

// ─────────────────────────────────────────────────────────────────────────────
// SetScheduleProcessor
// ─────────────────────────────────────────────────────────────────────────────
void ModelPerfBackend::SetScheduleProcessor(std::shared_ptr<EventDrivenScheduleProcessor> processor) {
  processor_ = processor;
}

// ─────────────────────────────────────────────────────────────────────────────
// StartPerf：每批 batch 提交一次请求，等待 Schedule 完成多轮后 kStopRequest 触发
// ─────────────────────────────────────────────────────────────────────────────
void ModelPerfBackend::StartPerf() {
  auto proc = processor_.lock();
  if (proc == nullptr) {
    KLLM_THROW("ModelPerfBackend::StartPerf: schedule processor has been destroyed");
  }

  PrintPerfStdout("[ModelPerf] Starting performance profiling, total_batches=" + std::to_string(batches_.size()) +
                  ", output_file=" + config_.output_file);

  for (size_t batch_idx = 0; batch_idx < batches_.size(); ++batch_idx) {
    if (stopped_.load()) break;

    const PerfBatchSpec& batch_spec = batches_[batch_idx];
    const int total_rounds = batch_spec.warmup_rounds + batch_spec.stat_rounds;

    KLLM_LOG_INFO << "ModelPerfBackend: starting batch '" << batch_spec.batch_id << "' (batch_idx=" << batch_idx
                  << ", warmup=" << batch_spec.warmup_rounds << ", stat=" << batch_spec.stat_rounds
                  << ", total_rounds=" << total_rounds << ")";

    // ── 计算本 batch 的请求总数 ───────────────────────────────────────────
    int total_req_count = 0;
    for (const auto& dp_spec : batch_spec.dp_batches) {
      for (const auto& req_spec : dp_spec.requests) {
        total_req_count += req_spec.req_num;
      }
    }

    // ── 初始化多轮状态（调度器线程通过这些字段决定每轮行为）──────────────
    {
      std::lock_guard<std::mutex> lock(req_spec_mutex_);
      total_rounds_ = total_rounds;
      round_idx_ = 0;
      stat_start_round_ = batch_spec.warmup_rounds;
      stat_start_ms_ = 0;
      stat_end_ms_ = 0;
      needs_reset_ = false;
      perf_req_states_.clear();
      current_stat_metrics_.clear();
    }

    // ── 构造 InferRequest 列表，一次性提交 ───────────────────────────────
    auto batch_waiter = std::make_shared<Waiter>(total_req_count);
    std::vector<std::shared_ptr<InferRequest>> batch_infer_reqs;

    // 持有 Request/KsanaPythonInput 的生命周期（StartPerf 持有直到 Wait 返回）
    std::vector<std::shared_ptr<KsanaPythonInput>> python_inputs;
    std::vector<std::shared_ptr<Request>> requests;

    {
      std::lock_guard<std::mutex> lock(req_spec_mutex_);

      for (const auto& dp_spec : batch_spec.dp_batches) {
        for (const auto& req_spec : dp_spec.requests) {
          for (int n = 0; n < req_spec.req_num; ++n) {
            // ── 初始化 KsanaPythonInput ─────────────────────────────────
            auto python_input = std::make_shared<KsanaPythonInput>();
            python_input->model_name = "perf_model";
            python_input->sampling_config.num_beams = 0;
            python_input->sampling_config.num_return_sequences = 1;
            python_input->sampling_config.max_new_tokens = 1;
            python_input->sampling_config.ignore_eos = true;
            python_input->sampling_config.topk = 1;
            const size_t total_token_len = req_spec.input_token_len + req_spec.query_token_num;
            const uint32_t req_seed =
                config_.token_seed ^
                static_cast<uint32_t>(batch_idx * 100000u + dp_spec.dp_id * 10000u + static_cast<size_t>(n));
            GenerateTokens(total_token_len, req_seed, vocab_size_, python_input->input_tokens);

            auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
            auto req = std::make_shared<Request>(python_input, req_ctx);
            req->waiter = batch_waiter;

            // ── 初始化 InferRequest ─────────────────────────────────────
            std::vector<std::shared_ptr<InferRequest>> infer_req_group;
            for (size_t i = 0; i < req->output_group.size(); ++i) {
              auto infer_req = std::make_shared<InferRequest>(req, i);
              infer_req->kv_cache_blocks.clear();
              infer_req->block_token_num = block_token_num_;
              infer_req->infer_stage = InferStage::kContext;
              infer_req->step = 0;
              infer_req->kv_cached_token_num = 0;
              infer_req->attn_dp_group_id = static_cast<uint32_t>(dp_spec.dp_id);
              infer_req_group.push_back(infer_req);
            }
            for (auto& ir : infer_req_group) {
              ir->SetReqGroup(infer_req_group);
            }

            // ── 注册 PerfReqState（供 Schedule 使用）─────────────────────
            for (auto& ir : infer_req_group) {
              PerfReqState state;
              state.input_token_len = req_spec.input_token_len;
              state.query_token_num = req_spec.query_token_num;
              state.dp_idx = dp_spec.dp_id;
              state.kv_allocated = false;
              perf_req_states_[ir->req_id] = state;
            }

            python_inputs.push_back(python_input);
            requests.push_back(req);
            for (auto& ir : infer_req_group) {
              batch_infer_reqs.push_back(ir);
            }
          }
        }
      }
    }

    // ── 通过事件队列一次性提交本 batch 的所有请求 ─────────────────────────
    proc->AddInferRequest(batch_infer_reqs);
    KLLM_LOG_INFO << "ModelPerfBackend: batch='" << batch_spec.batch_id << "' submitted " << batch_infer_reqs.size()
                  << " reqs, waiting for " << total_rounds << " round(s)";
    PrintPerfBatchStart("[ModelPerf] [" + std::to_string(batch_idx + 1) + "/" + std::to_string(batches_.size()) +
                        "] '" + batch_spec.batch_id + "' warmup=" + std::to_string(batch_spec.warmup_rounds) +
                        " stat=" + std::to_string(batch_spec.stat_rounds) + " reqs=" +
                        std::to_string(batch_infer_reqs.size()) + " running...");

    // ── 等待所有请求完成（kStopRequest 触发后 req->Notify() 递减计数）─────
    batch_waiter->Wait();
    KLLM_LOG_INFO << "ModelPerfBackend: batch='" << batch_spec.batch_id << "' all " << total_rounds << " round(s) done";
    int64_t stat_elapsed_ms = 0;
    {
      std::lock_guard<std::mutex> lock(req_spec_mutex_);
      if (stat_start_ms_ > 0 && stat_end_ms_ >= stat_start_ms_) {
        stat_elapsed_ms = stat_end_ms_ - stat_start_ms_;
      }
    }
    const int64_t per_round_ms = (batch_spec.stat_rounds > 0) ? (stat_elapsed_ms / batch_spec.stat_rounds) : 0;
    PrintPerfBatchDone(" done (total stat: " + std::to_string(stat_elapsed_ms) +
                       "ms, per round: " + std::to_string(per_round_ms) + "ms)");

    // ── 清理 perf_req_states_（新 batch 开始时已在 lock 内清理，此处再保险）
    {
      std::lock_guard<std::mutex> lock(req_spec_mutex_);
      perf_req_states_.clear();
    }

    // ── 保存本 batch 结果 ─────────────────────────────────────────────────
    BatchResult br;
    br.batch_id = batch_spec.batch_id;
    {
      std::lock_guard<std::mutex> lock(req_spec_mutex_);
      br.stat_metrics = std::move(current_stat_metrics_);
      current_stat_metrics_.clear();
    }
    results_.push_back(std::move(br));
  }

  KLLM_LOG_INFO << "ModelPerfBackend: all batches done, writing results to '" << config_.output_file << "'";
  PrintPerfStdout("[ModelPerf] All batches done, writing results to '" + config_.output_file + "'");
  Finalize();
  PrintPerfStdout("[ModelPerf] Performance profiling finished.");
  stopped_.store(true);
}

// ─────────────────────────────────────────────────────────────────────────────
// AllocateKvBlocksForReq：两步分配（prefix + query）
// ─────────────────────────────────────────────────────────────────────────────
bool ModelPerfBackend::AllocateKvBlocksForReq(const std::shared_ptr<InferRequest>& req, size_t prefix_len,
                                              size_t total_token_len, CacheManagerInterface& cache_manager) {
  req->kv_cache_blocks.clear();
  req->swa_kv_cache_blocks.clear();
  req->swa_kv_cache_block_idx_offsets.clear();
  req->kv_cached_token_num = 0;

  // 向 cache manager 注册本请求
  {
    size_t shared_block_num = 0, unique_block_num = 0, shared_token_num = 0;
    const std::vector<int>& prefill_toks = req->GetPrefillingTokens();
    Status st = cache_manager.GetRequestPrefixBlockNumber(req->req_id, prefill_toks, prefill_toks.size(),
                                                          shared_block_num, unique_block_num, shared_token_num);
    if (!st.OK()) {
      KLLM_LOG_ERROR << "ModelPerfBackend: GetRequestPrefixBlockNumber failed for req_id=" << req->req_id << ": "
                     << st.GetMessage();
      return false;
    }
  }

  // Step 1：分配 prefix 部分的 blocks
  if (prefix_len > 0) {
    const size_t prefix_block_num = cache_manager.GetRequestStepBlockNumber(req->req_id, prefix_len);
    if (prefix_block_num > 0) {
      Status st = cache_manager.AllocateRequestBlocks(req->req_id, prefix_block_num, req->kv_cache_blocks,
                                                      req->swa_kv_cache_blocks, req->swa_kv_cache_block_idx_offsets,
                                                      req->mamba_kv_cache_block_id);
      if (!st.OK()) {
        KLLM_LOG_ERROR << "ModelPerfBackend: failed to alloc prefix blocks for req_id=" << req->req_id
                       << ", prefix_len=" << prefix_len << ": " << st.GetMessage();
        return false;
      }
    }
    req->kv_cached_token_num = prefix_len;
  }

  // Step 2：分配 query 部分的 blocks（total_token_len 为 prefix + query 总长）
  const size_t total_block_num = cache_manager.GetRequestStepBlockNumber(req->req_id, total_token_len);
  if (total_block_num > 0) {
    Status st = cache_manager.AllocateRequestBlocks(req->req_id, total_block_num, req->kv_cache_blocks,
                                                    req->swa_kv_cache_blocks, req->swa_kv_cache_block_idx_offsets,
                                                    req->mamba_kv_cache_block_id);
    if (!st.OK()) {
      KLLM_LOG_ERROR << "ModelPerfBackend: failed to alloc query blocks for req_id=" << req->req_id
                     << ", total_token_len=" << total_token_len << ": " << st.GetMessage();
      return false;
    }
  }

  return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Schedule：多轮调度核心
//   - 第 0 轮（!kv_allocated）：分配 KV blocks，保存快照，加入 running_reqs
//   - 后续轮（needs_reset_=true）：恢复快照，加入 running_reqs
// ─────────────────────────────────────────────────────────────────────────────
void ModelPerfBackend::Schedule(const ScheduleContext& context,
                                std::vector<std::shared_ptr<PlanningBatch>>& planning_batches,
                                std::vector<ScheduleBackendEvent>& events) {
  events.clear();
  if (stopped_.load()) return;

  KLLM_CHECK(!planning_batches.empty());
  PlanningBatch& planning_batch = *planning_batches[0];
  const size_t multi_batch_id = 0;

  std::lock_guard<std::mutex> lock(req_spec_mutex_);

  if (round_idx_ >= stat_start_round_ && stat_start_ms_ == 0) {
    stat_start_ms_ = ProfileTimer::GetCurrentTimeInMs();
  }

  for (size_t dp_idx = 0; dp_idx < planning_batch.dp_batches.size(); ++dp_idx) {
    const DpScheduleState& dp_state = context.GetDpScheduleState(multi_batch_id, dp_idx);
    auto& cache_manager = context.GetDpCacheManager(dp_idx);

    for (const auto& req : dp_state.prefilling_reqs) {
      if (req == nullptr || req->IsStopped()) continue;
      if (planning_batch.dp_batches[dp_idx].IsInBatch(req->req_id)) continue;
      if (req->HasPlanningTask()) continue;

      auto it = perf_req_states_.find(req->req_id);
      if (it == perf_req_states_.end()) continue;
      PerfReqState& state = it->second;

      const size_t prefix_len = state.input_token_len;
      const size_t total_token_len = state.input_token_len + state.query_token_num;

      if (!state.kv_allocated) {
        // ── 第 0 轮：设置 workload，分配 KV blocks，保存快照 ─────────────
        ScheduleTaskWorkload wl;
        wl.prefill_token_num = state.query_token_num;
        wl.prefill_start_offset = prefix_len;
        wl.sampling_token_num = 1;
        req->SetPlanningWorkload(wl);
        req->SetRemainingWorkload(wl);
        req->forwarding_tokens.clear();
        req->kv_cached_token_num = 0;

        if (!AllocateKvBlocksForReq(req, prefix_len, total_token_len, *cache_manager)) {
          KLLM_LOG_ERROR << "ModelPerfBackend: OOM for req_id=" << req->req_id << ", skipping";
          continue;
        }
        req->kv_cached_token_num = prefix_len;

        // 保存快照供后续轮恢复
        state.kv_allocated = true;
        state.snap_kv_cached_token_num = prefix_len;
        state.snap_planning_workload = wl;
        state.snap_remaining_workload = wl;

        KLLM_LOG_DEBUG << "ModelPerfBackend: first round, req_id=" << req->req_id << " (dp=" << dp_idx
                       << ", input=" << state.input_token_len << ", query=" << state.query_token_num << ")";
      } else if (needs_reset_) {
        // ── 后续轮：恢复快照（KV blocks 已分配，无需重复分配）────────────
        req->infer_stage = InferStage::kContext;
        req->kv_cached_token_num = state.snap_kv_cached_token_num;
        req->SetPlanningWorkload(state.snap_planning_workload);
        req->SetRemainingWorkload(state.snap_remaining_workload);
        req->forwarding_tokens.clear();

        KLLM_LOG_DEBUG << "ModelPerfBackend: reset round=" << round_idx_ << ", req_id=" << req->req_id;
      } else {
        // needs_reset_=false 但 kv_allocated=true 说明请求刚被加入、尚未进入第 0 轮后的 Update
        // 不再重复加入
        continue;
      }

      planning_batch.AddRunningReq(dp_idx, req);
    }
  }

  // 所有 dp 处理完后，若 needs_reset_ 由 UpdateWithGenerationResult 设置，则在此清除
  if (needs_reset_ && planning_batch.running_req_num > 0) {
    needs_reset_ = false;
  }

  // 有 running reqs 时启用 profile_metrics
  if (planning_batch.running_req_num > 0) {
    planning_batch.enable_profile_metrics = true;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// UpdateWithGenerationResult：每轮结束后处理
//   - 清理 inflight 状态 + 更新 KV cache token 追踪
//   - 若 stat 轮：收集 profile_metrics（来自 generation_output.profile_metrics）
//   - 若未完成所有轮：设置 needs_reset_=true，等待下次 Schedule 恢复
//   - 若完成所有轮：发出 kStopRequest，触发 req->Notify() → batch_waiter 解除阻塞
// ─────────────────────────────────────────────────────────────────────────────
void ModelPerfBackend::UpdateWithGenerationResult(size_t /*multi_batch_id*/,
                                                  const GenerationOutputGroup& generation_output,
                                                  const ScheduleContext& context,
                                                  std::vector<ScheduleBackendEvent>& events) {
  events.clear();
  if (stopped_.load()) return;

  // 本轮是否有请求完成（generation_output 可能为空）
  bool any_req = false;
  for (size_t dp_idx = 0; dp_idx < generation_output.reqs.size(); ++dp_idx) {
    for (const auto& req : generation_output.reqs[dp_idx]) {
      if (req == nullptr || req->IsStopped()) continue;
      any_req = true;
    }
  }
  if (!any_req) return;

  std::lock_guard<std::mutex> lock(req_spec_mutex_);

  // ── 清理每个 req 的 inflight 状态 ───────────────────────────────────────
  for (size_t dp_idx = 0; dp_idx < generation_output.reqs.size(); ++dp_idx) {
    auto& cache_manager = context.GetDpCacheManager(dp_idx);

    for (const auto& req : generation_output.reqs[dp_idx]) {
      if (req == nullptr || req->IsStopped()) continue;
      if (!perf_req_states_.count(req->req_id)) continue;

      KLLM_CHECK(req->HasInflightTask());
      req->UpdateAfterInflightTaskFinished();
      req->ResetInflightTask();
      req->flexible_cached_copy_tasks.clear();

      bool block_merged = false;
      cache_manager->UpdateRequestTokens(req->req_id, req->forwarding_tokens, req->kv_cached_token_num,
                                         req->kv_cache_blocks, block_merged, req->swa_kv_cache_blocks,
                                         req->swa_kv_cache_block_idx_offsets);
    }
  }

  // ── 判断当前轮是 warmup 还是 stat ───────────────────────────────────────
  const bool is_stat_round = (round_idx_ >= stat_start_round_);
  if (is_stat_round) {
    if (generation_output.has_profile_metrics) {
      current_stat_metrics_.push_back(generation_output.profile_metrics);
      const ForwardPerfMetrics* main_metrics = generation_output.profile_metrics.Get(ForwardMetricsKind::kMain);
      const LayerPerfMetrics* attn_metrics =
          main_metrics != nullptr ? main_metrics->GetOp(ForwardProfileOp::kAttn) : nullptr;
      KLLM_LOG_INFO << "ModelPerfBackend: stat round=" << round_idx_
                    << ", forward_time_ns=" << (main_metrics != nullptr ? main_metrics->forward_time_ns : 0)
                    << ", attn_count=" << (attn_metrics != nullptr ? attn_metrics->count : 0);
    } else {
      KLLM_LOG_WARNING << "ModelPerfBackend: stat round=" << round_idx_
                       << " but no profile_metrics in generation_output";
      current_stat_metrics_.push_back(ProfileMetrics{});
    }
  }

  ++round_idx_;

  if (round_idx_ < total_rounds_) {
    // ── 还有更多轮：标记 needs_reset_，Schedule 下次调用时恢复快照 ─────────
    KLLM_LOG_DEBUG << "ModelPerfBackend: round=" << (round_idx_ - 1) << " done, next round=" << round_idx_
                   << ", setting needs_reset_";
    needs_reset_ = true;
  } else {
    // ── 所有轮完成：发出 kStopRequest ────────────────────────────────────
    stat_end_ms_ = ProfileTimer::GetCurrentTimeInMs();
    KLLM_LOG_INFO << "ModelPerfBackend: all " << total_rounds_ << " round(s) done, stopping perf reqs";
    for (size_t dp_idx = 0; dp_idx < generation_output.reqs.size(); ++dp_idx) {
      for (const auto& req : generation_output.reqs[dp_idx]) {
        if (req == nullptr || req->IsStopped()) continue;
        if (!perf_req_states_.count(req->req_id)) continue;

        req->finish_status = Status(RET_SUCCESS);
        req->finished = true;
        ScheduleBackendEvent ev;
        ev.type = ScheduleBackendEventType::kStopRequest;
        ev.dp_idx = dp_idx;
        ev.req_ptr = req;
        ev.status = Status(RET_SUCCESS);
        events.push_back(std::move(ev));
        KLLM_LOG_DEBUG << "ModelPerfBackend: emitting kStopRequest for req_id=" << req->req_id;
      }
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Stop
// ─────────────────────────────────────────────────────────────────────────────
void ModelPerfBackend::Stop() {
  stopped_.store(true);
  KLLM_LOG_INFO << "ModelPerfBackend: stopped.";
}

// ─────────────────────────────────────────────────────────────────────────────
// Finalize：写出 JSON 结果文件
// ─────────────────────────────────────────────────────────────────────────────
void ModelPerfBackend::Finalize() {
  nlohmann::json output_json;
  output_json["batches"] = nlohmann::json::array();

  for (const auto& br : results_) {
    nlohmann::json b_json;
    b_json["batch_id"] = br.batch_id;
    b_json["stat_summary"] = ProfileMetricsSummaryToJson(ComputeProfileMetricsSummary(br.stat_metrics));

    if (config_.output_level == ModelPerfOutputLevel::kDetail) {
      b_json["stat_rounds"] = nlohmann::json::array();
      for (const auto& m : br.stat_metrics) {
        b_json["stat_rounds"].push_back(ProfileMetricsToRoundJson(m));
      }
    }

    output_json["batches"].push_back(b_json);
  }

  std::ofstream out_file(config_.output_file);
  if (!out_file.is_open()) {
    KLLM_LOG_ERROR << "ModelPerfBackend: cannot write output file '" << config_.output_file << "'";
    return;
  }
  out_file << output_json.dump(2);
  KLLM_LOG_INFO << "ModelPerfBackend: results written to '" << config_.output_file << "'";
}

}  // namespace ksana_llm
