/* Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * ModelPerfBackend 单元测试
 * ==============================================================================*/

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "batch_scheduler/event_driven_scheduler/model_perf/model_perf_backend.h"
#include "batch_scheduler/event_driven_scheduler/model_perf/model_perf_token_utils.h"
#include "batch_scheduler/event_driven_scheduler/schedule_context.h"
#include "cache_manager/cache_manager_interface.h"
#include "loguru.hpp"
#include "nlohmann/json.hpp"
#include "runtime/infer_request.h"
#include "tests/test.h"
#include "utils/request.h"

namespace ksana_llm {
namespace {

class MaxLogVerbosityEnv : public ::testing::Environment {
 public:
  void SetUp() override { loguru::g_stderr_verbosity = loguru::Verbosity_MAX; }
};
static auto* unused_log_env = ::testing::AddGlobalTestEnvironment(new MaxLogVerbosityEnv());

class MockCacheManager : public CacheManagerInterface {
 public:
  MOCK_METHOD(void, InitializeCachedBlocks, (), (override));
  MOCK_METHOD(std::shared_ptr<BlockAllocatorGroupInterface>, GetBlockAllocatorGroup, (), (const, override));
  MOCK_METHOD(size_t, GetUsableBlockNumber, (), (override));
  MOCK_METHOD(size_t, GetHostFreeBlockNumber, (), (override));
  MOCK_METHOD(size_t, GetUsedBlockNumber, (), (override));
  MOCK_METHOD(size_t, GetRequestStepBlockNumber, (int64_t, size_t, size_t), (override));
  MOCK_METHOD(size_t, GetRequestStepBlockNumberForOneNextToken, (int64_t), (override));
  MOCK_METHOD(size_t, GetRequestUsableBlockNumber, (int64_t), (override));
  MOCK_METHOD(Status, GetRequestPrefixBlockNumber,
              (int64_t, const std::vector<int>&, size_t, size_t&, size_t&, size_t&), (override));
  MOCK_METHOD(Status, GetRequestPrefixBlockNumberBeforeRemoteCacheLoading,
              (int64_t, const std::vector<int>&, size_t, size_t&, size_t&, size_t&, size_t&, size_t&, size_t&),
              (override));
  MOCK_METHOD(Status, AllocateBlocksForRemoteCache,
              (int64_t, std::vector<int>&, size_t, size_t, std::vector<int>&, std::vector<int>&), (override));
  MOCK_METHOD(Status, ConfirmRemoteCacheLoadingResult,
              (int64_t, const std::vector<int>&, size_t, const std::vector<size_t>&), (override));
  MOCK_METHOD(Status, AllocateRequestBlocks,
              (int64_t, size_t, std::vector<int>&, std::vector<int>&, std::vector<size_t>&, int&), (override));
  MOCK_METHOD(void, UpdateFlexibleCache,
              (int64_t, const std::vector<int>&, int, std::vector<FlexibleCachedCopyTask>&, size_t&), (override));
  MOCK_METHOD(Status, UpdateRequestTokens,
              (int64_t, const std::vector<int>&, size_t, std::vector<int>&, bool&, std::vector<int>&,
               std::vector<size_t>&),
              (override));
  MOCK_METHOD(void, DestroyFinishedRequest, (int64_t), (override));
  MOCK_METHOD(bool, IsPrefixCachingEnabled, (), (override));
};

struct TempConfigDir {
  std::filesystem::path dir;

  explicit TempConfigDir(const std::string& name) {
    dir = std::filesystem::temp_directory_path() / name;
    std::filesystem::create_directories(dir);
  }

  ~TempConfigDir() { std::filesystem::remove_all(dir); }

  std::string WriteJson(const std::string& filename, const nlohmann::json& content) const {
    const std::filesystem::path path = dir / filename;
    std::ofstream out(path);
    out << content.dump(2);
    return path.string();
  }
};

BatchSchedulerConfig MakeBatchSchedulerConfig(size_t max_step_token_num = 4096) {
  BatchSchedulerConfig config;
  config.max_step_token_num = max_step_token_num;
  config.max_pp_batch_num = 1;
  return config;
}

nlohmann::json MakeBatchJson(const std::string& batch_id, size_t input_token_len, size_t query_token_num, int req_num,
                             int warmup_rounds = 0, int stat_rounds = 1, size_t dp_id = 0) {
  return nlohmann::json{
      {"batch_id", batch_id},
      {"warmup_rounds", warmup_rounds},
      {"stat_rounds", stat_rounds},
      {"dp_batches",
       {{{"dp_id", dp_id},
         {"requests",
          {{{"input_token_len", input_token_len}, {"query_token_num", query_token_num}, {"req_num", req_num}}}}}}}};
}

std::string WriteModelPerfConfig(const TempConfigDir& temp_dir, const nlohmann::json& batch_config,
                                 const std::string& output_file = "model_perf_result.json",
                                 const std::string& output_level = "detail") {
  const std::string batch_config_path = temp_dir.WriteJson("batch_config.json", batch_config);
  nlohmann::json model_perf_config = {{"output_file", output_file},
                                      {"metrics_granularity", "batch"},
                                      {"output_level", output_level},
                                      {"token_seed", 42},
                                      {"batch_config", batch_config_path}};
  return temp_dir.WriteJson("model_perf_config.json", model_perf_config);
}

std::vector<std::shared_ptr<Request>> g_alive_requests;

std::shared_ptr<InferRequest> MakePerfInferRequest(int64_t req_id, size_t input_token_len, size_t query_token_len,
                                                   uint32_t dp_id = 0, uint32_t vocab_size = 128) {
  auto python_input = std::make_shared<KsanaPythonInput>();
  GenerateTokens(input_token_len + query_token_len, 42u, vocab_size, python_input->input_tokens);
  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  auto request = std::make_shared<Request>(python_input, req_ctx);
  g_alive_requests.push_back(request);
  auto infer_req = std::make_shared<InferRequest>(request, 0);
  infer_req->req_id = req_id;
  infer_req->block_token_num = 16;
  infer_req->attn_dp_group_id = dp_id;
  infer_req->infer_stage = InferStage::kContext;
  infer_req->ResetPrefillingTokens();
  return infer_req;
}

// 模拟 Schedule + AddRunningReq + Executor forward 完成后的 inflight 状态。
void ArmPrefillInflightTask(const std::shared_ptr<InferRequest>& req, size_t prefix_len, size_t query_token_num) {
  ScheduleTaskWorkload wl;
  wl.prefill_token_num = query_token_num;
  wl.prefill_start_offset = prefix_len;
  wl.sampling_token_num = 1;
  req->infer_stage = InferStage::kContext;
  req->forwarding_tokens.clear();
  req->SetPlanningWorkload(wl);
  req->SetRemainingWorkload(wl);
  req->SetPlanningTask();
  req->LaunchPlanningTask();
  ASSERT_TRUE(req->HasInflightTask());
}

void SetupDefaultCacheManagerMock(MockCacheManager& mock_cm) {
  ON_CALL(mock_cm, GetRequestPrefixBlockNumber(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
      .WillByDefault(testing::Return(Status()));
  ON_CALL(mock_cm, GetRequestStepBlockNumber(testing::_, testing::_, testing::_)).WillByDefault(testing::Return(1));
  ON_CALL(mock_cm, AllocateRequestBlocks(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
      .WillByDefault(testing::Return(Status()));
  ON_CALL(mock_cm,
          UpdateRequestTokens(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
      .WillByDefault(testing::Return(Status()));
}

class ModelPerfBackendTest : public ::testing::Test {
 protected:
  void SetUp() override {
    InitLoguru();
    g_alive_requests.clear();
    mock_cm_ = std::make_shared<testing::NiceMock<MockCacheManager>>();
    SetupDefaultCacheManagerMock(*mock_cm_);
    dp_cache_managers_ = {mock_cm_};
    schedule_state_.schedule_states.assign(1, std::vector<DpScheduleState>(1));
    schedule_context_.SetScheduleState(&schedule_state_);
    schedule_context_.SetDpCacheManagers(&dp_cache_managers_);
  }

  std::shared_ptr<testing::NiceMock<MockCacheManager>> mock_cm_;
  std::vector<std::shared_ptr<CacheManagerInterface>> dp_cache_managers_;
  ScheduleState schedule_state_;
  ScheduleContext schedule_context_;
};

TEST_F(ModelPerfBackendTest, ConstructorRejectsZeroBlockTokenNum) {
  TempConfigDir temp_dir("model_perf_zero_block");
  const std::string config_path =
      WriteModelPerfConfig(temp_dir, nlohmann::json{{"batches", {MakeBatchJson("b0", 8, 8, 1)}}});
  BatchSchedulerConfig config = MakeBatchSchedulerConfig();
  EXPECT_THROW(ModelPerfBackend(config, config_path, /*vocab_size=*/128, /*block_token_num=*/0), std::runtime_error);
}

TEST_F(ModelPerfBackendTest, ConstructorRejectsZeroVocabSize) {
  TempConfigDir temp_dir("model_perf_zero_vocab");
  const std::string config_path =
      WriteModelPerfConfig(temp_dir, nlohmann::json{{"batches", {MakeBatchJson("b0", 8, 8, 1)}}});
  BatchSchedulerConfig config = MakeBatchSchedulerConfig();
  EXPECT_THROW(ModelPerfBackend(config, config_path, /*vocab_size=*/0, /*block_token_num=*/16), std::runtime_error);
}

TEST_F(ModelPerfBackendTest, ConstructorRejectsMissingConfigFile) {
  BatchSchedulerConfig config = MakeBatchSchedulerConfig();
  EXPECT_THROW(ModelPerfBackend(config, "/tmp/model_perf_missing_config.json", 128, 16), std::runtime_error);
}

TEST_F(ModelPerfBackendTest, ConstructorRejectsMissingBatchConfigField) {
  TempConfigDir temp_dir("model_perf_missing_batch_config");
  const std::string config_path = temp_dir.WriteJson("model_perf_config.json", {{"output_file", "out.json"}});
  BatchSchedulerConfig config = MakeBatchSchedulerConfig();
  EXPECT_THROW(ModelPerfBackend(config, config_path, 128, 16), std::runtime_error);
}

TEST_F(ModelPerfBackendTest, ConstructorRejectsEmptyBatches) {
  TempConfigDir temp_dir("model_perf_empty_batches");
  const std::string config_path = WriteModelPerfConfig(temp_dir, nlohmann::json{{"batches", nlohmann::json::array()}});
  BatchSchedulerConfig config = MakeBatchSchedulerConfig();
  EXPECT_THROW(ModelPerfBackend(config, config_path, 128, 16), std::runtime_error);
}

TEST_F(ModelPerfBackendTest, ConstructorSkipsBatchExceedingMaxStepTokenNum) {
  TempConfigDir temp_dir("model_perf_skip_batch");
  nlohmann::json batch_config = {{"batches", {MakeBatchJson("too_large", 64, 64, 2), MakeBatchJson("valid", 8, 8, 1)}}};
  const std::string config_path = WriteModelPerfConfig(temp_dir, batch_config);
  BatchSchedulerConfig config = MakeBatchSchedulerConfig(/*max_step_token_num=*/100);
  ModelPerfBackend backend(config, config_path, 128, 16);
  EXPECT_EQ(backend.batches_.size(), 1u);
  EXPECT_EQ(backend.batches_[0].batch_id, "valid");
}

TEST_F(ModelPerfBackendTest, ConstructorThrowsWhenAllBatchesSkipped) {
  TempConfigDir temp_dir("model_perf_all_skipped");
  const std::string config_path =
      WriteModelPerfConfig(temp_dir, nlohmann::json{{"batches", {MakeBatchJson("too_large", 64, 64, 2)}}});
  BatchSchedulerConfig config = MakeBatchSchedulerConfig(/*max_step_token_num=*/100);
  EXPECT_THROW(ModelPerfBackend(config, config_path, 128, 16), std::runtime_error);
}

TEST(ModelPerfBackendComputeProfileMetricsSummaryTest, ComputesMeanAndStddev) {
  std::vector<ProfileMetrics> metrics(2);
  metrics[0].schedule_forward_interval_ns = 50;
  metrics[0].GetOrCreate(ForwardMetricsKind::kMain).forward_time_ns = 100;
  metrics[0].GetOrCreate(ForwardMetricsKind::kMain).GetOrCreateOp(ForwardProfileOp::kAttn).cumulative_time_ns = 10;
  metrics[0].GetOrCreate(ForwardMetricsKind::kMain).GetOrCreateOp(ForwardProfileOp::kAttn).count = 2;
  metrics[1].schedule_forward_interval_ns = 150;
  metrics[1].GetOrCreate(ForwardMetricsKind::kMain).forward_time_ns = 200;
  metrics[1].GetOrCreate(ForwardMetricsKind::kMain).GetOrCreateOp(ForwardProfileOp::kAttn).cumulative_time_ns = 30;
  metrics[1].GetOrCreate(ForwardMetricsKind::kMain).GetOrCreateOp(ForwardProfileOp::kAttn).count = 4;

  const ProfileMetricsSummary summary = ModelPerfBackend::ComputeProfileMetricsSummary(metrics);
  EXPECT_EQ(summary.round_count, 2u);
  EXPECT_DOUBLE_EQ(summary.schedule_forward_interval_ns.mean, 100.0);
  EXPECT_DOUBLE_EQ(summary.schedule_forward_interval_ns.stddev, 50.0);
  ASSERT_EQ(summary.forward_metrics.count(ForwardMetricsKind::kMain), 1u);
  const ForwardPerfMetricsSummary& main_summary = summary.forward_metrics.at(ForwardMetricsKind::kMain);
  EXPECT_DOUBLE_EQ(main_summary.forward_time_ns.mean, 150.0);
  EXPECT_DOUBLE_EQ(main_summary.forward_time_ns.stddev, 50.0);
  ASSERT_EQ(main_summary.ops.count("attn"), 1u);
  const OpPerfMetricsSummary& attn_summary = main_summary.ops.at("attn");
  EXPECT_DOUBLE_EQ(attn_summary.cumulative_time_ns.mean, 20.0);
  EXPECT_DOUBLE_EQ(attn_summary.cumulative_time_ns.stddev, 10.0);
  EXPECT_DOUBLE_EQ(attn_summary.count.mean, 3.0);
  EXPECT_DOUBLE_EQ(attn_summary.count.stddev, 1.0);
}

TEST(ModelPerfBackendComputeBatchQueryTokenSumTest, SumsQueryTokensWithReqNum) {
  PerfBatchSpec spec;
  spec.batch_id = "sum_test";
  PerfDpBatchSpec dp0;
  dp0.dp_id = 0;
  dp0.requests.push_back(PerfRequestSpec{/*input_token_len=*/10, /*query_token_num=*/32, /*req_num=*/2});
  dp0.requests.push_back(PerfRequestSpec{/*input_token_len=*/10, /*query_token_num=*/8, /*req_num=*/1});
  spec.dp_batches.push_back(dp0);

  PerfDpBatchSpec dp1;
  dp1.dp_id = 1;
  dp1.requests.push_back(PerfRequestSpec{/*input_token_len=*/10, /*query_token_num=*/4, /*req_num=*/3});
  spec.dp_batches.push_back(dp1);

  EXPECT_EQ(ModelPerfBackend::ComputeBatchQueryTokenSum(spec), 32u * 2u + 8u + 4u * 3u);
}

TEST(ModelPerfGenerateTokensTest, DeterministicForSameSeed) {
  std::vector<int> tokens_a;
  std::vector<int> tokens_b;
  GenerateTokens(16, 42u, 128, tokens_a);
  GenerateTokens(16, 42u, 128, tokens_b);
  EXPECT_EQ(tokens_a, tokens_b);
}

TEST(ModelPerfGenerateTokensTest, DifferentSeedsProduceDifferentTokens) {
  std::vector<int> tokens_a;
  std::vector<int> tokens_b;
  GenerateTokens(16, 42u, 128, tokens_a);
  GenerateTokens(16, 43u, 128, tokens_b);
  EXPECT_NE(tokens_a, tokens_b);
}

TEST_F(ModelPerfBackendTest, ScheduleFirstRoundAllocatesKvAndAddsRunningReq) {
  TempConfigDir temp_dir("model_perf_schedule_first_round");
  const std::string config_path =
      WriteModelPerfConfig(temp_dir, nlohmann::json{{"batches", {MakeBatchJson("b0", 16, 32, 1)}}});
  BatchSchedulerConfig config = MakeBatchSchedulerConfig();
  ModelPerfBackend backend(config, config_path, 128, 16);

  auto req = MakePerfInferRequest(/*req_id=*/1001, /*input_token_len=*/16, /*query_token_num=*/32);
  schedule_state_.schedule_states[0][0].prefilling_reqs.push_back(req);
  schedule_state_.req_id_to_req_ptr[req->req_id] = req;

  backend.total_rounds_ = 2;
  backend.round_idx_ = 0;
  backend.stat_start_round_ = 1;
  backend.needs_reset_ = false;
  backend.perf_req_states_[req->req_id] = PerfReqState{16, 32, 0, false, 0, {}, {}};

  auto planning_batch = std::make_shared<PlanningBatch>();
  planning_batch->dp_batches.resize(1);
  std::vector<std::shared_ptr<PlanningBatch>> planning_batches = {planning_batch};
  std::vector<ScheduleBackendEvent> events;

  backend.Schedule(schedule_context_, planning_batches, events);

  EXPECT_EQ(planning_batch->running_req_num, 1u);
  EXPECT_TRUE(planning_batch->enable_profile_metrics);
  ASSERT_TRUE(planning_batch->dp_batches[0].running_reqs.count(req->req_id) > 0);
  EXPECT_TRUE(backend.perf_req_states_[req->req_id].kv_allocated);
  EXPECT_EQ(req->kv_cached_token_num, 16u);
  // AddRunningReq 会调用 SetPlanningTask()，workload 已转入 planning_task_，planning_workload_ 被清空。
  EXPECT_TRUE(req->HasPlanningTask());
  // Prefill counts the full sequence (prefill_start_offset + prefill_token_num = 16 + 32)
  // against running_step_token_num so prefix-cache hits are charged to max_step_token.
  EXPECT_EQ(planning_batch->running_step_token_num, 48u);
  EXPECT_EQ(backend.perf_req_states_[req->req_id].snap_planning_workload.prefill_token_num, 32u);
  EXPECT_EQ(backend.perf_req_states_[req->req_id].snap_planning_workload.prefill_start_offset, 16u);
  EXPECT_EQ(req->GetRemainingWorkload().prefill_token_num, 0u);
  EXPECT_EQ(req->GetRemainingWorkload().prefill_start_offset, 48u);
}

TEST_F(ModelPerfBackendTest, ScheduleResetRoundRestoresSnapshot) {
  TempConfigDir temp_dir("model_perf_schedule_reset");
  const std::string config_path =
      WriteModelPerfConfig(temp_dir, nlohmann::json{{"batches", {MakeBatchJson("b0", 8, 16, 1)}}});
  BatchSchedulerConfig config = MakeBatchSchedulerConfig();
  ModelPerfBackend backend(config, config_path, 128, 16);

  auto req = MakePerfInferRequest(/*req_id=*/1002, /*input_token_len=*/8, /*query_token_num=*/16);
  schedule_state_.schedule_states[0][0].prefilling_reqs.push_back(req);

  ScheduleTaskWorkload snap_wl;
  snap_wl.prefill_token_num = 16;
  snap_wl.prefill_start_offset = 8;
  snap_wl.sampling_token_num = 1;

  PerfReqState state;
  state.input_token_len = 8;
  state.query_token_num = 16;
  state.kv_allocated = true;
  state.snap_kv_cached_token_num = 8;
  state.snap_planning_workload = snap_wl;
  state.snap_remaining_workload = snap_wl;
  backend.perf_req_states_[req->req_id] = state;
  backend.needs_reset_ = true;
  backend.round_idx_ = 1;

  auto planning_batch = std::make_shared<PlanningBatch>();
  planning_batch->dp_batches.resize(1);
  std::vector<std::shared_ptr<PlanningBatch>> planning_batches = {planning_batch};
  std::vector<ScheduleBackendEvent> events;

  backend.Schedule(schedule_context_, planning_batches, events);

  EXPECT_EQ(planning_batch->running_req_num, 1u);
  EXPECT_EQ(req->kv_cached_token_num, 8u);
  EXPECT_TRUE(req->HasPlanningTask());
  // Full sequence: prefill_start_offset(8) + prefill_token_num(16) = 24.
  EXPECT_EQ(planning_batch->running_step_token_num, 24u);
  EXPECT_EQ(req->GetRemainingWorkload().prefill_token_num, 0u);
  EXPECT_EQ(req->GetRemainingWorkload().prefill_start_offset, 24u);
  EXPECT_FALSE(backend.needs_reset_);
}

TEST_F(ModelPerfBackendTest, UpdateWithGenerationResultCollectsStatMetricsAndStopEvents) {
  TempConfigDir temp_dir("model_perf_update_rounds");
  const std::string output_file = (temp_dir.dir / "perf_out.json").string();
  const std::string config_path =
      WriteModelPerfConfig(temp_dir, nlohmann::json{{"batches", {MakeBatchJson("b0", 4, 8, 1, 1, 2)}}}, output_file);
  BatchSchedulerConfig config = MakeBatchSchedulerConfig();
  ModelPerfBackend backend(config, config_path, 128, 16);

  auto req = MakePerfInferRequest(/*req_id=*/1003, /*input_token_len=*/4, /*query_token_num=*/8);
  backend.total_rounds_ = 3;
  backend.round_idx_ = 0;
  backend.stat_start_round_ = 1;

  ScheduleTaskWorkload snap_wl;
  snap_wl.prefill_token_num = 8;
  snap_wl.prefill_start_offset = 4;
  snap_wl.sampling_token_num = 1;
  backend.perf_req_states_[req->req_id] = PerfReqState{4, 8, 0, true, 4, snap_wl, snap_wl};

  ArmPrefillInflightTask(req, /*prefix_len=*/4, /*query_token_num=*/8);

  GenerationOutputGroup generation_output;
  generation_output.reqs = {{req}};
  generation_output.has_profile_metrics = true;
  generation_output.profile_metrics.GetOrCreate(ForwardMetricsKind::kMain).forward_time_ns = 111;

  std::vector<ScheduleBackendEvent> events;

  backend.UpdateWithGenerationResult(/*multi_batch_id=*/0, generation_output, schedule_context_, events);
  EXPECT_EQ(backend.round_idx_, 1);
  EXPECT_TRUE(backend.current_stat_metrics_.empty());
  EXPECT_TRUE(events.empty());
  EXPECT_TRUE(backend.needs_reset_);

  ArmPrefillInflightTask(req, /*prefix_len=*/4, /*query_token_num=*/8);
  generation_output.profile_metrics.GetOrCreate(ForwardMetricsKind::kMain).forward_time_ns = 222;
  backend.UpdateWithGenerationResult(/*multi_batch_id=*/0, generation_output, schedule_context_, events);
  ASSERT_EQ(backend.current_stat_metrics_.size(), 1u);
  const ForwardPerfMetrics* main_metrics_0 = backend.current_stat_metrics_[0].Get(ForwardMetricsKind::kMain);
  ASSERT_NE(main_metrics_0, nullptr);
  EXPECT_EQ(main_metrics_0->forward_time_ns, 222u);
  EXPECT_EQ(backend.round_idx_, 2);
  EXPECT_TRUE(events.empty());

  ArmPrefillInflightTask(req, /*prefix_len=*/4, /*query_token_num=*/8);
  generation_output.profile_metrics.GetOrCreate(ForwardMetricsKind::kMain).forward_time_ns = 333;
  backend.UpdateWithGenerationResult(/*multi_batch_id=*/0, generation_output, schedule_context_, events);
  ASSERT_EQ(backend.current_stat_metrics_.size(), 2u);
  const ForwardPerfMetrics* main_metrics_1 = backend.current_stat_metrics_[1].Get(ForwardMetricsKind::kMain);
  ASSERT_NE(main_metrics_1, nullptr);
  EXPECT_EQ(main_metrics_1->forward_time_ns, 333u);
  ASSERT_EQ(events.size(), 1u);
  EXPECT_EQ(events[0].type, ScheduleBackendEventType::kStopRequest);
  EXPECT_EQ(events[0].req_ptr, req);
}

TEST_F(ModelPerfBackendTest, FinalizeWritesStatRoundsToJson) {
  TempConfigDir temp_dir("model_perf_finalize");
  const std::string output_file = (temp_dir.dir / "result.json").string();
  const std::string config_path =
      WriteModelPerfConfig(temp_dir, nlohmann::json{{"batches", {MakeBatchJson("batch_a", 4, 8, 1)}}}, output_file);
  BatchSchedulerConfig config = MakeBatchSchedulerConfig();
  ModelPerfBackend backend(config, config_path, 128, 16);

  ProfileMetrics metric;
  metric.schedule_forward_interval_ns = 500;
  ForwardPerfMetrics& main_metric = metric.GetOrCreate(ForwardMetricsKind::kMain);
  main_metric.forward_time_ns = 12345;
  main_metric.GetOrCreateOp(ForwardProfileOp::kAttn).cumulative_time_ns = 100;
  main_metric.GetOrCreateOp(ForwardProfileOp::kAttn).count = 2;
  main_metric.GetOrCreateOp(ForwardProfileOp::kFfn).cumulative_time_ns = 200;
  main_metric.GetOrCreateOp(ForwardProfileOp::kFfn).count = 3;

  ModelPerfBackend::BatchResult batch_result;
  batch_result.batch_id = "batch_a";
  batch_result.stat_metrics.push_back(metric);
  backend.results_.push_back(std::move(batch_result));

  backend.Finalize();

  std::ifstream in(output_file);
  ASSERT_TRUE(in.is_open());
  nlohmann::json output_json;
  in >> output_json;
  ASSERT_TRUE(output_json.contains("batches"));
  ASSERT_EQ(output_json["batches"].size(), 1u);
  EXPECT_EQ(output_json["batches"][0]["batch_id"], "batch_a");
  ASSERT_TRUE(output_json["batches"][0].contains("stat_summary"));
  EXPECT_EQ(output_json["batches"][0]["stat_summary"]["round_count"], 1);
  EXPECT_EQ(output_json["batches"][0]["stat_summary"]["schedule_forward_interval_ns"]["mean"], 500);
  EXPECT_EQ(output_json["batches"][0]["stat_summary"]["schedule_forward_interval_ns"]["stddev"], 0);
  EXPECT_EQ(output_json["batches"][0]["stat_summary"]["forward_metrics"]["main"]["forward_time_ns"]["mean"], 12345);
  EXPECT_EQ(output_json["batches"][0]["stat_summary"]["forward_metrics"]["main"]["forward_time_ns"]["stddev"], 0);
  ASSERT_EQ(output_json["batches"][0]["stat_rounds"].size(), 1u);
  EXPECT_EQ(output_json["batches"][0]["stat_rounds"][0]["schedule_forward_interval_ns"], 500);
  EXPECT_EQ(output_json["batches"][0]["stat_rounds"][0]["forward_metrics"]["main"]["forward_time_ns"], 12345);
  EXPECT_EQ(output_json["batches"][0]["stat_rounds"][0]["forward_metrics"]["main"]["ops"]["attn"]["count"], 2);
  EXPECT_EQ(output_json["batches"][0]["stat_rounds"][0]["forward_metrics"]["main"]["ops"]["ffn"]["count"], 3);
  EXPECT_FALSE(output_json["batches"][0]["stat_rounds"][0]["forward_metrics"].contains("speculative"));
}

TEST_F(ModelPerfBackendTest, FinalizeWritesSummaryOnlyWhenConfigured) {
  TempConfigDir temp_dir("model_perf_finalize_summary");
  const std::string output_file = (temp_dir.dir / "result_summary.json").string();
  const std::string config_path = WriteModelPerfConfig(
      temp_dir, nlohmann::json{{"batches", {MakeBatchJson("batch_a", 4, 8, 1)}}}, output_file, "summary");
  BatchSchedulerConfig config = MakeBatchSchedulerConfig();
  ModelPerfBackend backend(config, config_path, 128, 16);

  ProfileMetrics metric_a;
  metric_a.schedule_forward_interval_ns = 100;
  metric_a.GetOrCreate(ForwardMetricsKind::kMain).forward_time_ns = 100;
  ProfileMetrics metric_b;
  metric_b.schedule_forward_interval_ns = 300;
  metric_b.GetOrCreate(ForwardMetricsKind::kMain).forward_time_ns = 200;

  ModelPerfBackend::BatchResult batch_result;
  batch_result.batch_id = "batch_a";
  batch_result.stat_metrics.push_back(metric_a);
  batch_result.stat_metrics.push_back(metric_b);
  backend.results_.push_back(std::move(batch_result));

  backend.Finalize();

  std::ifstream in(output_file);
  ASSERT_TRUE(in.is_open());
  nlohmann::json output_json;
  in >> output_json;
  ASSERT_EQ(output_json["batches"].size(), 1u);
  EXPECT_FALSE(output_json["batches"][0].contains("stat_rounds"));
  EXPECT_EQ(output_json["batches"][0]["stat_summary"]["round_count"], 2);
  EXPECT_EQ(output_json["batches"][0]["stat_summary"]["forward_metrics"]["main"]["forward_time_ns"]["mean"], 150);
  EXPECT_EQ(output_json["batches"][0]["stat_summary"]["forward_metrics"]["main"]["forward_time_ns"]["stddev"], 50);
}

TEST_F(ModelPerfBackendTest, FinalizeRoundsSummaryMeanStddevToInteger) {
  TempConfigDir temp_dir("model_perf_finalize_round");
  const std::string output_file = (temp_dir.dir / "result_round.json").string();
  const std::string config_path = WriteModelPerfConfig(
      temp_dir, nlohmann::json{{"batches", {MakeBatchJson("batch_a", 4, 8, 1)}}}, output_file, "summary");
  BatchSchedulerConfig config = MakeBatchSchedulerConfig();
  ModelPerfBackend backend(config, config_path, 128, 16);

  ProfileMetrics metric_a;
  metric_a.GetOrCreate(ForwardMetricsKind::kMain).forward_time_ns = 100;
  ProfileMetrics metric_b;
  metric_b.GetOrCreate(ForwardMetricsKind::kMain).forward_time_ns = 101;

  ModelPerfBackend::BatchResult batch_result;
  batch_result.batch_id = "batch_a";
  batch_result.stat_metrics.push_back(metric_a);
  batch_result.stat_metrics.push_back(metric_b);
  backend.results_.push_back(std::move(batch_result));

  backend.Finalize();

  std::ifstream in(output_file);
  ASSERT_TRUE(in.is_open());
  nlohmann::json output_json;
  in >> output_json;
  EXPECT_EQ(output_json["batches"][0]["stat_summary"]["forward_metrics"]["main"]["forward_time_ns"]["mean"], 101);
  EXPECT_EQ(output_json["batches"][0]["stat_summary"]["forward_metrics"]["main"]["forward_time_ns"]["stddev"], 1);
}

TEST_F(ModelPerfBackendTest, ConstructorRejectsInvalidOutputLevel) {
  TempConfigDir temp_dir("model_perf_invalid_output_level");
  const std::string batch_config_path =
      temp_dir.WriteJson("batch_config.json", nlohmann::json{{"batches", {MakeBatchJson("b0", 4, 8, 1)}}});
  const std::string config_path = temp_dir.WriteJson(
      "model_perf_config.json",
      {{"output_file", "out.json"}, {"output_level", "invalid"}, {"batch_config", batch_config_path}});
  BatchSchedulerConfig config = MakeBatchSchedulerConfig();
  EXPECT_THROW(ModelPerfBackend(config, config_path, 128, 16), std::runtime_error);
}

TEST_F(ModelPerfBackendTest, StopPreventsFurtherSchedule) {
  TempConfigDir temp_dir("model_perf_stop");
  const std::string config_path =
      WriteModelPerfConfig(temp_dir, nlohmann::json{{"batches", {MakeBatchJson("b0", 4, 8, 1)}}});
  BatchSchedulerConfig config = MakeBatchSchedulerConfig();
  ModelPerfBackend backend(config, config_path, 128, 16);
  backend.Stop();

  auto planning_batch = std::make_shared<PlanningBatch>();
  planning_batch->dp_batches.resize(1);
  std::vector<std::shared_ptr<PlanningBatch>> planning_batches = {planning_batch};
  std::vector<ScheduleBackendEvent> events;
  backend.Schedule(schedule_context_, planning_batches, events);
  EXPECT_TRUE(events.empty());
  EXPECT_EQ(planning_batch->running_req_num, 0u);
}

}  // namespace
}  // namespace ksana_llm
