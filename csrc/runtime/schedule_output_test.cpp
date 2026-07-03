/* Copyright 2024 Tencent Inc.  All rights reserved.
 *
 * ==============================================================================*/

#include <gtest/gtest.h>

#include <memory>
#include <unordered_map>
#include <vector>

#include "batch_scheduler/batch_scheduler_interface.h"
#include "batch_scheduler/schedule_processor_interface.h"
#include "runtime/infer_request.h"
#include "runtime/infer_stage.h"
#include "runtime/schedule_output.h"
#include "test.h"

namespace ksana_llm {
namespace {

class DummyBatchScheduler : public BatchSchedulerInterface {
 public:
  std::shared_ptr<ScheduleOutputGroup> Schedule(size_t) override { return nullptr; }
  bool TryToLaunchPlannedScheduleOutput(size_t, ScheduleOutput&, std::vector<std::shared_ptr<InferRequest>>&) override {
    return false;
  }
  void UpdateWithGenerationResult(size_t, const GenerationOutputGroup&) override {}
  Status AddInferRequest(std::vector<std::shared_ptr<InferRequest>>&) override { return Status(); }
  void SetCacheManager(std::shared_ptr<CacheManagerInterface>, int) override {}
  std::shared_ptr<CacheManagerInterface>& GetCacheManager(int) override { return cache_manager_; }
  bool IsIdle(size_t) override { return true; }
  void WaitUntilHaveReqs(size_t) override {}
  void Stop() override {}

 private:
  std::shared_ptr<CacheManagerInterface> cache_manager_ = nullptr;
};

class DummyScheduleProcessor : public ScheduleProcessorInterface {
 public:
  void Initialize(std::shared_ptr<BatchSchedulerInterface> batch_scheduler) override {
    batch_scheduler_ = batch_scheduler;
  }
  Status AddInferRequest(std::vector<std::shared_ptr<InferRequest>>&) override { return Status(); }
  std::shared_ptr<ScheduleResult> GetNextScheduleResult() override { return nullptr; }
  void UpdateWithGenerationResult(const GenerationOutputGroup&) override {}
  void Lock() override {}
  void Unlock() override {}
  void Start() override {}
  void Stop() override {}
};

template <typename T>
bool CheckVectorsVector(const std::vector<std::vector<T>>& src, const std::vector<std::vector<T>>& dst) {
  return src == dst;
}

TargetDescribe BuildTargetDescribe() {
  TargetDescribe target;
  target.cutoff_layer = {1, 3};
  target.token_id = {42};
  target.slice_pos = {{0, 1}};
  target.token_reduce_mode = TokenReduceMode::GATHER_ALL;
  return target;
}

std::shared_ptr<Request> BuildRequest(std::shared_ptr<KsanaPythonInput>& input) {
  input->model_name = "schedule_output_test";
  input->input_tokens = {101, 103, 105};
  input->sampling_config.topk = 3;
  input->sampling_config.temperature = 0.7f;
  input->sampling_config.stop_token_ids = {7, 9};
  input->request_target["logits"] = BuildTargetDescribe();

  auto ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  return std::make_shared<Request>(input, ctx);
}

std::shared_ptr<InferRequest> BuildInferRequest(std::shared_ptr<Request>& request, int64_t req_id,
                                                bool patch_transfer) {
  auto infer_req = std::make_shared<InferRequest>(request, 0);
  infer_req->req_id = req_id;
  infer_req->logits_custom_length = 12;
  infer_req->sampling_token_num = 2;
  infer_req->last_step_token_num = 1;
  infer_req->mrotary_embedding_pos_offset = 3;
  infer_req->xdrotary_embedding_pos_offset = 4;
  infer_req->output_tokens = {201, 203};
  infer_req->draft_tokens.mtp = {301};
  infer_req->accepted_tokens = {401};
  infer_req->forwarding_tokens_draft_num = 1;
  infer_req->generated_tokens = {501, 503};
  infer_req->attn_dp_group_id = 2;
  infer_req->kv_comm_group_key = "group_key";
  infer_req->forwarding_tokens = {601, 603, 605};
  infer_req->sampling_result_tokens = {701};
  infer_req->infer_stage = InferStage::kDecode;
  infer_req->step = 5;
  infer_req->kv_cached_token_num = 2;
  infer_req->kv_cache_blocks = {11, 13, 21};
  infer_req->block_token_num = 16;
  infer_req->logits_offset = 7;
  infer_req->prefix_cache_len = 9;
  infer_req->incremental_decoded_str = "partial";
  infer_req->need_recreate_generator = false;
  infer_req->patch_transfer = patch_transfer;
  return infer_req;
}

bool CheckRequestTarget(const std::unordered_map<std::string, TargetDescribe>& src,
                        const std::unordered_map<std::string, TargetDescribe>& dst) {
  if (src.size() != dst.size()) {
    return false;
  }
  for (const auto& [key, value] : src) {
    auto it = dst.find(key);
    if (it == dst.end()) {
      return false;
    }
    const auto& other = it->second;
    if (value.cutoff_layer != other.cutoff_layer || value.token_id != other.token_id ||
        value.slice_pos != other.slice_pos || value.token_reduce_mode != other.token_reduce_mode) {
      return false;
    }
  }
  return true;
}

ScheduleOutput BuildScheduleOutput() {
  ScheduleOutput output;
  output.schedule_id = 42;
  output.finish_req_ids = {{1, 3, 5}, {7}};

  return output;
}

TEST(ScheduleOutputTest, SerializeAndDeserializeScheduleOutput) {
  auto scheduler = std::make_shared<DummyBatchScheduler>();
  auto schedule_processor = std::make_shared<DummyScheduleProcessor>();
  schedule_processor->Initialize(scheduler);
  ScheduleOutputParser::SetScheduleProcessor(schedule_processor);

  ScheduleOutput src = BuildScheduleOutput();
  size_t bytes = ScheduleOutputParser::GetSerializedSize(&src);
  std::vector<uint8_t> buffer(bytes);

  size_t serialized_bytes = 0;
  Status status = ScheduleOutputParser::SerializeScheduleOutput(&src, buffer.data(), &serialized_bytes);
  EXPECT_TRUE(status.OK());
  EXPECT_GT(serialized_bytes, 0u);

  ScheduleOutput dst;
  status = ScheduleOutputParser::DeserializeScheduleOutput(buffer.data(), &dst);
  EXPECT_TRUE(status.OK());

  EXPECT_EQ(src.schedule_id, dst.schedule_id);
  EXPECT_TRUE(CheckVectorsVector(src.finish_req_ids, dst.finish_req_ids));
  EXPECT_TRUE(dst.executor_running_reqs.empty());
  EXPECT_TRUE(dst.running_req_patches.empty());
}

TEST(ScheduleOutputTest, SerializeInferRequestAndPatch) {
  auto scheduler = std::make_shared<DummyBatchScheduler>();
  auto schedule_processor = std::make_shared<DummyScheduleProcessor>();
  schedule_processor->Initialize(scheduler);
  ScheduleOutputParser::SetScheduleProcessor(schedule_processor);

  auto input = std::make_shared<KsanaPythonInput>();
  auto request = BuildRequest(input);
  auto full_req = BuildInferRequest(request, 1001, false);
  auto patch_req = BuildInferRequest(request, 1002, true);
  patch_req->forwarding_tokens = {901, 903};
  patch_req->kv_cache_blocks = {31, 33};
  patch_req->kv_cached_token_num = 3;
  patch_req->forwarding_tokens_draft_num = 2;
  patch_req->sampling_token_num = 4;
  patch_req->prefix_cache_len = 11;
  patch_req->infer_stage = InferStage::kContext;
  patch_req->step = 1;

  ScheduleOutput src = BuildScheduleOutput();
  src.running_reqs = {full_req, patch_req};

  size_t bytes = ScheduleOutputParser::GetSerializedSize(&src);
  std::vector<uint8_t> buffer(bytes);

  size_t serialized_bytes = 0;
  Status status = ScheduleOutputParser::SerializeScheduleOutput(&src, buffer.data(), &serialized_bytes);
  EXPECT_TRUE(status.OK());

  ScheduleOutput dst;
  status = ScheduleOutputParser::DeserializeScheduleOutput(buffer.data(), &dst);
  EXPECT_TRUE(status.OK());

  ASSERT_EQ(dst.executor_running_reqs.size(), 1u);
  ASSERT_EQ(dst.running_req_patches.size(), 1u);

  const auto& exec_req = dst.executor_running_reqs[0];
  EXPECT_EQ(exec_req->req_id, full_req->req_id);
  EXPECT_EQ(exec_req->logits_custom_length, full_req->logits_custom_length);
  EXPECT_EQ(exec_req->last_step_token_num, full_req->last_step_token_num);
  EXPECT_EQ(exec_req->input_tokens, full_req->input_tokens);
  EXPECT_EQ(exec_req->mrotary_embedding_pos_offset, full_req->mrotary_embedding_pos_offset);
  EXPECT_EQ(exec_req->xdrotary_embedding_pos_offset, full_req->xdrotary_embedding_pos_offset);
  EXPECT_EQ(exec_req->output_tokens, full_req->output_tokens);
  EXPECT_EQ(exec_req->draft_tokens.mtp, full_req->draft_tokens.mtp);
  EXPECT_EQ(exec_req->accepted_tokens, full_req->accepted_tokens);
  EXPECT_EQ(exec_req->forwarding_tokens_draft_num, full_req->forwarding_tokens_draft_num);
  EXPECT_EQ(exec_req->generated_tokens, full_req->generated_tokens);
  EXPECT_EQ(exec_req->forwarding_tokens, full_req->forwarding_tokens);
  EXPECT_EQ(exec_req->sampling_result_tokens, full_req->sampling_result_tokens);
  EXPECT_EQ(exec_req->kv_cache_blocks, full_req->kv_cache_blocks);
  EXPECT_EQ(exec_req->block_token_num, full_req->block_token_num);
  EXPECT_EQ(exec_req->logits_offset, full_req->logits_offset);
  EXPECT_EQ(exec_req->prefix_cache_len, full_req->prefix_cache_len);
  EXPECT_EQ(exec_req->infer_stage, full_req->infer_stage);
  EXPECT_EQ(exec_req->step, full_req->step);
  EXPECT_EQ(exec_req->kv_cached_token_num, full_req->kv_cached_token_num);
  EXPECT_EQ(exec_req->kv_comm_group_key, full_req->kv_comm_group_key);
  EXPECT_EQ(exec_req->attn_dp_group_id, full_req->attn_dp_group_id);
  EXPECT_EQ(exec_req->sampling_token_num, full_req->sampling_token_num);
  EXPECT_EQ(exec_req->incremental_decoded_str, full_req->incremental_decoded_str);
  EXPECT_EQ(exec_req->need_create_generator, full_req->need_recreate_generator);
  EXPECT_TRUE(CheckRequestTarget(full_req->request_target, exec_req->request_target));
  EXPECT_EQ(exec_req->sampling_config.topk, full_req->sampling_config.topk);
  EXPECT_EQ(exec_req->sampling_config.temperature, full_req->sampling_config.temperature);
  EXPECT_EQ(exec_req->sampling_config.stop_token_ids, full_req->sampling_config.stop_token_ids);

  const auto& patch = dst.running_req_patches[0];
  EXPECT_EQ(patch->req_id, patch_req->req_id);
  EXPECT_EQ(patch->delta_forwarding_tokens, patch_req->forwarding_tokens);
  EXPECT_EQ(patch->forwarding_tokens_offset, 0u);
  EXPECT_EQ(patch->forwarding_tokens_draft_num, patch_req->forwarding_tokens_draft_num);
  EXPECT_EQ(patch->kv_cached_token_num, patch_req->kv_cached_token_num);
  EXPECT_EQ(patch->prefix_cache_len, patch_req->prefix_cache_len);
  EXPECT_EQ(patch->delta_kv_cache_blocks, patch_req->kv_cache_blocks);
  EXPECT_EQ(patch->kv_cache_blocks_offset, 0u);
  EXPECT_EQ(patch->infer_stage, patch_req->infer_stage);
  EXPECT_EQ(patch->step, patch_req->step);
  EXPECT_EQ(patch->sampling_token_num, patch_req->sampling_token_num);
}

// =============================================================================
// build-ahead 相关：IsLaunchable(max_depth) / LaunchScheduleOutput placeholder / 序列化快照
// =============================================================================

// InferRequest 持有 Request 内部字段的引用；测试夹具须同时 pin 住 request 与 infer_req 生命周期。
struct DecodeReadyInferRequestFixture {
  std::shared_ptr<Request> request;
  std::shared_ptr<InferRequest> infer_req;
};

// 构造已进入 decode 稳态的 InferRequest（prefill 已 retire，inflight_count_=0）。
DecodeReadyInferRequestFixture BuildDecodeReadyInferRequest(int64_t req_id, int first_decoded_token = 10) {
  auto input = std::make_shared<KsanaPythonInput>();
  input->input_tokens = {1, 2, 3};
  auto ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  auto request = std::make_shared<Request>(input, ctx);
  auto req = std::make_shared<InferRequest>(request, 0);
  req->req_id = req_id;
  req->output_tokens = {1, 2, 3};
  req->ResetPrefillingTokens();

  ScheduleTaskWorkload prefill_wl;
  prefill_wl.prefill_token_num = 3;
  prefill_wl.prefill_start_offset = 0;
  prefill_wl.sampling_token_num = 1;
  req->SetPlanningWorkload(prefill_wl);
  req->SetPlanningTask();
  req->LaunchPlanningTask();
  req->generated_tokens = {first_decoded_token};
  req->UpdateAfterInflightTaskFinished();
  req->ResetInflightTask();
  EXPECT_EQ(req->infer_stage, InferStage::kDecode);
  EXPECT_EQ(req->GetInflightCount(), 0u);
  return {request, req};
}

void ArmDecodePlanningWorkload(const std::shared_ptr<InferRequest>& req) {
  ScheduleTaskWorkload wl;
  wl.generated_token_num = 1;
  wl.draft_token_num = 0;
  wl.sampling_token_num = 1;
  req->SetPlanningWorkload(wl);
}

TEST(ScheduleOutputTest, IsLaunchableRespectsMaxDepth) {
  auto fixture = BuildDecodeReadyInferRequest(/*req_id=*/2001);
  auto& req = fixture.infer_req;
  ScheduleOutput output;
  output.running_reqs = {req};

  // 无 inflight 时 slow / fast 均可 launch
  EXPECT_TRUE(output.IsLaunchable(/*max_depth=*/1));
  EXPECT_TRUE(output.IsLaunchable(/*max_depth=*/2));

  // 第一次 decode launch（无 placeholder）
  ArmDecodePlanningWorkload(req);
  output.SetPlanningTask();
  output.LaunchScheduleOutput(/*max_depth=*/2);
  ASSERT_EQ(req->GetInflightCount(), 1u);

  // max_depth=1 已满；max_depth=2 仍可再 push 一步
  EXPECT_FALSE(output.IsLaunchable(/*max_depth=*/1));
  EXPECT_TRUE(output.IsLaunchable(/*max_depth=*/2));

  // 第二次 launch（build-ahead placeholder 步）
  req->generated_tokens.clear();
  ArmDecodePlanningWorkload(req);
  output.SetPlanningTask();
  output.LaunchScheduleOutput(/*max_depth=*/2);
  ASSERT_EQ(req->GetInflightCount(), 2u);
  EXPECT_FALSE(output.IsLaunchable(/*max_depth=*/2));
}

TEST(ScheduleOutputTest, LaunchScheduleOutputUsesPlaceholderWhenInflight) {
  auto fixture = BuildDecodeReadyInferRequest(/*req_id=*/2002, /*first_decoded_token=*/42);
  auto& req = fixture.infer_req;
  ScheduleOutput output;
  output.running_reqs = {req};

  // 首步 decode：无在飞 step，末位应为真实 token
  ArmDecodePlanningWorkload(req);
  output.SetPlanningTask();
  output.LaunchScheduleOutput(/*max_depth=*/2);
  EXPECT_NE(req->forwarding_tokens.back(), InferRequest::kFastPathPlaceholderTokenId);
  EXPECT_EQ(output.req_serialization_snapshots[req->req_id].forwarding_tokens.back(), 42);

  // 二步 decode：已有 inflight，末位写 placeholder
  req->generated_tokens.clear();
  ArmDecodePlanningWorkload(req);
  output.SetPlanningTask();
  output.LaunchScheduleOutput(/*max_depth=*/2);
  EXPECT_EQ(req->forwarding_tokens.back(), InferRequest::kFastPathPlaceholderTokenId);
  EXPECT_EQ(output.req_serialization_snapshots[req->req_id].forwarding_tokens.back(),
            InferRequest::kFastPathPlaceholderTokenId);
}

// 多 req 各自独立 snapshot：后续 launch 只覆盖对应 req_id 的条目。
TEST(ScheduleOutputTest, PerReqSnapshotsIndependentAcrossLaunches) {
  auto fixture_a = BuildDecodeReadyInferRequest(/*req_id=*/2004, /*first_decoded_token=*/11);
  auto fixture_b = BuildDecodeReadyInferRequest(/*req_id=*/2005, /*first_decoded_token=*/22);
  auto& req_a = fixture_a.infer_req;
  auto& req_b = fixture_b.infer_req;
  ScheduleOutput output;
  output.running_reqs = {req_a, req_b};

  ArmDecodePlanningWorkload(req_a);
  ArmDecodePlanningWorkload(req_b);
  output.SetPlanningTask();
  output.LaunchScheduleOutput(/*max_depth=*/2);

  const auto snap_a_1 = output.req_serialization_snapshots[req_a->req_id].forwarding_tokens;
  const auto snap_b_1 = output.req_serialization_snapshots[req_b->req_id].forwarding_tokens;
  ASSERT_FALSE(snap_a_1.empty());
  ASSERT_FALSE(snap_b_1.empty());
  EXPECT_EQ(snap_a_1.back(), 11);
  EXPECT_EQ(snap_b_1.back(), 22);

  // 仅 req_a 进入第二步 placeholder launch（req_b 本步不 schedule，snapshot 应保持第一步）
  req_a->generated_tokens.clear();
  ArmDecodePlanningWorkload(req_a);
  output.running_reqs = {req_a};
  output.SetPlanningTask();
  output.LaunchScheduleOutput(/*max_depth=*/2);

  EXPECT_EQ(output.req_serialization_snapshots[req_a->req_id].forwarding_tokens.back(),
            InferRequest::kFastPathPlaceholderTokenId);
  // req_b 未再 launch，snapshot 保持第一步
  EXPECT_EQ(output.req_serialization_snapshots[req_b->req_id].forwarding_tokens, snap_b_1);
}

TEST(ScheduleOutputTest, SerializePreferSnapshotOverLiveForwardingTokens) {
  auto scheduler = std::make_shared<DummyBatchScheduler>();
  auto schedule_processor = std::make_shared<DummyScheduleProcessor>();
  schedule_processor->Initialize(scheduler);
  ScheduleOutputParser::SetScheduleProcessor(schedule_processor);

  auto fixture = BuildDecodeReadyInferRequest(/*req_id=*/2003, /*first_decoded_token=*/77);
  auto& req = fixture.infer_req;
  ScheduleOutput output;
  output.running_reqs = {req};
  output.schedule_id = 99;

  ArmDecodePlanningWorkload(req);
  output.SetPlanningTask();
  output.LaunchScheduleOutput(/*max_depth=*/2);

  const std::vector<int> snap_fwd = output.req_serialization_snapshots[req->req_id].forwarding_tokens;
  ASSERT_FALSE(snap_fwd.empty());
  EXPECT_EQ(snap_fwd.back(), 77);

  // 模拟后续 schedule 步污染 live 字段；本 output 序列化应仍读 launch 时刻快照
  req->forwarding_tokens.push_back(InferRequest::kFastPathPlaceholderTokenId);
  req->forwarding_tokens.push_back(99999);

  size_t bytes = ScheduleOutputParser::GetSerializedSize(&output);
  std::vector<uint8_t> buffer(bytes);
  size_t serialized_bytes = 0;
  Status status = ScheduleOutputParser::SerializeScheduleOutput(&output, buffer.data(), &serialized_bytes);
  EXPECT_TRUE(status.OK());

  ScheduleOutput dst;
  status = ScheduleOutputParser::DeserializeScheduleOutput(buffer.data(), &dst);
  EXPECT_TRUE(status.OK());
  ASSERT_EQ(dst.executor_running_reqs.size(), 1u);
  EXPECT_EQ(dst.executor_running_reqs[0]->forwarding_tokens, snap_fwd);
  EXPECT_NE(dst.executor_running_reqs[0]->forwarding_tokens.back(), InferRequest::kFastPathPlaceholderTokenId);
}

// stopped req 不参与 inflight capacity 检查，与 EventDrivenStateTest 语义对齐。
TEST(ScheduleOutputTest, IsLaunchableSkipsStoppedReqAtMaxDepth) {
  auto active_fixture = BuildDecodeReadyInferRequest(/*req_id=*/2010);
  auto stopped_fixture = BuildDecodeReadyInferRequest(/*req_id=*/2011);
  auto& active = active_fixture.infer_req;
  auto& stopped = stopped_fixture.infer_req;
  stopped->Stop();

  ScheduleOutput output;
  output.running_reqs = {active, stopped};

  ArmDecodePlanningWorkload(active);
  output.SetPlanningTask();
  output.LaunchScheduleOutput(/*max_depth=*/2);
  ASSERT_EQ(active->GetInflightCount(), 1u);

  // active 在 max_depth=2 下仍可再 launch；stopped 已满 inflight 也不应阻塞。
  EXPECT_TRUE(output.IsLaunchable(/*max_depth=*/2));
}

TEST(ScheduleOutputTest, ScheduleOutputPool) {
  auto scheduler = std::make_shared<DummyBatchScheduler>();
  auto schedule_processor = std::make_shared<DummyScheduleProcessor>();
  schedule_processor->Initialize(scheduler);
  ScheduleOutputParser::SetScheduleProcessor(schedule_processor);

  ScheduleOutputPool schedule_output_pool;
  ScheduleOutput* schedule_output = schedule_output_pool.GetFreeScheduleOutput();
  ASSERT_NE(schedule_output, nullptr);
  *schedule_output = BuildScheduleOutput();

  schedule_output_pool.PutSendScheduleOutput(schedule_output);
  Packet* send_schedule_output_packet = schedule_output_pool.GetSendSerializedPacket();
  ASSERT_NE(send_schedule_output_packet, nullptr);

  ScheduleOutput* new_schedule_output = new ScheduleOutput();
  Status status =
      ScheduleOutputParser::DeserializeScheduleOutput(send_schedule_output_packet->body, new_schedule_output);
  EXPECT_TRUE(status.OK());

  EXPECT_EQ(schedule_output->schedule_id, new_schedule_output->schedule_id);
  EXPECT_TRUE(CheckVectorsVector(schedule_output->finish_req_ids, new_schedule_output->finish_req_ids));

  schedule_output_pool.FreeScheduleOutput(new_schedule_output);
  ScheduleOutput* free_schedule_output = schedule_output_pool.GetFreeScheduleOutput();
  (void)free_schedule_output;
}

}  // namespace
}  // namespace ksana_llm
