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

bool CheckSwapReqBlockInfo(const SwapReqBlockInfo& src, const SwapReqBlockInfo& dst) {
  return src.dev_block_ids == dst.dev_block_ids && src.host_block_ids == dst.host_block_ids;
}

bool CheckSwapReqBlockMaps(const std::vector<std::unordered_map<int64_t, SwapReqBlockInfo>>& src,
                           const std::vector<std::unordered_map<int64_t, SwapReqBlockInfo>>& dst) {
  if (src.size() != dst.size()) {
    return false;
  }
  for (size_t i = 0; i < src.size(); ++i) {
    if (src[i].size() != dst[i].size()) {
      return false;
    }
    for (const auto& [req_id, info] : src[i]) {
      auto it = dst[i].find(req_id);
      if (it == dst[i].end()) {
        return false;
      }
      if (!CheckSwapReqBlockInfo(info, it->second)) {
        return false;
      }
    }
  }
  return true;
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
  infer_req->kv_comm_request_id = 1234;
  infer_req->kv_comm_group_key = "group_key";
  infer_req->forwarding_tokens = {601, 603, 605};
  infer_req->sampling_result_tokens = {701};
  infer_req->infer_stage = InferStage::kDecode;
  infer_req->step = 5;
  infer_req->kv_cached_token_num = 2;
  infer_req->kv_cache_blocks = {11, 13, 21};
  infer_req->block_token_num = 16;
  infer_req->logits_offset = 7;
  infer_req->swap_pending = true;
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
  output.merged_swapout_req_ids = {{11, 13}, {15}};
  output.merged_swapin_req_ids = {{12, 14}, {16}};

  std::unordered_map<int64_t, SwapReqBlockInfo> swapout_map;
  swapout_map[101] = {{111, 113}, {211, 213}};
  swapout_map[102] = {{115}, {215}};
  output.swapout_req_block_ids.push_back(swapout_map);

  std::unordered_map<int64_t, SwapReqBlockInfo> swapin_map;
  swapin_map[201] = {{121, 123}, {221, 223}};
  swapin_map[202] = {{125}, {225}};
  output.swapin_req_block_ids.push_back(swapin_map);

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
  EXPECT_TRUE(CheckVectorsVector(src.merged_swapout_req_ids, dst.merged_swapout_req_ids));
  EXPECT_TRUE(CheckVectorsVector(src.merged_swapin_req_ids, dst.merged_swapin_req_ids));
  EXPECT_TRUE(CheckSwapReqBlockMaps(src.swapout_req_block_ids, dst.swapout_req_block_ids));
  EXPECT_TRUE(CheckSwapReqBlockMaps(src.swapin_req_block_ids, dst.swapin_req_block_ids));
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
  EXPECT_EQ(exec_req->swap_pending, full_req->swap_pending);
  EXPECT_EQ(exec_req->prefix_cache_len, full_req->prefix_cache_len);
  EXPECT_EQ(exec_req->infer_stage, full_req->infer_stage);
  EXPECT_EQ(exec_req->step, full_req->step);
  EXPECT_EQ(exec_req->kv_cached_token_num, full_req->kv_cached_token_num);
  EXPECT_EQ(exec_req->kv_comm_request_id, full_req->kv_comm_request_id);
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
  EXPECT_TRUE(CheckVectorsVector(schedule_output->merged_swapout_req_ids, new_schedule_output->merged_swapout_req_ids));
  EXPECT_TRUE(CheckVectorsVector(schedule_output->merged_swapin_req_ids, new_schedule_output->merged_swapin_req_ids));
  EXPECT_TRUE(
      CheckSwapReqBlockMaps(schedule_output->swapout_req_block_ids, new_schedule_output->swapout_req_block_ids));
  EXPECT_TRUE(CheckSwapReqBlockMaps(schedule_output->swapin_req_block_ids, new_schedule_output->swapin_req_block_ids));

  schedule_output_pool.FreeScheduleOutput(new_schedule_output);
  ScheduleOutput* free_schedule_output = schedule_output_pool.GetFreeScheduleOutput();
  (void)free_schedule_output;
}

}  // namespace
}  // namespace ksana_llm
