/* Copyright 2024 Tencent Inc.  All rights reserved.
==============================================================================*/

#include <gtest/gtest.h>

#include <limits>
#include <set>

#include "connector/task_key.h"
#include "endpoints/streaming/streaming_iterator.h"
#include "runtime/infer_request.h"
#include "utils/id_generator.h"
#include "utils/request.h"
#include "utils/status.h"

namespace ksana_llm {

class RequestTest : public testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

// Test for `VerifyArgs()` of `SamplingConfig`.
TEST_F(RequestTest, SamplingConfig) {
  SamplingConfig sampling_config;
  EXPECT_TRUE(sampling_config.VerifyArgs().OK());

  sampling_config.topp = 0.f;
  EXPECT_TRUE(sampling_config.VerifyArgs().OK());
  EXPECT_EQ(sampling_config.topp, 1.f);

  sampling_config.temperature = 0.f;
  EXPECT_TRUE(sampling_config.VerifyArgs().OK());
  EXPECT_EQ(sampling_config.temperature, 1.f);

  sampling_config.topk = 0;
  EXPECT_EQ(sampling_config.VerifyArgs().GetCode(), RET_INVALID_ARGUMENT);

  sampling_config.topk = 1234;
  EXPECT_EQ(sampling_config.VerifyArgs().GetCode(), RET_INVALID_ARGUMENT);

  sampling_config.topk = 1024;
  sampling_config.no_repeat_ngram_size = sampling_config.encoder_no_repeat_ngram_size = 1;
  EXPECT_EQ(sampling_config.VerifyArgs().GetCode(), RET_INVALID_ARGUMENT);
}

TEST_F(RequestTest, RequestStatus) {
  std::shared_ptr<KsanaPythonInput> ksana_python_input = std::make_shared<KsanaPythonInput>();
  std::shared_ptr<std::unordered_map<std::string, std::string>> req_ctx =
      std::make_shared<std::unordered_map<std::string, std::string>>();

  std::shared_ptr<Request> request = std::make_shared<Request>(ksana_python_input, req_ctx);
  request->finish_status = Status(RET_SUCCESS, "success");
  auto ksana_python_output = KsanaPythonOutput(request);
  EXPECT_EQ(ksana_python_output.finish_status.GetCode(), RetCode::RET_SUCCESS);
}

// Test for prefix_cache_hit_tokens in KsanaPythonOutput
TEST_F(RequestTest, PrefixCacheHitTokens) {
  std::shared_ptr<KsanaPythonInput> ksana_python_input = std::make_shared<KsanaPythonInput>();
  std::shared_ptr<std::unordered_map<std::string, std::string>> req_ctx =
      std::make_shared<std::unordered_map<std::string, std::string>>();

  // Test case 1: No prefix_cache_hit_tokens in req_ctx
  {
    std::shared_ptr<Request> request = std::make_shared<Request>(ksana_python_input, req_ctx);
    auto ksana_python_output = KsanaPythonOutput(request);
    EXPECT_EQ(ksana_python_output.prefix_cache_hit_tokens, 0);
  }

  // Test case 2: With prefix_cache_hit_tokens in req_ctx
  {
    (*req_ctx)["prefix_cache_hit_tokens"] = "42";
    std::shared_ptr<Request> request = std::make_shared<Request>(ksana_python_input, req_ctx);
    auto ksana_python_output = KsanaPythonOutput(request);
    EXPECT_EQ(ksana_python_output.prefix_cache_hit_tokens, 42);
  }

  // Test case 3: With different prefix_cache_hit_tokens value
  {
    (*req_ctx)["prefix_cache_hit_tokens"] = "128";
    std::shared_ptr<Request> request = std::make_shared<Request>(ksana_python_input, req_ctx);
    auto ksana_python_output = KsanaPythonOutput(request);
    EXPECT_EQ(ksana_python_output.prefix_cache_hit_tokens, 128);
  }

  // Test case 4: req_ctx is nullptr
  {
    std::shared_ptr<Request> request = std::make_shared<Request>(ksana_python_input, nullptr);
    auto ksana_python_output = KsanaPythonOutput(request);
    EXPECT_EQ(ksana_python_output.prefix_cache_hit_tokens, 0);
  }
}

// Test for StreamingIterator with prefix_cache_hit_tokens
TEST_F(RequestTest, StreamingIteratorPrefixCacheHitTokens) {
  std::shared_ptr<KsanaPythonInput> ksana_python_input = std::make_shared<KsanaPythonInput>();
  ksana_python_input->input_tokens = {1, 2, 3, 4, 5};
  ksana_python_input->sampling_config.logprobs_num = 0;

  std::shared_ptr<std::unordered_map<std::string, std::string>> req_ctx =
      std::make_shared<std::unordered_map<std::string, std::string>>();
  (*req_ctx)["prefix_cache_hit_tokens"] = "50";

  std::shared_ptr<Request> request = std::make_shared<Request>(ksana_python_input, req_ctx);
  // Initialize required members for StreamingIterator
  request->finisheds = {true};
  request->abort_waiter = std::make_shared<Waiter>(0);
  request->output_group.push_back(
      std::make_tuple(std::vector<int>{1, 2, 3, 4, 5, 6, 7}, std::vector<std::vector<std::pair<int, float>>>(), 0.0f));

  StreamingIterator iterator(request, ksana_python_input);

  KsanaPythonOutput output;
  bool has_output = iterator.AddOutput(output);

  EXPECT_TRUE(has_output);
  EXPECT_EQ(output.prefix_cache_hit_tokens, 50);
  EXPECT_EQ(output.input_tokens.size(), 5);
}

// ==================== IdGenerator encoding tests ====================

TEST_F(RequestTest, IdGeneratorLowBitsAreZero) {
  IdGenerator gen;
  constexpr int kSubIdBits = IdGenerator::kSubIdBits;
  constexpr int64_t kLowBitsMask = (1LL << kSubIdBits) - 1;

  for (int i = 0; i < 100; ++i) {
    int64_t id = gen.Gen();
    EXPECT_EQ(id & kLowBitsMask, 0) << "id=" << id << " has non-zero low " << kSubIdBits << " bits";
    EXPECT_GE(id, 0) << "id should be non-negative";
  }
}

TEST_F(RequestTest, IdGeneratorHighBitsIncrement) {
  IdGenerator gen;
  constexpr int kSubIdBits = IdGenerator::kSubIdBits;

  int64_t prev = gen.Gen();
  for (int i = 0; i < 100; ++i) {
    int64_t curr = gen.Gen();
    EXPECT_EQ((curr >> kSubIdBits) - (prev >> kSubIdBits), 1)
        << "High bits should increment by 1 between consecutive Gen() calls";
    prev = curr;
  }
}

// ==================== Request: output_group == 1 (default) ====================

TEST_F(RequestTest, ReqIdAutoGeneratedWithoutHeader) {
  std::shared_ptr<KsanaPythonInput> ksana_python_input = std::make_shared<KsanaPythonInput>();
  ksana_python_input->input_tokens = {1, 2, 3};

  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();

  std::shared_ptr<Request> request = std::make_shared<Request>(ksana_python_input, req_ctx);

  // req_id should equal req_ids[0]
  EXPECT_EQ(request->req_id, request->req_ids[0]);
  EXPECT_NE(request->req_id, 0);
  // Low 8 bits should be 0
  EXPECT_EQ(request->req_id & 0xFF, 0);
}

TEST_F(RequestTest, ReqIdAutoGeneratedWithNullCtx) {
  std::shared_ptr<KsanaPythonInput> ksana_python_input = std::make_shared<KsanaPythonInput>();
  ksana_python_input->input_tokens = {1, 2, 3};

  std::shared_ptr<Request> request = std::make_shared<Request>(ksana_python_input, nullptr);

  EXPECT_EQ(request->req_id, request->req_ids[0]);
}

TEST_F(RequestTest, ReqIdFromRequestIdHeader) {
  std::shared_ptr<KsanaPythonInput> ksana_python_input = std::make_shared<KsanaPythonInput>();
  ksana_python_input->input_tokens = {1, 2, 3};

  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  // Use a value with low bits already cleared (as Python router would produce)
  const int64_t header_value = 9876543LL << IdGenerator::kSubIdBits;
  (*req_ctx)["request-id"] = std::to_string(header_value);

  std::shared_ptr<Request> request = std::make_shared<Request>(ksana_python_input, req_ctx);

  EXPECT_EQ(request->req_id, header_value);
  EXPECT_EQ(request->req_ids[0], header_value);
}

TEST_F(RequestTest, ReqIdLargeInt64FromHeader) {
  std::shared_ptr<KsanaPythonInput> ksana_python_input = std::make_shared<KsanaPythonInput>();
  ksana_python_input->input_tokens = {1, 2, 3};

  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  const int64_t header_value = 5000000000LL << IdGenerator::kSubIdBits;
  (*req_ctx)["request-id"] = std::to_string(header_value);

  std::shared_ptr<Request> request = std::make_shared<Request>(ksana_python_input, req_ctx);

  EXPECT_EQ(request->req_id, header_value);
  EXPECT_GT(request->req_id, static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
}

// Test that kv_comm_group_key is parsed from "kv-comm-group-key" header.
TEST_F(RequestTest, KvCommGroupKeyFromHeader) {
  std::shared_ptr<KsanaPythonInput> ksana_python_input = std::make_shared<KsanaPythonInput>();
  ksana_python_input->input_tokens = {1, 2, 3};

  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  (*req_ctx)["kv-comm-group-key"] = "prefill_node__decode_node";

  std::shared_ptr<Request> request = std::make_shared<Request>(ksana_python_input, req_ctx);

  EXPECT_EQ(request->kv_comm_group_key, "prefill_node__decode_node");
}

// Test that kv_comm_group_key is empty when no header is present.
TEST_F(RequestTest, KvCommGroupKeyEmptyWithoutHeader) {
  std::shared_ptr<KsanaPythonInput> ksana_python_input = std::make_shared<KsanaPythonInput>();
  ksana_python_input->input_tokens = {1, 2, 3};

  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();

  std::shared_ptr<Request> request = std::make_shared<Request>(ksana_python_input, req_ctx);

  EXPECT_EQ(request->kv_comm_group_key, "");
}

// ==================== Request: output_group > 1 (beam search / multi-return) ====================

TEST_F(RequestTest, MultiOutputGroupReqIdsUnique) {
  std::shared_ptr<KsanaPythonInput> ksana_python_input = std::make_shared<KsanaPythonInput>();
  ksana_python_input->input_tokens = {1, 2, 3};
  ksana_python_input->sampling_config.num_beams = 4;

  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  std::shared_ptr<Request> request = std::make_shared<Request>(ksana_python_input, req_ctx);

  // req_ids size matches output_group size
  EXPECT_EQ(request->req_ids.size(), 4u);

  // Each req_ids[i] == req_id | i
  for (size_t i = 0; i < request->req_ids.size(); ++i) {
    EXPECT_EQ(request->req_ids[i], request->req_id | static_cast<int64_t>(i));
  }

  // req_ids[0] == req_id (branch 0 has same id as request-level id)
  EXPECT_EQ(request->req_ids[0], request->req_id);

  // All req_ids are unique
  std::set<int64_t> id_set(request->req_ids.begin(), request->req_ids.end());
  EXPECT_EQ(id_set.size(), request->req_ids.size());

  // Can recover base id from any req_ids[i]
  for (size_t i = 0; i < request->req_ids.size(); ++i) {
    EXPECT_EQ(request->req_ids[i] & ~0xFF, request->req_id);
  }
}

TEST_F(RequestTest, MultiOutputGroupWithHeaderOverride) {
  std::shared_ptr<KsanaPythonInput> ksana_python_input = std::make_shared<KsanaPythonInput>();
  ksana_python_input->input_tokens = {1, 2, 3};
  ksana_python_input->sampling_config.num_return_sequences = 3;

  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  const int64_t header_value = 12345LL << IdGenerator::kSubIdBits;
  (*req_ctx)["request-id"] = std::to_string(header_value);

  std::shared_ptr<Request> request = std::make_shared<Request>(ksana_python_input, req_ctx);

  EXPECT_EQ(request->req_id, header_value);
  EXPECT_EQ(request->req_ids.size(), 3u);

  for (size_t i = 0; i < request->req_ids.size(); ++i) {
    EXPECT_EQ(request->req_ids[i], header_value | static_cast<int64_t>(i));
  }

  // All req_ids are unique
  std::set<int64_t> id_set(request->req_ids.begin(), request->req_ids.end());
  EXPECT_EQ(id_set.size(), request->req_ids.size());
}

// ==================== Invalid header fallback ====================

TEST_F(RequestTest, InvalidRequestIdHeaderFallsBack) {
  std::shared_ptr<KsanaPythonInput> ksana_python_input = std::make_shared<KsanaPythonInput>();
  ksana_python_input->input_tokens = {1, 2, 3};

  // Non-numeric header should not crash, should fall back to auto-generated id
  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  (*req_ctx)["request-id"] = "not_a_number";

  std::shared_ptr<Request> request;
  EXPECT_NO_THROW(request = std::make_shared<Request>(ksana_python_input, req_ctx));
  EXPECT_NE(request->req_id, 0);
  EXPECT_EQ(request->req_id & 0xFF, 0);  // auto-generated, low bits zero
  EXPECT_EQ(request->req_id, request->req_ids[0]);
}

TEST_F(RequestTest, EmptyRequestIdHeaderFallsBack) {
  std::shared_ptr<KsanaPythonInput> ksana_python_input = std::make_shared<KsanaPythonInput>();
  ksana_python_input->input_tokens = {1, 2, 3};

  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  (*req_ctx)["request-id"] = "";

  std::shared_ptr<Request> request;
  EXPECT_NO_THROW(request = std::make_shared<Request>(ksana_python_input, req_ctx));
  EXPECT_NE(request->req_id, 0);
}

// ==================== InferRequest per-branch uniqueness ====================

TEST_F(RequestTest, InferRequestReqIdFromRequestReqId) {
  std::shared_ptr<KsanaPythonInput> ksana_python_input = std::make_shared<KsanaPythonInput>();
  ksana_python_input->input_tokens = {1, 2, 3};
  ksana_python_input->sampling_config.logprobs_num = 0;

  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  const int64_t header_value = 7777777LL << IdGenerator::kSubIdBits;
  (*req_ctx)["request-id"] = std::to_string(header_value);

  std::shared_ptr<Request> request = std::make_shared<Request>(ksana_python_input, req_ctx);

  // Verify the Request's req_id is from header
  EXPECT_EQ(request->req_id, header_value);

  // Build InferRequest
  request->finisheds = {false};
  request->abort_waiter = std::make_shared<Waiter>(0);
  auto infer_request = std::make_shared<InferRequest>(request, 0);

  // InferRequest::req_id should match request->req_ids[0] == request->req_id (index 0)
  EXPECT_EQ(infer_request->req_id, header_value);
  EXPECT_EQ(infer_request->req_id, request->req_id);
  EXPECT_EQ(infer_request->req_id, request->req_ids[0]);
}

TEST_F(RequestTest, InferRequestMultiBranchUnique) {
  std::shared_ptr<KsanaPythonInput> ksana_python_input = std::make_shared<KsanaPythonInput>();
  ksana_python_input->input_tokens = {1, 2, 3};
  ksana_python_input->sampling_config.num_beams = 4;
  ksana_python_input->sampling_config.logprobs_num = 0;

  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  std::shared_ptr<Request> request = std::make_shared<Request>(ksana_python_input, req_ctx);

  request->finisheds = std::deque<bool>(4, false);
  request->abort_waiter = std::make_shared<Waiter>(0);

  std::set<int64_t> infer_req_ids;
  for (int i = 0; i < 4; ++i) {
    auto infer_request = std::make_shared<InferRequest>(request, i);
    EXPECT_EQ(infer_request->req_id, request->req_ids[i]);
    infer_req_ids.insert(infer_request->req_id);
  }

  // All InferRequest req_ids are unique
  EXPECT_EQ(infer_req_ids.size(), 4u);
}

TEST_F(RequestTest, InferRequestMultiBranchWithHeaderOverride) {
  std::shared_ptr<KsanaPythonInput> ksana_python_input = std::make_shared<KsanaPythonInput>();
  ksana_python_input->input_tokens = {1, 2, 3};
  ksana_python_input->sampling_config.num_beams = 3;
  ksana_python_input->sampling_config.logprobs_num = 0;

  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  const int64_t header_value = 55555LL << IdGenerator::kSubIdBits;
  (*req_ctx)["request-id"] = std::to_string(header_value);

  std::shared_ptr<Request> request = std::make_shared<Request>(ksana_python_input, req_ctx);

  request->finisheds = std::deque<bool>(3, false);
  request->abort_waiter = std::make_shared<Waiter>(0);

  std::set<int64_t> infer_req_ids;
  for (int i = 0; i < 3; ++i) {
    auto infer_request = std::make_shared<InferRequest>(request, i);
    EXPECT_EQ(infer_request->req_id, header_value | static_cast<int64_t>(i));
    infer_req_ids.insert(infer_request->req_id);
  }

  EXPECT_EQ(infer_req_ids.size(), 3u);
}

// ==================== output_group upper limit protection ====================

TEST_F(RequestTest, OutputGroupExceedsMaxThrows) {
  std::shared_ptr<KsanaPythonInput> ksana_python_input = std::make_shared<KsanaPythonInput>();
  ksana_python_input->input_tokens = {1, 2, 3};
  // 256 exceeds the max (255)
  ksana_python_input->sampling_config.num_beams = 256;

  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  EXPECT_THROW(std::make_shared<Request>(ksana_python_input, req_ctx), std::runtime_error);
}

TEST_F(RequestTest, OutputGroupAtMaxSucceeds) {
  std::shared_ptr<KsanaPythonInput> ksana_python_input = std::make_shared<KsanaPythonInput>();
  ksana_python_input->input_tokens = {1, 2, 3};
  // 255 is exactly at the max
  ksana_python_input->sampling_config.num_beams = 255;

  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  std::shared_ptr<Request> request;
  EXPECT_NO_THROW(request = std::make_shared<Request>(ksana_python_input, req_ctx));
  EXPECT_EQ(request->req_ids.size(), 255u);

  // Verify all are unique
  std::set<int64_t> id_set(request->req_ids.begin(), request->req_ids.end());
  EXPECT_EQ(id_set.size(), 255u);
}

// ==================== InferRequest::ScheduleStateToStr coverage ====================

TEST_F(RequestTest, ScheduleStateToStrContainsRequestId) {
  std::shared_ptr<KsanaPythonInput> ksana_python_input = std::make_shared<KsanaPythonInput>();
  ksana_python_input->input_tokens = {1, 2, 3};

  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  const int64_t header_value = 42LL << IdGenerator::kSubIdBits;
  (*req_ctx)["request-id"] = std::to_string(header_value);
  (*req_ctx)["kv-comm-group-key"] = "test_group";

  std::shared_ptr<Request> request = std::make_shared<Request>(ksana_python_input, req_ctx);
  request->finisheds = {false};
  request->abort_waiter = std::make_shared<Waiter>(0);

  auto infer_request = std::make_shared<InferRequest>(request, 0);
  std::string state_str = infer_request->ScheduleStateToStr();

  EXPECT_NE(state_str.find("request_id="), std::string::npos);
  EXPECT_NE(state_str.find(std::to_string(header_value)), std::string::npos);
  EXPECT_NE(state_str.find("kv_comm_group_key=test_group"), std::string::npos);
}

// ==================== TaskKey int64_t coverage ====================

TEST_F(RequestTest, TaskKeyInt64Construction) {
  int64_t large_id = 5000000000LL;
  TaskKey key(large_id, 1, 2, 3, 1024);

  EXPECT_EQ(key.req_id, large_id);
  EXPECT_EQ(key.block_idx, 1);
  EXPECT_EQ(key.layer_idx, 2);
  EXPECT_EQ(key.hash_device_id, 3);
  EXPECT_EQ(key.tensor_size, 1024);
}

TEST_F(RequestTest, TaskKeyHashInt64) {
  TaskKey::Hash hasher;

  TaskKey key1(5000000000LL, 1, 2, 3, 1024);
  TaskKey key2(5000000001LL, 1, 2, 3, 1024);
  TaskKey key3(5000000000LL, 1, 2, 3, 1024);

  EXPECT_EQ(hasher(key1), hasher(key3));
  EXPECT_NE(hasher(key1), hasher(key2));
}

TEST_F(RequestTest, TaskKeyHashCompare) {
  TaskKey key1(9999999999LL, 1, 2, 3, 1024);
  TaskKey key2(9999999999LL, 1, 2, 3, 1024);

  EXPECT_TRUE(TaskKey::HashCompare::equal(key1, key2));
  EXPECT_EQ(TaskKey::HashCompare::hash(key1), TaskKey::HashCompare::hash(key2));
}

TEST_F(RequestTest, TaskKeyShardIndexInt64) {
  int64_t large_id = 5000000000LL;
  TaskKey key(large_id, 1, 2, 3, 1024);

  size_t shard = key.GetShardIndex(8);
  EXPECT_LT(shard, 8u);
  EXPECT_EQ(shard, static_cast<size_t>(large_id) % 8);
}

}  // namespace ksana_llm
