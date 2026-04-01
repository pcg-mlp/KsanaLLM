/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "runtime/draft_generator/trie_generator.h"

#include <gtest/gtest.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include "configure/environment.h"
#include "runtime/draft_generator/draft_generator_controller.h"
#include "runtime/executor_infer_request.h"

namespace ksana_llm {

class TrieTreeFriendTest : public TrieTree {
 public:
  using TrieTree::TrieTree;
  size_t GetMaxNodes() const { return max_nodes_; }

  void Display(const Node* root) const {
    if (!root) {
      std::cout << "Tree is empty." << std::endl;
      return;
    }
    std::cout << "Trie Tree Structure:" << std::endl;
    DisplayHelper(root, 0);
  }

  void DisplayHelper(const Node* node, int indent) const {
    // First level indentation
    for (int i = 0; i < indent; ++i) std::cout << "  ";
    std::cout << "Token: " << node->token_id << std::endl;

    // Display input frequencies
    for (int i = 0; i < indent; ++i) std::cout << "  ";
    std::cout << "User Frequencies:" << std::endl;

    std::vector<std::pair<int, double>> sorted_freqs(node->input_freqs.begin(), node->input_freqs.end());
    std::sort(sorted_freqs.begin(), sorted_freqs.end());

    for (const auto& [key, freq] : sorted_freqs) {
      for (int i = 0; i < indent; ++i) std::cout << "  ";
      std::cout << "  " << key << ": " << std::fixed << std::setprecision(4) << freq << std::endl;
    }

    // Display output frequency
    for (int i = 0; i < indent; ++i) std::cout << "  ";
    std::cout << "Output Frequency: " << std::fixed << std::setprecision(4) << node->output_freq << std::endl;

    // Recursively display children nodes
    for (int i = 0; i < indent; ++i) std::cout << "  ";
    std::cout << "Children:" << std::endl;
    if (node->children.empty()) {
      for (int i = 0; i < indent; ++i) std::cout << "  ";
      std::cout << "  None" << std::endl;
    } else {
      for (const auto& [token, child] : node->children) {
        DisplayHelper(child, indent + 2);
      }
    }
  }
};

class TrieTreeTest : public testing::Test {
 protected:
  void SetUp() override { trie_ = std::make_shared<TrieTreeFriendTest>(); }

  void TearDown() override { trie_->Reset(); }

  std::shared_ptr<TrieTreeFriendTest> trie_;
};

TEST_F(TrieTreeTest, PutBasic) {
  std::vector<int> tokens = {1, 2, 3};
  trie_->Put(tokens, 0, 3, true, TrieTree::FrequencyType::INPUT, 0);

  EXPECT_EQ(trie_->GetCurrentNodes(), 4);
  auto node = trie_->GetRoot();
  for (int token : tokens) {
    auto it = node->children.find(token);
    ASSERT_NE(it, node->children.end());
    EXPECT_DOUBLE_EQ(it->second->input_freqs.at(0), 1.0);
    node = it->second;
  }
}

TEST_F(TrieTreeTest, QueryOneBranchWithThreshold) {
  // Prepare test data
  const std::vector<int> path1 = {1, 2, 3};
  const std::vector<int> path2 = {1, 2, 4};
  trie_->Put(path1, 0, 3, true, TrieTree::FrequencyType::INPUT, 0);
  trie_->Put(path2, 0, 3, true, TrieTree::FrequencyType::INPUT, 0);

  // Execute query
  const std::vector<int> token_ids = {1, 2};
  std::vector<int> result_ids;
  trie_->QueryOneBranch(token_ids, 0, 2, 0, result_ids, 10, 1.0);

  // Verify results
  ASSERT_EQ(result_ids.size(), 1);
  EXPECT_TRUE(result_ids[0] == 3 || result_ids[0] == 4);
}

TEST_F(TrieTreeTest, DFSDelete) {
  // Prepare test data
  const std::vector<int> path1 = {1, 2, 3};
  const std::vector<int> path2 = {1, 2, 4};
  trie_->Put(path1, 0, 3, true, TrieTree::FrequencyType::INPUT, 0);
  trie_->Put(path2, 0, 3, true, TrieTree::FrequencyType::INPUT, 0);

  // Execute deletion
  trie_->DFSDelete(trie_->GetRoot(), 1);
  EXPECT_EQ(trie_->GetCurrentNodes(), 1);
}

TEST_F(TrieTreeTest, ResetInputFreqs) {
  // Prepare test data
  const std::vector<int> tokens = {1, 2};
  trie_->Put(tokens, 0, 2, true, TrieTree::FrequencyType::INPUT, 0);
  trie_->Put(tokens, 0, 2, true, TrieTree::FrequencyType::INPUT, 0);

  // Execute reset
  trie_->ResetInputFreqs(0);

  // Verify reset results
  auto node = trie_->GetRoot()->children.at(1)->children.at(2);
  EXPECT_EQ(node->input_freqs.count(0), 0);
}

class TrieGeneratorTest : public testing::Test {
 protected:
  void SetUp() override {
    generator_ = std::make_shared<TrieGenerator>();

    // 创建 ExecutorInferRequest 用于测试
    request_ = std::make_shared<ExecutorInferRequest>();
    request_->req_id = 1;
  }

  void TearDown() override { generator_->GetTrie()->Reset(); }

  std::shared_ptr<TrieGenerator> generator_;
  std::shared_ptr<ExecutorInferRequest> request_;
};

TEST_F(TrieGeneratorTest, StreamPutAndPredict) {
  // Prepare test data
  std::vector<int> tokens1;
  for (int i = 0; i < 50; ++i) tokens1.push_back(i);

  std::vector<int> tokens2;
  for (int i = 2; i <= 50; i += 2) tokens2.push_back(i);
  // Execute operations
  generator_->StreamPut(tokens1, 0, false, TrieTree::FrequencyType::OUTPUT, 0);
  generator_->StreamPut(tokens2, 0, false, TrieTree::FrequencyType::OUTPUT, 0);

  std::vector<int> spec_ids;
  generator_->Predict({1001, 1003, 1005, 2}, spec_ids, 10, 0, "Hierarchy", 1.0);

  // Verify results
  EXPECT_EQ(spec_ids.size(), 3);
}

TEST_F(TrieGeneratorTest, PredictOneBranchSuccess) {
  std::vector<int> tokens1;
  for (int i = 2; i <= 50; i++) tokens1.emplace_back(i);
  std::vector<int> tokens2;
  for (int i = 2; i <= 50; i += 2) tokens2.emplace_back(i);
  generator_->StreamPut(tokens1, 0, false, TrieTree::FrequencyType::OUTPUT, 0);
  generator_->StreamPut(tokens2, 0, false, TrieTree::FrequencyType::OUTPUT, 0);
  std::vector<int> token_ids = {1, 2, 3, 4, 5, 6};
  std::vector<int> spec_ids;
  generator_->Predict(token_ids, spec_ids, 10, 0, "OneBranch", 1.0);
  EXPECT_EQ(spec_ids.size(), 10);
  EXPECT_EQ(spec_ids[0], 7);
}

TEST_F(TrieGeneratorTest, PredictEmpty) {
  std::vector<int> tokens1;
  for (int i = 0; i <= 10; i++) {
    tokens1.emplace_back(i);
  }
  std::vector<int> token_ids = {99999};
  std::vector<int> spec_ids;
  generator_->Predict(token_ids, spec_ids, 10, 0, "OneBranch", 1.0);
  EXPECT_TRUE(spec_ids.empty());

  generator_->Predict({}, spec_ids, 10, 0, "OneBranch", 1.0);
  EXPECT_TRUE(spec_ids.empty());

  generator_->Predict({}, spec_ids, 0, 0, "OneBranch", 1.0);
  EXPECT_TRUE(spec_ids.empty());
}

TEST_F(TrieGeneratorTest, BasicTest) {
  request_->step = 1;
  request_->forwarding_tokens_draft_num = 0;
  request_->forwarding_tokens = {1001, 1003, 1005, 10, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1005, 10};
  request_->accepted_tokens = {};
  request_->generated_tokens = {2};
  request_->draft_tokens.clear();

  generator_->GenerateDraft(request_);
  std::vector<int> draft_tokens = request_->draft_tokens.GetDraftTokens();

  EXPECT_EQ(9, draft_tokens.size());
  EXPECT_EQ(3, draft_tokens[0]);
}

TEST_F(TrieGeneratorTest, HitRateTest) {
  const std::string csv_path = "speculative_decoding_accuracy.csv";
  int total_hits = 0;
  int total_tokens = 0;
  std::ifstream file(csv_path);
  if (!file.is_open()) {
    GTEST_SKIP() << "Skipping test: Missing required CSV file " << csv_path;
    return;
  }
  std::string line;
  std::getline(file, line);
  int req = 0;
  while (std::getline(file, line)) {
    std::istringstream iss(line);
    std::string prefix_str, response_str;
    std::getline(iss, prefix_str, ',');
    std::getline(iss, response_str);
    auto parse_tokens = [](const std::string& str) {
      std::vector<int> tokens;
      std::istringstream token_stream(str);
      int token;
      while (token_stream >> token) {
        tokens.push_back(token);
      }
      return tokens;
    };
    const auto prefix = parse_tokens(prefix_str);
    const auto response = parse_tokens(response_str);
    // Simulate incremental decoding process
    request_->forwarding_tokens = prefix;
    size_t pos = 0;
    request_->step = 1;
    request_->accepted_tokens.clear();
    while (pos < response.size()) {
      generator_->GenerateDraft(request_);
      std::vector<int> draft_tokens = request_->draft_tokens.GetDraftTokens();
      request_->step += 1;
      total_tokens += draft_tokens.size();
      int accepted_token = 0;
      for (size_t i = 0; i < draft_tokens.size(); i++) {
        if (pos < response.size() && draft_tokens[i] == response[pos]) {
          request_->forwarding_tokens.push_back(response[pos]);
          ++total_hits;
          ++pos;
          ++accepted_token;
        } else {
          break;
        }
      }
      std::cout << " draft_tokens.size() = " << draft_tokens.size() << " accepted_token = " << accepted_token
                << std::endl;
      if (pos < response.size()) {
        request_->forwarding_tokens.push_back(response[pos]);
        pos++;
      }
    }
    const double actual_hit_rate = total_tokens > 0 ? static_cast<double>(total_hits) / total_tokens : 0.0;
    std::cout << "Hit tokens: " << total_hits << "\n";
    std::cout << "Total tokens: " << total_tokens << "\n";
    std::cout << "Hit rate: " << actual_hit_rate * 100 << "%\n";
    ++req;
  }
}

// Tests for LlmRuntime::DraftTokenFilter
// Mock GrammarMatcherWrapper for testing
class MockGrammarMatcher : public GrammarMatcherWrapper {
 public:
  explicit MockGrammarMatcher(const std::vector<int>& accepted_tokens)
      : GrammarMatcherWrapper(nullptr), accepted_tokens_(accepted_tokens) {}

  bool FillNextTokenBitmask(void* bitmask_data, int batch_index = 0) override { return true; }
  bool IsTerminated() const override { return false; }
  bool IsInitialized() const override { return true; }

  void Rollback(int rollback_token_num) override {
    // Mock implementation: do nothing for testing
  }

  bool AcceptToken(int token_id) override {
    // 只接受在 accepted_tokens_ 列表中的 token
    return std::find(accepted_tokens_.begin(), accepted_tokens_.end(), token_id) != accepted_tokens_.end();
  }

 private:
  std::vector<int> accepted_tokens_;
};

class DraftTokenFilterTest : public testing::Test {
 protected:
  void SetUp() override {
    runtime_config_ = RuntimeConfig();
    batch_scheduler_config_ = BatchSchedulerConfig();
    context_ = std::make_shared<Context>();  // tp_size=1, attn_dp_size=1, max_multi_batch=1
  }

  std::shared_ptr<ExecutorInferRequest> CreateMockInferRequest(
      const std::vector<int>& draft_tokens_mtp, const std::vector<int>& draft_tokens_trie,
      const std::vector<int>& sampling_result_tokens, const std::vector<int>& stop_token_ids = {},
      std::shared_ptr<GrammarMatcherWrapper> grammar_matcher = nullptr) {
    auto executor_req = std::make_shared<ExecutorInferRequest>();

    executor_req->req_id = 1;
    executor_req->input_tokens = {1, 2, 3};
    executor_req->output_tokens = {};
    executor_req->logprobs = {};
    executor_req->sampling_config.stop_token_ids = stop_token_ids;
    executor_req->finished = false;
    executor_req->aborted = false;
    executor_req->grammar_matcher = grammar_matcher;

    executor_req->draft_tokens.mtp = draft_tokens_mtp;
    executor_req->draft_tokens.spec = draft_tokens_trie;
    executor_req->sampling_result_tokens = sampling_result_tokens;

    return executor_req;
  }

  RuntimeConfig runtime_config_;
  BatchSchedulerConfig batch_scheduler_config_;
  std::shared_ptr<Context> context_;
};

TEST_F(DraftTokenFilterTest, GrammarMatcherRejectsNewToken) {
  // 测试 grammar_matcher 拒绝新生成 token 的情况
  std::vector<int> draft_mtp = {10};
  std::vector<int> draft_trie = {20, 30};
  std::vector<int> sampling_result = {10, 20, 30, 40};

  auto grammar_matcher = std::make_shared<MockGrammarMatcher>(std::vector<int>{20});
  auto req = CreateMockInferRequest(draft_mtp, draft_trie, sampling_result, {}, grammar_matcher);
  std::vector<std::shared_ptr<ExecutorInferRequest>> reqs = {req};

  // 调用 DefaultFilterDraftTokens 来处理 draft tokens
  size_t accepted_num = DefaultFilterDraftTokens(req);

  EXPECT_EQ(accepted_num, 1);  // 只有第一个 draft token (10) 被接受
  EXPECT_EQ(req->accepted_tokens.size(), 1);
  EXPECT_EQ(req->accepted_tokens[0], 10);
  EXPECT_EQ(req->generated_tokens.size(), 1);
  EXPECT_EQ(req->generated_tokens[0], 20);
}

TEST_F(DraftTokenFilterTest, GrammarMatcherAcceptsAllTokens) {
  // 测试 grammar_matcher 接受所有 tokens 的情况
  std::vector<int> draft_mtp = {10};
  std::vector<int> draft_trie = {20, 30};
  std::vector<int> sampling_result = {10, 20, 30, 40};

  auto grammar_matcher = std::make_shared<MockGrammarMatcher>(std::vector<int>{20, 30, 40});
  auto req = CreateMockInferRequest(draft_mtp, draft_trie, sampling_result, {}, grammar_matcher);
  std::vector<std::shared_ptr<ExecutorInferRequest>> reqs = {req};

  // 调用 DefaultFilterDraftTokens 来处理 draft tokens
  size_t accepted_num = DefaultFilterDraftTokens(req);

  EXPECT_EQ(accepted_num, 3);  // 所有 3 个 draft tokens 都被接受
  EXPECT_EQ(req->accepted_tokens.size(), 3);
  EXPECT_EQ(req->accepted_tokens[0], 10);
  EXPECT_EQ(req->accepted_tokens[1], 20);
  EXPECT_EQ(req->accepted_tokens[2], 30);
  EXPECT_EQ(req->generated_tokens.size(), 1);
  EXPECT_EQ(req->generated_tokens[0], 40);
}

// Tests for DraftGeneratorController::FilterDraftTokens
class TrieWithControllerTest : public testing::Test {
 protected:
  void SetUp() override { controller_ = std::make_shared<DraftGeneratorController>(); }

  std::shared_ptr<ExecutorInferRequest> CreateMockInferRequest(
      const std::vector<int>& draft_tokens_mtp, const std::vector<int>& draft_tokens_trie,
      const std::vector<int>& sampling_result_tokens, const std::vector<int>& stop_token_ids = {},
      std::shared_ptr<GrammarMatcherWrapper> grammar_matcher = nullptr) {
    auto executor_req = std::make_shared<ExecutorInferRequest>();

    executor_req->req_id = 1;
    executor_req->input_tokens = {1, 2, 3};
    executor_req->output_tokens = {};
    executor_req->logprobs = {};
    executor_req->sampling_config.stop_token_ids = stop_token_ids;
    executor_req->finished = false;
    executor_req->aborted = false;
    executor_req->grammar_matcher = grammar_matcher;

    executor_req->draft_tokens.mtp = draft_tokens_mtp;
    executor_req->draft_tokens.spec = draft_tokens_trie;
    executor_req->sampling_result_tokens = sampling_result_tokens;

    return executor_req;
  }

  std::shared_ptr<DraftGeneratorController> controller_;
};

TEST_F(TrieWithControllerTest, FilterDraftTokensWithoutGenerator) {
  // 测试没有 generator 时使用默认过滤逻辑
  std::vector<int> draft_mtp = {10};
  std::vector<int> draft_trie = {20, 30};
  std::vector<int> sampling_result = {10, 20, 30, 40};

  auto req = CreateMockInferRequest(draft_mtp, draft_trie, sampling_result);
  std::vector<std::shared_ptr<ExecutorInferRequest>> reqs = {req};

  // 调用 FilterDraftTokens
  Status status = controller_->FilterDraftTokens(reqs);

  EXPECT_TRUE(status.OK());
  EXPECT_EQ(req->accepted_tokens.size(), 3);
  EXPECT_EQ(req->accepted_tokens[0], 10);
  EXPECT_EQ(req->accepted_tokens[1], 20);
  EXPECT_EQ(req->accepted_tokens[2], 30);
  EXPECT_EQ(req->generated_tokens.size(), 1);
  EXPECT_EQ(req->generated_tokens[0], 40);
}

TEST_F(TrieWithControllerTest, FilterDraftTokensWithGrammarMatcher) {
  // 测试带 grammar_matcher 的过滤
  std::vector<int> draft_mtp = {10};
  std::vector<int> draft_trie = {20, 30};
  std::vector<int> sampling_result = {10, 20, 30, 40};

  auto grammar_matcher = std::make_shared<MockGrammarMatcher>(std::vector<int>{20});
  auto req = CreateMockInferRequest(draft_mtp, draft_trie, sampling_result, {}, grammar_matcher);
  std::vector<std::shared_ptr<ExecutorInferRequest>> reqs = {req};

  // 调用 FilterDraftTokens
  Status status = controller_->FilterDraftTokens(reqs);

  EXPECT_TRUE(status.OK());
  EXPECT_EQ(req->accepted_tokens.size(), 1);
  EXPECT_EQ(req->accepted_tokens[0], 10);
  EXPECT_EQ(req->generated_tokens.size(), 1);
  EXPECT_EQ(req->generated_tokens[0], 20);
}

TEST_F(TrieWithControllerTest, FilterDraftTokensMultipleRequests) {
  // 测试多个请求的批量过滤
  std::vector<int> draft_mtp1 = {10};
  std::vector<int> draft_trie1 = {20, 30};
  std::vector<int> sampling_result1 = {10, 20, 30, 40};
  auto req1 = CreateMockInferRequest(draft_mtp1, draft_trie1, sampling_result1);

  std::vector<int> draft_mtp2 = {50};
  std::vector<int> draft_trie2 = {60};
  std::vector<int> sampling_result2 = {50, 60, 70};
  auto req2 = CreateMockInferRequest(draft_mtp2, draft_trie2, sampling_result2);

  std::vector<std::shared_ptr<ExecutorInferRequest>> reqs = {req1, req2};

  // 调用 FilterDraftTokens
  Status status = controller_->FilterDraftTokens(reqs);

  EXPECT_TRUE(status.OK());

  // 验证第一个请求
  EXPECT_EQ(req1->accepted_tokens.size(), 3);
  EXPECT_EQ(req1->generated_tokens[0], 40);

  // 验证第二个请求
  EXPECT_EQ(req2->accepted_tokens.size(), 2);
  EXPECT_EQ(req2->generated_tokens[0], 70);
}

TEST_F(TrieWithControllerTest, FilterDraftTokensWithStopToken) {
  // 测试遇到 stop token 时的过滤
  std::vector<int> draft_mtp = {10};
  std::vector<int> draft_trie = {20, 30};
  std::vector<int> sampling_result = {10, 20, 30, 40};
  std::vector<int> stop_tokens = {20};  // 20 是 stop token

  auto req = CreateMockInferRequest(draft_mtp, draft_trie, sampling_result, stop_tokens);
  std::vector<std::shared_ptr<ExecutorInferRequest>> reqs = {req};

  // 调用 FilterDraftTokens
  Status status = controller_->FilterDraftTokens(reqs);

  EXPECT_TRUE(status.OK());
  EXPECT_EQ(req->accepted_tokens.size(), 1);  // 只接受到 10，遇到 stop token 20 就停止
  EXPECT_EQ(req->accepted_tokens[0], 10);
  EXPECT_EQ(req->generated_tokens.size(), 1);
  EXPECT_EQ(req->generated_tokens[0], 20);
}

// Test for DefaultFilterDraftTokens special branch: size mismatch
TEST_F(DraftTokenFilterTest, DefaultFilterDraftTokensSizeMismatch) {
  // 测试 sampling_result_tokens.size() - kStepGenerateTokenNum != draft_tokens.size() 的情况
  std::vector<int> draft_mtp = {10};
  std::vector<int> draft_trie = {20, 30};
  // sampling_result_tokens 的大小不匹配：应该是 draft_tokens.size() + kStepGenerateTokenNum
  // draft_tokens.size() = 3, 所以 sampling_result_tokens 应该是 4
  // 但我们故意设置为 5 来触发特殊分支
  std::vector<int> sampling_result = {10, 20, 30, 40, 50};

  auto req = CreateMockInferRequest(draft_mtp, draft_trie, sampling_result);

  // 调用 DefaultFilterDraftTokens
  size_t accepted_num = DefaultFilterDraftTokens(req);

  // 验证返回值为 0（表示出错）
  EXPECT_EQ(accepted_num, 0);
  // 验证 accepted_tokens 被清空
  EXPECT_TRUE(req->accepted_tokens.empty());
}

// Test for DraftGeneratorInterface::FilterDraftTokens (default implementation)
class DraftGeneratorInterfaceTest : public testing::Test {
 protected:
  // 创建一个简单的 DraftGeneratorInterface 实现用于测试
  class TestDraftGenerator : public DraftGeneratorInterface {
   public:
    void GenerateDraft(std::shared_ptr<ExecutorInferRequest> req) override {
      // 简单的实现：生成一些 draft tokens
      req->draft_tokens.mtp = {100, 200};
    }
    // 使用默认的 FilterDraftTokens 实现
  };

  void SetUp() override { generator_ = std::make_shared<TestDraftGenerator>(); }

  std::shared_ptr<ExecutorInferRequest> CreateMockInferRequest(const std::vector<int>& draft_tokens_mtp,
                                                               const std::vector<int>& sampling_result_tokens) {
    auto executor_req = std::make_shared<ExecutorInferRequest>();

    executor_req->req_id = 1;
    executor_req->input_tokens = {1, 2, 3};
    executor_req->output_tokens = {};
    executor_req->finished = false;
    executor_req->aborted = false;

    executor_req->draft_tokens.mtp = draft_tokens_mtp;
    executor_req->sampling_result_tokens = sampling_result_tokens;

    return executor_req;
  }

  std::shared_ptr<TestDraftGenerator> generator_;
};

TEST_F(DraftGeneratorInterfaceTest, FilterDraftTokensDefaultImplementation) {
  // 测试 DraftGeneratorInterface::FilterDraftTokens 的默认实现
  std::vector<int> draft_mtp = {100, 200};
  std::vector<int> sampling_result = {100, 200, 300};

  auto req = CreateMockInferRequest(draft_mtp, sampling_result);

  // 调用 FilterDraftTokens（使用默认实现）
  size_t accepted_num = generator_->FilterDraftTokens(req);

  // 验证返回值为 2（两个 draft tokens 都被接受）
  EXPECT_EQ(accepted_num, 2);
  // 验证 accepted_tokens
  EXPECT_EQ(req->accepted_tokens.size(), 2);
  EXPECT_EQ(req->accepted_tokens[0], 100);
  EXPECT_EQ(req->accepted_tokens[1], 200);
  // 验证 generated_tokens
  EXPECT_EQ(req->generated_tokens.size(), 1);
  EXPECT_EQ(req->generated_tokens[0], 300);
}

TEST_F(DraftGeneratorInterfaceTest, FilterDraftTokensPartialMatch) {
  // 测试部分匹配的情况
  std::vector<int> draft_mtp = {100, 200, 300};
  std::vector<int> sampling_result = {100, 999, 300, 400};

  auto req = CreateMockInferRequest(draft_mtp, sampling_result);

  // 调用 FilterDraftTokens
  size_t accepted_num = generator_->FilterDraftTokens(req);

  // 验证返回值为 1（只有第一个 draft token 被接受）
  EXPECT_EQ(accepted_num, 1);
  // 验证 accepted_tokens
  EXPECT_EQ(req->accepted_tokens.size(), 1);
  EXPECT_EQ(req->accepted_tokens[0], 100);
  // 验证 generated_tokens
  EXPECT_EQ(req->generated_tokens.size(), 1);
  EXPECT_EQ(req->generated_tokens[0], 999);
}

}  // namespace ksana_llm
