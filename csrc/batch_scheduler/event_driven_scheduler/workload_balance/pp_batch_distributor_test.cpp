/* Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * Unit tests for PpBatchDistributor.
 * ==============================================================================*/

#include "batch_scheduler/event_driven_scheduler/workload_balance/pp_batch_distributor.h"

#include <gtest/gtest.h>

#include <memory>
#include <unordered_map>
#include <vector>

#include "batch_scheduler/event_driven_scheduler/event_driven_state.h"
#include "configure/schedule_config_parser.h"
#include "runtime/infer_request.h"
#include "runtime/infer_stage.h"
#include "tests/test.h"
#include "utils/request.h"

namespace ksana_llm {
namespace {

class PpBatchDistributorTest : public ::testing::Test {
 protected:
  std::shared_ptr<InferRequest> MakeReq(int64_t req_id, int attn_dp_group_id, size_t planning_query_len = 1) {
    auto input = std::make_shared<KsanaPythonInput>();
    input->input_tokens.assign(planning_query_len, 1);
    auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
    auto request = std::make_shared<Request>(input, req_ctx);
    auto infer_req = std::make_shared<InferRequest>(request, 0);
    infer_req->req_id = req_id;
    infer_req->attn_dp_group_id = attn_dp_group_id;
    infer_req->output_tokens.resize(planning_query_len);
    infer_req->kv_cached_token_num = 0;
    return infer_req;
  }

  void SetUp() override {
    schedule_states_.resize(max_pp_batch_num_);
    for (size_t bid = 0; bid < max_pp_batch_num_; ++bid) {
      schedule_states_[bid].resize(dp_num_);
    }
  }

  static PPMultibatchWBStrategy MakeStrategy(bool enable) {
    PPMultibatchWBStrategy s;
    s.enable = enable;
    return s;
  }

  size_t max_pp_batch_num_ = 2;
  size_t dp_num_ = 1;
  std::vector<std::vector<DpScheduleState>> schedule_states_;
  std::unordered_map<int64_t, size_t> req_to_batch_idx_;
};

TEST_F(PpBatchDistributorTest, EmptyRequests) {
  PpBatchDistributor distributor(MakeStrategy(false));
  std::vector<std::shared_ptr<InferRequest>> requests;
  distributor.DistributeRequests(requests, max_pp_batch_num_, dp_num_, schedule_states_, req_to_batch_idx_);
  EXPECT_TRUE(req_to_batch_idx_.empty());
}

TEST_F(PpBatchDistributorTest, SingleBatchAssignment) {
  max_pp_batch_num_ = 1;
  schedule_states_.resize(max_pp_batch_num_);
  schedule_states_[0].resize(dp_num_);

  PpBatchDistributor distributor(MakeStrategy(false));
  std::vector<std::shared_ptr<InferRequest>> requests;
  requests.push_back(MakeReq(1001, 0));
  requests.push_back(MakeReq(1002, 0));

  distributor.DistributeRequests(requests, max_pp_batch_num_, dp_num_, schedule_states_, req_to_batch_idx_);

  // All requests go to batch 0.
  EXPECT_EQ(req_to_batch_idx_.at(1001), 0u);
  EXPECT_EQ(req_to_batch_idx_.at(1002), 0u);
  EXPECT_EQ(schedule_states_[0][0].prefilling_reqs.size(), 2u);
}

TEST_F(PpBatchDistributorTest, GreedyMinWorkloadDistribution) {
  PpBatchDistributor distributor(MakeStrategy(false));
  std::vector<std::shared_ptr<InferRequest>> requests;
  requests.push_back(MakeReq(1001, 0));
  requests.push_back(MakeReq(1002, 0));
  requests.push_back(MakeReq(1003, 0));
  requests.push_back(MakeReq(1004, 0));

  distributor.DistributeRequests(requests, max_pp_batch_num_, dp_num_, schedule_states_, req_to_batch_idx_);

  // With 4 requests and 2 batches, greedy should give 2 to each batch.
  EXPECT_EQ(schedule_states_[0][0].prefilling_reqs.size(), 2u);
  EXPECT_EQ(schedule_states_[1][0].prefilling_reqs.size(), 2u);

  // All requests should be in req_to_batch_idx_.
  for (const auto& req : requests) {
    EXPECT_TRUE(req_to_batch_idx_.count(req->req_id) > 0);
  }
}

TEST_F(PpBatchDistributorTest, RespectsExistingWorkload) {
  // Pre-populate batch 0 with decoding requests to make it heavier.
  schedule_states_[0][0].decoding_reqs.push_back(MakeReq(2001, 0));

  PpBatchDistributor distributor(MakeStrategy(false));
  std::vector<std::shared_ptr<InferRequest>> requests;
  requests.push_back(MakeReq(1001, 0));

  distributor.DistributeRequests(requests, max_pp_batch_num_, dp_num_, schedule_states_, req_to_batch_idx_);

  // New request should go to batch 1 (lighter because batch 0 has one decoding req).
  EXPECT_EQ(req_to_batch_idx_.at(1001), 1u);
  EXPECT_EQ(schedule_states_[1][0].prefilling_reqs.size(), 1u);
  EXPECT_TRUE(schedule_states_[0][0].prefilling_reqs.empty());
}

TEST_F(PpBatchDistributorTest, MultiDpDistribution) {
  dp_num_ = 2;
  schedule_states_.resize(max_pp_batch_num_);
  for (size_t bid = 0; bid < max_pp_batch_num_; ++bid) {
    schedule_states_[bid].resize(dp_num_);
  }

  PpBatchDistributor distributor(MakeStrategy(false));
  std::vector<std::shared_ptr<InferRequest>> requests;
  // DP 0 requests
  requests.push_back(MakeReq(1001, 0));
  requests.push_back(MakeReq(1002, 0));
  // DP 1 requests
  requests.push_back(MakeReq(2001, 1));
  requests.push_back(MakeReq(2002, 1));

  distributor.DistributeRequests(requests, max_pp_batch_num_, dp_num_, schedule_states_, req_to_batch_idx_);

  // Each DP should have requests distributed across both batches.
  EXPECT_EQ(schedule_states_[0][0].prefilling_reqs.size() + schedule_states_[1][0].prefilling_reqs.size(), 2u);
  EXPECT_EQ(schedule_states_[0][1].prefilling_reqs.size() + schedule_states_[1][1].prefilling_reqs.size(), 2u);
}

TEST_F(PpBatchDistributorTest, NullRequestSkipped) {
  PpBatchDistributor distributor(MakeStrategy(false));
  std::vector<std::shared_ptr<InferRequest>> requests;
  requests.push_back(MakeReq(1001, 0));
  requests.push_back(nullptr);  // should be skipped
  requests.push_back(MakeReq(1002, 0));

  distributor.DistributeRequests(requests, max_pp_batch_num_, dp_num_, schedule_states_, req_to_batch_idx_);

  // 2 valid requests should be distributed.
  EXPECT_EQ(schedule_states_[0][0].prefilling_reqs.size() + schedule_states_[1][0].prefilling_reqs.size(), 2u);
  EXPECT_EQ(req_to_batch_idx_.size(), 2u);
}

TEST_F(PpBatchDistributorTest, WbReqTokenStrategy) {
  PpBatchDistributor distributor(MakeStrategy(true));
  std::vector<std::shared_ptr<InferRequest>> requests;
  requests.push_back(MakeReq(1001, 0, /*planning_query_len=*/10));
  requests.push_back(MakeReq(1002, 0, /*planning_query_len=*/1));

  distributor.DistributeRequests(requests, max_pp_batch_num_, dp_num_, schedule_states_, req_to_batch_idx_);

  // Both requests should end up in different batches due to workload difference.
  // batch 0 gets the heavier request (1001, output_tokens=10), batch 1 gets the lighter (1002, output_tokens=1).
  EXPECT_EQ(schedule_states_[0][0].prefilling_reqs.size(), 1u);
  EXPECT_EQ(schedule_states_[1][0].prefilling_reqs.size(), 1u);
}

TEST_F(PpBatchDistributorTest, BatchIsolationAfterDistribution) {
  // Verify that once assigned, requests do NOT appear in the wrong batch's queues.
  PpBatchDistributor distributor(MakeStrategy(false));
  std::vector<std::shared_ptr<InferRequest>> requests;
  requests.push_back(MakeReq(1001, 0));
  requests.push_back(MakeReq(1002, 0));
  requests.push_back(MakeReq(1003, 0));

  distributor.DistributeRequests(requests, max_pp_batch_num_, dp_num_, schedule_states_, req_to_batch_idx_);

  // Each batch's prefilling_reqs should only contain requests assigned to that batch.
  for (size_t bid = 0; bid < max_pp_batch_num_; ++bid) {
    for (const auto& req : schedule_states_[bid][0].prefilling_reqs) {
      ASSERT_TRUE(req_to_batch_idx_.count(req->req_id) > 0);
      EXPECT_EQ(req_to_batch_idx_.at(req->req_id), bid)
          << "req_id=" << req->req_id << " found in batch " << bid << " but assigned to batch "
          << req_to_batch_idx_.at(req->req_id);
    }
  }
}

TEST_F(PpBatchDistributorTest, ReqToBatchIdxConsistency) {
  // Every distributed request must have an entry in req_to_batch_idx,
  // and the entry must match the batch where the request was placed.
  PpBatchDistributor distributor(MakeStrategy(false));
  std::vector<std::shared_ptr<InferRequest>> requests;
  requests.push_back(MakeReq(1001, 0));
  requests.push_back(MakeReq(1002, 0));
  requests.push_back(MakeReq(1003, 0));
  requests.push_back(MakeReq(1004, 0));
  requests.push_back(MakeReq(1005, 0));

  distributor.DistributeRequests(requests, max_pp_batch_num_, dp_num_, schedule_states_, req_to_batch_idx_);

  // All requests must be in req_to_batch_idx.
  EXPECT_EQ(req_to_batch_idx_.size(), requests.size());

  // Each entry must point to a batch that actually contains the request.
  for (const auto& req : requests) {
    ASSERT_TRUE(req_to_batch_idx_.count(req->req_id) > 0);
    size_t assigned_batch = req_to_batch_idx_.at(req->req_id);
    bool found = false;
    for (const auto& r : schedule_states_[assigned_batch][0].prefilling_reqs) {
      if (r->req_id == req->req_id) {
        found = true;
        break;
      }
    }
    EXPECT_TRUE(found) << "req_id=" << req->req_id << " assigned to batch " << assigned_batch
                       << " but not found in that batch's prefilling_reqs";
  }
}

}  // namespace
}  // namespace ksana_llm