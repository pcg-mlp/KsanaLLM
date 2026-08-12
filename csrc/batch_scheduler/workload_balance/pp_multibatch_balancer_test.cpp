/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "batch_scheduler/workload_balance/pp_multibatch_balancer.h"

#include <gtest/gtest.h>
#include <memory>
#include <unordered_map>
#include <vector>

#include "batch_scheduler/state/batch_state.h"
#include "runtime/infer_request.h"
#include "runtime/schedule_output.h"
#include "utils/logger.h"
#include "utils/request.h"

namespace ksana_llm {
namespace {

class PPMultibatchWorkloadBalancerTest : public testing::Test {
 protected:
  void SetUp() override {
    InitLoguru();
    schedule_output_pool_ = std::make_shared<ScheduleOutputPool>();
  }

  void TearDown() override {
    request_holders_.clear();
    input_holders_.clear();
    schedule_output_pool_.reset();
  }

  BatchSchedulerConfig MakeConfig(size_t max_pp_batch_num = 2, size_t max_step_token_num = 100,
                                  size_t split_fuse_token_num = 0, int prefill_hold_iters = 30,
                                  float threshold_factor = 0.8f) {
    BatchSchedulerConfig cfg;
    cfg.max_batch_size = 128;
    cfg.max_waiting_queue_len = 128;
    cfg.max_pp_batch_num = max_pp_batch_num;
    cfg.max_step_token_num = max_step_token_num;
    cfg.split_fuse_token_num = split_fuse_token_num;
    cfg.pp_multibatch_wb_strategy.enable = true;
    cfg.pp_multibatch_wb_strategy.prefill_hold_iters = prefill_hold_iters;
    cfg.pp_multibatch_wb_strategy.threshold_factor = threshold_factor;
    return cfg;
  }

  std::vector<std::shared_ptr<BatchState>> MakeBatchStates(const BatchSchedulerConfig& cfg) {
    std::vector<std::shared_ptr<BatchState>> states;
    for (size_t i = 0; i < cfg.max_pp_batch_num; ++i) {
      states.push_back(std::make_shared<BatchState>(i, cfg, schedule_output_pool_));
    }
    return states;
  }

  // Build a request with a fixed remaining workload; keep the backing Request /
  // KsanaPythonInput alive for the whole test.
  std::shared_ptr<InferRequest> MakeReq(int64_t req_id, size_t prefill_token_num, size_t prefill_start_offset = 0,
                                        size_t generated_token_num = 0) {
    auto python_input = std::make_shared<KsanaPythonInput>();
    // input_tokens size only affects logs; give it something reasonable.
    python_input->input_tokens.assign(std::max<size_t>(1, prefill_token_num + prefill_start_offset), 1);
    auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
    auto request = std::make_shared<Request>(python_input, req_ctx);
    input_holders_.push_back(python_input);
    request_holders_.push_back(request);

    auto req = std::make_shared<InferRequest>(request, 0);
    req->req_id = req_id;

    ScheduleTaskWorkload wl;
    wl.prefill_token_num = prefill_token_num;
    wl.prefill_start_offset = prefill_start_offset;
    wl.generated_token_num = generated_token_num;
    wl.sampling_token_num = 1;
    req->SetRemainingWorkload(wl);
    return req;
  }

  // Chunked prefill req: has an already-cached prefix (offset > 0, kv_cached > 0).
  std::shared_ptr<InferRequest> MakeChunkedReq(int64_t req_id, size_t remaining_prefill, size_t prefill_start_offset) {
    auto req = MakeReq(req_id, remaining_prefill, prefill_start_offset);
    req->SetKvCachedTokenNum(prefill_start_offset);
    return req;
  }

  std::shared_ptr<InferRequest> MakeDecodeReq(int64_t req_id) { return MakeReq(req_id, /*prefill_token_num=*/0); }

  static std::vector<int64_t> Ids(const std::vector<std::shared_ptr<InferRequest>>& reqs) {
    std::vector<int64_t> ids;
    ids.reserve(reqs.size());
    for (const auto& r : reqs) ids.push_back(r->req_id);
    return ids;
  }

  std::shared_ptr<ScheduleOutputPool> schedule_output_pool_;
  std::vector<std::shared_ptr<KsanaPythonInput>> input_holders_;
  std::vector<std::shared_ptr<Request>> request_holders_;
};

// ---------- Constructor validation ---------------------------------------------------------------

TEST_F(PPMultibatchWorkloadBalancerTest, ConstructorRejectsStrategyDisabled) {
  auto cfg = MakeConfig();
  cfg.pp_multibatch_wb_strategy.enable = false;
  EXPECT_ANY_THROW({ PPMultibatchWorkloadBalancer balancer(cfg); });
}

TEST_F(PPMultibatchWorkloadBalancerTest, ConstructorRejectsSinglePPBatch) {
  auto cfg = MakeConfig(/*max_pp_batch_num=*/1);
  EXPECT_ANY_THROW({ PPMultibatchWorkloadBalancer balancer(cfg); });
}

TEST_F(PPMultibatchWorkloadBalancerTest, ConstructorRejectsThresholdAboveOne) {
  auto cfg = MakeConfig();
  cfg.pp_multibatch_wb_strategy.threshold_factor = 1.5f;
  EXPECT_ANY_THROW({ PPMultibatchWorkloadBalancer balancer(cfg); });
}

// ---------- HasHeldWaitingReqs -------------------------------------------------------------------

TEST_F(PPMultibatchWorkloadBalancerTest, HasHeldWaitingReqsIsFalseInitially) {
  auto cfg = MakeConfig();
  PPMultibatchWorkloadBalancer balancer(cfg);
  EXPECT_FALSE(balancer.HasHeldWaitingReqs());
}

// ---------- DistributeWaitingReqs: hold / force-commit -----------------------------------------

// prefill_hold_iters=1 → first (only) iter is a force-commit path: waiting_reqs are distributed
// greedily to the two candidate states, and HasHeldWaitingReqs stays false.
TEST_F(PPMultibatchWorkloadBalancerTest, WaitingForceCommitGreedyDistribution) {
  auto cfg = MakeConfig(/*max_pp_batch_num=*/2, /*max_step_token_num=*/100, /*split_fuse_token_num=*/0,
                        /*prefill_hold_iters=*/1);
  auto states = MakeBatchStates(cfg);

  PPMultibatchWorkloadBalancer balancer(cfg);

  // Two waiting reqs, each prefill_token_num=10. Both cur and next start empty.
  std::vector<std::shared_ptr<InferRequest>> waiting;
  waiting.push_back(MakeReq(100, /*prefill_token_num=*/10));
  waiting.push_back(MakeReq(101, /*prefill_token_num=*/10));

  balancer.BalancePPMultiBatchReqs(/*multi_batch_id=*/0, waiting, states);

  // Both req consumed; force-commit path leaves nothing held.
  EXPECT_TRUE(waiting.empty());
  EXPECT_FALSE(balancer.HasHeldWaitingReqs());
  // Greedy picks min-workload; first req lands on cur (0), second on next (1) after cur load increases.
  EXPECT_EQ(states[0]->waiting_queue.size() + states[1]->waiting_queue.size(), 2u);
  EXPECT_EQ(states[0]->waiting_queue.size(), 1u);
  EXPECT_EQ(states[1]->waiting_queue.size(), 1u);
}

// prefill_hold_iters=10, first call: not both saturated → the balancer holds all waiting reqs.
// waiting_reqs is emptied by the balancer (consumed via std::move), HasHeldWaitingReqs is true.
TEST_F(PPMultibatchWorkloadBalancerTest, WaitingHeldWhenNotBothSaturated) {
  auto cfg = MakeConfig(/*max_pp_batch_num=*/2, /*max_step_token_num=*/100, /*split_fuse_token_num=*/0,
                        /*prefill_hold_iters=*/10);
  auto states = MakeBatchStates(cfg);

  PPMultibatchWorkloadBalancer balancer(cfg);

  std::vector<std::shared_ptr<InferRequest>> waiting;
  waiting.push_back(MakeReq(100, /*prefill_token_num=*/10));

  balancer.BalancePPMultiBatchReqs(/*multi_batch_id=*/0, waiting, states);

  EXPECT_TRUE(waiting.empty());
  EXPECT_TRUE(balancer.HasHeldWaitingReqs());
  EXPECT_EQ(states[0]->waiting_queue.size(), 0u);
  EXPECT_EQ(states[1]->waiting_queue.size(), 0u);
}

// Held reqs are re-injected on the next call so the balancer can retry them. With
// prefill_hold_iters=2, the first call holds; the second call fires (hold_iter_ reaches 2).
TEST_F(PPMultibatchWorkloadBalancerTest, HeldReqsAreReinjectedOnNextCall) {
  auto cfg = MakeConfig(/*max_pp_batch_num=*/2, /*max_step_token_num=*/100, /*split_fuse_token_num=*/0,
                        /*prefill_hold_iters=*/2);
  auto states = MakeBatchStates(cfg);
  PPMultibatchWorkloadBalancer balancer(cfg);

  std::vector<std::shared_ptr<InferRequest>> waiting;
  waiting.push_back(MakeReq(100, /*prefill_token_num=*/10));
  balancer.BalancePPMultiBatchReqs(0, waiting, states);
  ASSERT_TRUE(balancer.HasHeldWaitingReqs());
  ASSERT_TRUE(waiting.empty());
  ASSERT_TRUE(states[0]->waiting_queue.empty());

  // Second call with no fresh waiting reqs: held req re-injected, then force-committed.
  std::vector<std::shared_ptr<InferRequest>> empty_waiting;
  balancer.BalancePPMultiBatchReqs(0, empty_waiting, states);

  EXPECT_FALSE(balancer.HasHeldWaitingReqs());
  EXPECT_EQ(states[0]->waiting_queue.size() + states[1]->waiting_queue.size(), 1u);
}

// If cur's workload already exceeds step_token_num_bound_ at entry, every new waiting req is held
// unconditionally via the saturated early-return path, before any greedy assignment or
// anti-starvation hold-iter logic is reached. Use a large prefill_hold_iters to make it clear this
// test does NOT depend on the anti-starvation mechanism.
TEST_F(PPMultibatchWorkloadBalancerTest, WaitingHeldWhenCurAlreadySaturated) {
  auto cfg = MakeConfig(/*max_pp_batch_num=*/2, /*max_step_token_num=*/100, /*split_fuse_token_num=*/0,
                        /*prefill_hold_iters=*/30, /*threshold_factor=*/0.5f);
  auto states = MakeBatchStates(cfg);

  // step_token_num_bound_ = 100 * 0.5 = 50. Fill cur decoding with 60 reqs (each load 1) so
  // workloads[cur] = 60 > 50.
  for (int64_t i = 0; i < 60; ++i) states[0]->decoding_queue.push_back(MakeDecodeReq(i));

  PPMultibatchWorkloadBalancer balancer(cfg);

  std::vector<std::shared_ptr<InferRequest>> waiting;
  waiting.push_back(MakeReq(1000, /*prefill_token_num=*/10));
  balancer.BalancePPMultiBatchReqs(0, waiting, states);

  EXPECT_TRUE(waiting.empty());
  EXPECT_TRUE(balancer.HasHeldWaitingReqs());
  // No assignment happened.
  EXPECT_TRUE(states[0]->waiting_queue.empty());
  EXPECT_TRUE(states[1]->waiting_queue.empty());
}

// Non-chunked reqs sitting in next's waiting_queue are pulled back into the global pool so they
// can be re-balanced. Chunked ones (prefill_start_offset > 0 && kv_cached_token_num > 0) stay put.
TEST_F(PPMultibatchWorkloadBalancerTest, NonChunkedReqsInNextArePulledBackForRebalancing) {
  auto cfg = MakeConfig(/*max_pp_batch_num=*/2, /*max_step_token_num=*/100, /*split_fuse_token_num=*/0,
                        /*prefill_hold_iters=*/1);
  auto states = MakeBatchStates(cfg);

  // Pre-seed batch 1's waiting queue with a mix: one chunked (stays), one non-chunked (pulled back).
  auto chunked = MakeChunkedReq(500, /*remaining_prefill=*/10, /*prefill_start_offset=*/5);
  auto non_chunked = MakeReq(501, /*prefill_token_num=*/10);
  states[1]->waiting_queue.push_back(chunked);
  states[1]->waiting_queue.push_back(non_chunked);

  PPMultibatchWorkloadBalancer balancer(cfg);

  std::vector<std::shared_ptr<InferRequest>> waiting;
  waiting.push_back(MakeReq(600, /*prefill_token_num=*/10));
  balancer.BalancePPMultiBatchReqs(0, waiting, states);

  // chunked req still on batch 1.
  bool chunked_on_batch1 = false;
  for (const auto& r : states[1]->waiting_queue) {
    if (r->req_id == 500) {
      chunked_on_batch1 = true;
      break;
    }
  }
  EXPECT_TRUE(chunked_on_batch1) << "chunked req 500 must stay on batch 1";

  // The pulled-back non_chunked (501) + the fresh incoming (600) got greedily assigned across
  // cur/next.
  const size_t assigned = states[0]->waiting_queue.size() + (states[1]->waiting_queue.size() - 1 /* chunked */);
  EXPECT_EQ(assigned, 2u);
  EXPECT_FALSE(balancer.HasHeldWaitingReqs());
}

// split_fuse cap applies to per-req step load: a long prefill req is capped at split_fuse_token_num.
TEST_F(PPMultibatchWorkloadBalancerTest, SplitFuseCapAppliedToStepLoad) {
  // max_step_token_num large, split_fuse cap small. bound = step_token_num * 0.5 = 200.
  auto cfg = MakeConfig(/*max_pp_batch_num=*/2, /*max_step_token_num=*/400, /*split_fuse_token_num=*/50,
                        /*prefill_hold_iters=*/1, /*threshold_factor=*/0.5f);
  auto states = MakeBatchStates(cfg);

  PPMultibatchWorkloadBalancer balancer(cfg);

  // 5 waiting reqs, each with 1000 raw prefill tokens. Capped at 50 per step, so each contributes 50.
  // total added workload if all placed on cur = 250 (> bound=200), but placement alternates greedily
  // so both cur and next receive some.
  std::vector<std::shared_ptr<InferRequest>> waiting;
  for (int64_t i = 0; i < 5; ++i) {
    waiting.push_back(MakeReq(700 + i, /*prefill_token_num=*/1000));
  }
  balancer.BalancePPMultiBatchReqs(0, waiting, states);

  EXPECT_TRUE(waiting.empty());
  EXPECT_FALSE(balancer.HasHeldWaitingReqs());
  const size_t total_assigned = states[0]->waiting_queue.size() + states[1]->waiting_queue.size();
  // Greedy stops when both candidates >= step_token_num (400). Each add contributes ~50 to
  // some batch; all 5 should be consumed comfortably under 400.
  EXPECT_EQ(total_assigned, 5u);
}

// ---------- ShiftChunkedReqs ---------------------------------------------------------------------

// Only reqs with prefill_start_offset>0 && kv_cached_token_num>0 are shifted. Order among the
// shifted reqs is preserved. Non-chunked reqs stay.
TEST_F(PPMultibatchWorkloadBalancerTest, ShiftChunkedOnlyMovesChunkedReqs) {
  auto cfg = MakeConfig();
  auto states = MakeBatchStates(cfg);
  PPMultibatchWorkloadBalancer balancer(cfg);

  auto chunked_a = MakeChunkedReq(1, /*remaining_prefill=*/8, /*prefill_start_offset=*/4);
  auto plain = MakeReq(2, /*prefill_token_num=*/8);  // offset=0 → not chunked
  auto chunked_b = MakeChunkedReq(3, /*remaining_prefill=*/6, /*prefill_start_offset=*/2);
  states[0]->waiting_queue.push_back(chunked_a);
  states[0]->waiting_queue.push_back(plain);
  states[0]->waiting_queue.push_back(chunked_b);

  balancer.ShiftChunkedReqs(/*multi_batch_id=*/0, states);

  // Chunked reqs moved to front of next, preserving relative order.
  ASSERT_EQ(states[1]->waiting_queue.size(), 2u);
  EXPECT_EQ(states[1]->waiting_queue[0]->req_id, 1);
  EXPECT_EQ(states[1]->waiting_queue[1]->req_id, 3);
  // Plain req stays on cur.
  ASSERT_EQ(states[0]->waiting_queue.size(), 1u);
  EXPECT_EQ(states[0]->waiting_queue[0]->req_id, 2);
}

// Shifted reqs are prepended to the destination queue, keeping the existing dst reqs after them.
TEST_F(PPMultibatchWorkloadBalancerTest, ShiftChunkedInsertsAtFrontOfDest) {
  auto cfg = MakeConfig();
  auto states = MakeBatchStates(cfg);
  PPMultibatchWorkloadBalancer balancer(cfg);

  states[0]->waiting_queue.push_back(MakeChunkedReq(10, 8, 4));
  states[1]->waiting_queue.push_back(MakeReq(20, /*prefill_token_num=*/8));

  balancer.ShiftChunkedReqs(0, states);

  ASSERT_EQ(states[1]->waiting_queue.size(), 2u);
  EXPECT_EQ(states[1]->waiting_queue[0]->req_id, 10);
  EXPECT_EQ(states[1]->waiting_queue[1]->req_id, 20);
  EXPECT_TRUE(states[0]->waiting_queue.empty());
}

// No eligible chunked reqs → no-op.
TEST_F(PPMultibatchWorkloadBalancerTest, ShiftChunkedIsNoOpWhenNoneEligible) {
  auto cfg = MakeConfig();
  auto states = MakeBatchStates(cfg);
  PPMultibatchWorkloadBalancer balancer(cfg);

  states[0]->waiting_queue.push_back(MakeReq(1, /*prefill_token_num=*/8));  // offset=0
  balancer.ShiftChunkedReqs(0, states);

  EXPECT_EQ(states[0]->waiting_queue.size(), 1u);
  EXPECT_TRUE(states[1]->waiting_queue.empty());
}

}  // namespace
}  // namespace ksana_llm
