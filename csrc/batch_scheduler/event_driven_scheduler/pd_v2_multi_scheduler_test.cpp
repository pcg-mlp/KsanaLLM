/* Copyright 2026 Tencent Inc.  All rights reserved.

 * Phase A multi-scheduler tests for pd_v2 (T01~T09), built on top of the
 * in-process FakePdV2Connector topology in pd_v2_multi_scheduler_test_helper.
 * See spec/scheduler/TAPD-1070131059135905223-pd-multi-scheduler-test/
 * 1-change-plan.md §E for the T01~T09 -> TEST_F mapping this file implements.
==============================================================================*/
#include "batch_scheduler/event_driven_scheduler/pd_v2_multi_scheduler_test_helper.h"

#include <chrono>
#include <thread>

#include "batch_scheduler/batch_scheduler_test.h"
#include "test.h"

using namespace ksana_llm;         // NOLINT(build/namespaces)
using namespace ksana_llm::pd_v2;  // NOLINT(build/namespaces)

namespace {

// Polls req_waiter's remaining count instead of calling Waiter::Wait()
// directly: Wait() blocks indefinitely with no timeout, so a genuine bug
// (e.g. a lost async dispatch) would hang the whole test binary rather than
// failing the one affected test. Every TEST_F below uses this instead.
bool WaitForCompletion(const std::shared_ptr<Waiter>& waiter, int timeout_sec) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_sec);
  while (std::chrono::steady_clock::now() < deadline) {
    if (waiter->Cnt() <= 0) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  return waiter->Cnt() <= 0;
}

// Same construction PdV2SimNode::SubmitRequest performs, but accepts a full
// multi-segment seeds list (see BatchSchedulerEnvironmentSimulator::InitRequest
// / GetFakeSeed) instead of a single scalar seed. Needed by the prefix-cache
// tests (T05/T09) to build two requests that share an identical leading token
// span (same seed segment) while differing afterward (distinct trailing
// segment) -- mirrors the existing `seeds[0]=<shared>, seeds[1]=<per-req>`
// pattern in batch_scheduler_test_client.h's SplitFuse prefix-cache test.
//
// IMPORTANT (see PdV2SimNode::SubmitRequest's doc): whichever seed value
// governs the offset range [input_token_num, input_token_num + 1) (i.e. the
// segment active at `offset == input_token_num`) MUST equal this request's
// own req_id if the request is expected to actually cross to Prefill,
// because PdV2PrefillIntakeAdapter bootstraps its own env_simulator's seed
// for a given req_id from that req_id itself.
ParallelTester::RequestInfo SubmitRequestWithSeeds(PdV2SimNode& node, int64_t req_id, int input_token_num,
                                                   int expected_output_token_num,
                                                   const std::vector<std::pair<int, int>>& seeds) {
  ParallelTester::RequestInfo info;
  info.req_id = req_id;
  info.input_token_num = input_token_num;
  info.expect_output_token_num = expected_output_token_num;
  info.infer_req_group = node.env_simulator->InitRequest(req_id, input_token_num, expected_output_token_num, info.req,
                                                         info.ksana_python_input, seeds);
  Status s = node.eds->AddInferRequest(info.infer_req_group);
  EXPECT_TRUE(s.OK()) << "AddInferRequest failed for req_id=" << req_id << ": " << s.GetMessage();
  return info;
}

}  // namespace

// Each TEST_F builds its own MultiNodeTopology, which owns a fresh
// PdV2FakeConnectorRegistry (see pd_v2_multi_scheduler_test_helper.h) --
// unlike the mooncake::TransferEngine::Reset() precedent this class's
// Reset() references, there is no process-wide singleton address book to
// clear between cases here, so no explicit PdV2FakeConnectorRegistry::Reset()
// call is needed in SetUp/TearDown: cross-test inference_addr string reuse
// (e.g. every test's Decode node is named "decode-0") is safe by
// construction because each topology's registry is a fully independent
// instance, destroyed (draining its thread pool first) when the TEST_F's
// MultiNodeTopology goes out of scope.
class PdV2MultiSchedulerTest : public testing::Test {
 protected:
  static void SetUpTestSuite() { InitLoguru(); }
};

// T01: 1P1D basic coordination. A single request round-trips Decode ->
// Prefill -> Decode and its final output_tokens match the same
// BatchSchedulerEnvironmentSimulator fake-token rules single-node
// EventDrivenSchedulerTest verifies against.
TEST_F(PdV2MultiSchedulerTest, BasicOnePOneDCoordinationTest) {
  MultiNodeTopologyConfig config;
  MultiNodeTopology topo(config);
  auto& decode = topo.DecodeNode(0);

  const int64_t req_id = 101;
  auto info = decode.SubmitRequest(req_id, /*input_token_num=*/20, /*expected_output_token_num=*/10,
                                   /*seed=*/static_cast<int>(req_id));
  ASSERT_TRUE(WaitForCompletion(info.req->waiter, 15)) << "req_id=" << req_id << " never completed";
  EXPECT_TRUE(info.req->finish_status.OK()) << info.req->finish_status.GetMessage();
  EXPECT_GT(decode.connector->submit_calls.load(), 0);
  decode.env_simulator->CheckRequestOutput(info.infer_req_group[0]);
}

// T02: KV block address match. Several concurrently-running requests of
// different lengths share the same 1P1D topology; SimulatedKvWriteBridge
// writes each request's freshly-prefilled prompt span into Decode blocks.
// RunAStep's own EXPECT_EQ(forwarding_tokens, kv_contents) self-check (see
// batch_scheduler_test_helper.h) already fails loudly the moment a write
// lands on a block belonging to a different, concurrently in-flight
// request, so a clean pass across all requests here is itself the
// "never touches out-of-range blocks" assertion T02 calls for.
TEST_F(PdV2MultiSchedulerTest, KvBlockAddressMatchTest) {
  MultiNodeTopologyConfig config;
  MultiNodeTopology topo(config);
  auto& decode = topo.DecodeNode(0);

  constexpr int kRequestNum = 6;
  std::vector<ParallelTester::RequestInfo> infos;
  infos.reserve(kRequestNum);
  for (int i = 0; i < kRequestNum; ++i) {
    const int64_t req_id = 200 + i;
    infos.push_back(decode.SubmitRequest(req_id, /*input_token_num=*/12 + i * 3, /*expected_output_token_num=*/6,
                                         /*seed=*/static_cast<int>(req_id)));
  }
  for (auto& info : infos) {
    ASSERT_TRUE(WaitForCompletion(info.req->waiter, 20)) << "req_id=" << info.req_id << " never completed";
    EXPECT_TRUE(info.req->finish_status.OK())
        << "req_id=" << info.req_id << ": " << info.req->finish_status.GetMessage();
    decode.env_simulator->CheckRequestOutput(info.infer_req_group[0]);
  }
}

// T03: 2P2D concurrency, no deadlock. Every request must reach a terminal
// state (waiter fires) within the timeout window; none may stay parked in
// kWaitAsyncResult forever.
TEST_F(PdV2MultiSchedulerTest, MultiNodeConcurrentNoDeadlockTest) {
  MultiNodeTopologyConfig config;
  config.num_prefill = 2;
  config.num_decode = 2;
  MultiNodeTopology topo(config);

  constexpr int kRequestNum = 12;
  std::vector<ParallelTester::RequestInfo> infos;
  infos.reserve(kRequestNum);
  for (int i = 0; i < kRequestNum; ++i) {
    const int64_t req_id = 300 + i;
    auto& decode = topo.DecodeNode(static_cast<size_t>(i) % topo.NumDecodeNodes());
    infos.push_back(decode.SubmitRequest(req_id, /*input_token_num=*/16 + i, /*expected_output_token_num=*/5,
                                         /*seed=*/static_cast<int>(req_id)));
  }
  for (auto& info : infos) {
    EXPECT_TRUE(WaitForCompletion(info.req->waiter, 25))
        << "req_id=" << info.req_id << " stuck (possible deadlock / lost async dispatch)";
  }
}

// T04: concurrent cross-node, no cross-talk. Requests of distinct lengths
// spread across multiple Decode nodes must each end up with their own
// (and only their own) output_tokens sequence -- verified via
// CheckRequestOutput, which recomputes the expected sequence purely from
// that request's own req_id-derived seed bookkeeping, so a request that
// somehow observed another request's tokens/KV would fail this check.
TEST_F(PdV2MultiSchedulerTest, ConcurrentCrossNodeNoCrossTalkTest) {
  MultiNodeTopologyConfig config;
  config.num_prefill = 2;
  config.num_decode = 2;
  MultiNodeTopology topo(config);

  constexpr int kRequestNum = 16;
  std::vector<ParallelTester::RequestInfo> infos;
  std::vector<size_t> decode_idx_for_req;
  infos.reserve(kRequestNum);
  decode_idx_for_req.reserve(kRequestNum);
  for (int i = 0; i < kRequestNum; ++i) {
    const int64_t req_id = 400 + i;
    const size_t decode_idx = static_cast<size_t>(i) % topo.NumDecodeNodes();
    auto& decode = topo.DecodeNode(decode_idx);
    infos.push_back(decode.SubmitRequest(req_id, /*input_token_num=*/10 + i, /*expected_output_token_num=*/4 + (i % 3),
                                         /*seed=*/static_cast<int>(req_id)));
    decode_idx_for_req.push_back(decode_idx);
  }
  for (size_t i = 0; i < infos.size(); ++i) {
    auto& info = infos[i];
    ASSERT_TRUE(WaitForCompletion(info.req->waiter, 25)) << "req_id=" << info.req_id << " never completed";
    EXPECT_TRUE(info.req->finish_status.OK())
        << "req_id=" << info.req_id << ": " << info.req->finish_status.GetMessage();
    auto& decode = topo.DecodeNode(decode_idx_for_req[i]);
    decode.env_simulator->CheckRequestOutput(info.infer_req_group[0]);
  }
}

// T05: prefix-cache soft bypass skips Prefill. Local prefix query is hard-
// capped at seq-1 AND rounded down to whole blocks (prefix_cache_manager:
// max_match_blocks = (check-1)/block_token_num), so for a fully-shared prompt
// of length L that is a multiple of block_token_num the residual is exactly
// one block (remaining = block_token_num), never 1. The remaining<=0 path
// therefore never fires here; we set local_complete_threshold >=
// block_token_num so PdV2DecodeHook's soft-bypass (remaining <= threshold)
// keeps req2 entirely local -- SubmitPrefillRequest must NOT be called a
// second time.
TEST_F(PdV2MultiSchedulerTest, PrefixCacheHitSkipPrefillTest) {
  MultiNodeTopologyConfig config;
  config.enable_prefix_cache = true;
  config.block_token_num = 6;
  // Must cover one full residual block after the seq-1 / block-granularity
  // hard cap (remaining == block_token_num), not just a single leftover token.
  config.local_complete_threshold = static_cast<int>(config.block_token_num);
  MultiNodeTopology topo(config);
  auto& decode = topo.DecodeNode(0);

  const int64_t req1_id = 501;
  // Multiple of block_token_num so the residual after the hard cap is exactly
  // one full block -- the case that threshold=1 would miss.
  const int input_token_num = 24;
  auto info1 =
      decode.SubmitRequest(req1_id, input_token_num, /*expected_output_token_num=*/5, static_cast<int>(req1_id));
  ASSERT_TRUE(WaitForCompletion(info1.req->waiter, 15)) << "req1 (warm-up) never completed";
  EXPECT_TRUE(info1.req->finish_status.OK()) << info1.req->finish_status.GetMessage();
  decode.env_simulator->CheckRequestOutput(info1.infer_req_group[0]);
  const int submit_calls_after_warmup = decode.connector->submit_calls.load();
  EXPECT_GT(submit_calls_after_warmup, 0) << "warm-up request should have gone through Prefill at least once";

  // Fully-shared prompt: single seed segment == req1_id reproduces req1's
  // exact input_tokens bytes (see SubmitRequestWithSeeds doc) for the whole
  // [0, input_token_num) range, and the trailing segment (req2's own id)
  // governs only req2's own post-prompt output generation, which never
  // leaves Decode on this bypass path.
  std::vector<std::pair<int, int>> seeds2{{0, static_cast<int>(req1_id)}, {input_token_num, 502}};
  auto info2 = SubmitRequestWithSeeds(decode, /*req_id=*/502, input_token_num, /*expected_output_token_num=*/5, seeds2);
  ASSERT_TRUE(WaitForCompletion(info2.req->waiter, 15)) << "req2 (cache hit) never completed";
  EXPECT_TRUE(info2.req->finish_status.OK()) << info2.req->finish_status.GetMessage();
  decode.env_simulator->CheckRequestOutput(info2.infer_req_group[0]);

  EXPECT_EQ(decode.connector->submit_calls.load(), submit_calls_after_warmup)
      << "PdV2DecodeHook soft-bypass (local_complete_threshold=" << config.local_complete_threshold
      << " >= block_token_num=" << config.block_token_num << ") should have skipped peer Prefill for req2";
}

// T06: Prefill busy backoff. Starve the Prefill node's block pool relative
// to Decode's so several concurrent requests contend for very few blocks.
// The invariant under test is "no request is silently lost" -- resource
// pressure may legitimately fail some requests outright, so unlike T03/T04
// this does not require finish_status.OK(), only that every request
// reaches SOME terminal state (success or explicit failure) rather than
// hanging forever.
TEST_F(PdV2MultiSchedulerTest, PrefillBusyBackoffTest) {
  MultiNodeTopologyConfig config;
  config.block_token_num = 6;
  config.prefill_device_block_num = 6;  // Only ~1 request's worth of blocks on Prefill.
  MultiNodeTopology topo(config);
  auto& decode = topo.DecodeNode(0);

  constexpr int kRequestNum = 5;
  std::vector<ParallelTester::RequestInfo> infos;
  infos.reserve(kRequestNum);
  for (int i = 0; i < kRequestNum; ++i) {
    const int64_t req_id = 600 + i;
    infos.push_back(decode.SubmitRequest(req_id, /*input_token_num=*/10, /*expected_output_token_num=*/3,
                                         /*seed=*/static_cast<int>(req_id)));
  }
  for (auto& info : infos) {
    EXPECT_TRUE(WaitForCompletion(info.req->waiter, 25))
        << "req_id=" << info.req_id << " never reached a terminal state under Prefill resource pressure";
  }
}

// T07: completion notification vs. KV write-back race. SimulatedKvWriteBridge
// is given an artificial delay; since OnPrefillFinished calls WriteBack
// synchronously to completion BEFORE calling SendPrefillComplete (see
// pd_v2_multi_scheduler_test_helper.cpp), the end-to-end request latency
// observed here must be at least that delay -- if SendPrefillComplete
// somehow raced ahead of the write, completion could arrive faster than
// the configured delay.
TEST_F(PdV2MultiSchedulerTest, CompletionKvWriteRaceTest) {
  MultiNodeTopologyConfig config;
  MultiNodeTopology topo(config);
  auto& prefill = topo.PrefillNode(0);
  auto& decode = topo.DecodeNode(0);

  const auto kDelay = std::chrono::milliseconds(50);
  prefill.kv_bridge->SetArtificialDelay(std::chrono::duration_cast<std::chrono::microseconds>(kDelay));

  const int64_t req_id = 701;
  const auto t_start = std::chrono::steady_clock::now();
  auto info =
      decode.SubmitRequest(req_id, /*input_token_num=*/20, /*expected_output_token_num=*/5, static_cast<int>(req_id));
  ASSERT_TRUE(WaitForCompletion(info.req->waiter, 15)) << "req_id=" << req_id << " never completed";
  const auto elapsed = std::chrono::steady_clock::now() - t_start;

  EXPECT_GE(elapsed, kDelay) << "completion observed faster than the configured KV write-back delay -- "
                                "SendPrefillComplete may have raced ahead of SimulatedKvWriteBridge::WriteBack";
  EXPECT_GT(prefill.kv_bridge->LastWriteBackCompleteTimeUs(), 0);
  EXPECT_TRUE(info.req->finish_status.OK()) << info.req->finish_status.GetMessage();
  decode.env_simulator->CheckRequestOutput(info.infer_req_group[0]);
}

// T08: SWA cross-node coordination -- Phase A reduced scope. DecodeTarget
// carries no SWA block ids, so SimulatedKvWriteBridge cannot represent
// cross-node SWA KV bridging yet; MultiNodeTopology::BuildNode keeps
// cache_manager_config.sliding_window_size pinned to 0 regardless of this
// config knob (see its doc comment and 2-coding-snapshot.md "已知不确定项").
// This test therefore only verifies the 1P1D round trip still completes
// correctly with the SWA knob set (no crash / no hang, matching the
// non-SWA baseline's correctness), NOT that SWA window content survives
// the cross-node hop. Full coverage is deferred to a follow-up ai-design
// pass that extends the wire schema.
TEST_F(PdV2MultiSchedulerTest, SwaCrossNodeCoordinationTest) {
  MultiNodeTopologyConfig config;
  config.sliding_window_size = 12;  // Accepted but not yet wired into cache_manager_config; see above.
  MultiNodeTopology topo(config);
  auto& decode = topo.DecodeNode(0);

  const int64_t req_id = 801;
  auto info =
      decode.SubmitRequest(req_id, /*input_token_num=*/30, /*expected_output_token_num=*/10, static_cast<int>(req_id));
  ASSERT_TRUE(WaitForCompletion(info.req->waiter, 15)) << "req_id=" << req_id << " never completed";
  EXPECT_TRUE(info.req->finish_status.OK()) << info.req->finish_status.GetMessage();
  decode.env_simulator->CheckRequestOutput(info.infer_req_group[0]);
}

// T09: partial prefix cache hit. req2 shares only a leading span of req1's
// prompt (shared_prefix_len tokens); the remainder differs, so
// PdV2DecodeHook must still forward the miss tail to peer Prefill
// (submit_calls increases again) rather than treating it as a full local
// hit. Additionally asserts:
//   (1) Decode→Prefill prefix-cache hint equals the local shared span
//       (decode_prefix_cache_offset / cached_block_count);
//   (2) SimulatedKvWriteBridge::WriteBack only transfers the miss-tail KV
//       [decode_prefix_cache_offset, prompt_len) — never re-sends the
//       Decode-local hit prefix.
TEST_F(PdV2MultiSchedulerTest, PartialPrefixCacheCrossNodeTest) {
  MultiNodeTopologyConfig config;
  config.enable_prefix_cache = true;
  config.block_token_num = 6;
  MultiNodeTopology topo(config);
  auto& decode = topo.DecodeNode(0);
  auto& prefill = topo.PrefillNode(0);

  const int64_t req1_id = 901;
  const int64_t req2_id = 902;
  const int input_token_num = 42;
  // Exact multiple of block_token_num, and below the seq-1 hard-cap for
  // input=42 (max_match_blocks = (42-1)/6 = 6), so Decode's local prefix
  // query reports shared_token_num == shared_prefix_len exactly.
  const int shared_prefix_len = 18;  // 3 blocks @ block_token_num=6.
  const uint32_t expected_decode_prefix_cache_offset = static_cast<uint32_t>(shared_prefix_len);
  const uint32_t expected_cached_block_count =
      expected_decode_prefix_cache_offset / static_cast<uint32_t>(config.block_token_num);
  const size_t expected_kv_tokens_written =
      static_cast<size_t>(input_token_num) - static_cast<size_t>(shared_prefix_len);

  auto info1 =
      decode.SubmitRequest(req1_id, input_token_num, /*expected_output_token_num=*/5, static_cast<int>(req1_id));
  ASSERT_TRUE(WaitForCompletion(info1.req->waiter, 15)) << "req1 never completed";
  EXPECT_TRUE(info1.req->finish_status.OK()) << info1.req->finish_status.GetMessage();
  decode.env_simulator->CheckRequestOutput(info1.infer_req_group[0]);
  const int submit_calls_after_req1 = decode.connector->submit_calls.load();
  EXPECT_GT(submit_calls_after_req1, 0);

  std::vector<std::pair<int, int>> seeds2{{0, static_cast<int>(req1_id)},
                                          {shared_prefix_len, static_cast<int>(req2_id)}};
  auto info2 = SubmitRequestWithSeeds(decode, req2_id, input_token_num, /*expected_output_token_num=*/5, seeds2);
  ASSERT_TRUE(WaitForCompletion(info2.req->waiter, 15)) << "req2 (partial hit) never completed";
  EXPECT_TRUE(info2.req->finish_status.OK()) << info2.req->finish_status.GetMessage();
  decode.env_simulator->CheckRequestOutput(info2.infer_req_group[0]);

  EXPECT_GT(decode.connector->submit_calls.load(), submit_calls_after_req1)
      << "the non-shared tail must still be forwarded to peer Prefill via SubmitPrefillRequest";

  FakePdV2Connector::SubmitPrefillRecord submit_rec;
  ASSERT_TRUE(decode.connector->GetLastSubmitRecord(static_cast<uint64_t>(req2_id), &submit_rec))
      << "Decode must have called SubmitPrefillRequest for req2 with a prefix-cache hint";
  EXPECT_EQ(submit_rec.decode_prefix_cache_offset, expected_decode_prefix_cache_offset)
      << "Decode→Prefill decode_prefix_cache_offset must equal the local shared prefix length";
  EXPECT_EQ(submit_rec.cached_block_count, expected_cached_block_count)
      << "Decode→Prefill cached_block_count must be decode_prefix_cache_offset / block_token_num";
  EXPECT_EQ(submit_rec.input_token_num, static_cast<size_t>(input_token_num));

  SimulatedKvWriteBridge::WriteBackRecord wb_rec;
  ASSERT_TRUE(prefill.kv_bridge->GetLastWriteBackRecord(static_cast<uint64_t>(req2_id), &wb_rec))
      << "Prefill must have WriteBack'd KV for req2's miss tail";
  EXPECT_EQ(wb_rec.decode_prefix_cache_offset, expected_decode_prefix_cache_offset);
  EXPECT_EQ(wb_rec.prompt_len, static_cast<size_t>(input_token_num));
  EXPECT_EQ(wb_rec.tokens_written, expected_kv_tokens_written)
      << "Prefill→Decode KV transfer must cover only the miss-tail tokens "
         "[decode_prefix_cache_offset, prompt_len), not the Decode-local hit prefix";
}
