/* Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * Phase A multi-scheduler test infrastructure for pd_v2 (Prefill/Decode
 * disaggregation v2). Builds an in-process M(Prefill)/N(Decode) topology of
 * real EventDrivenScheduleProcessor instances wired together through a
 * test-only FakePdV2Connector pairing, replacing the real
 * MooncakeEngineConnector + fake_tent + Redis discovery stack.
 *
 * See spec/scheduler/TAPD-1070131059135905223-pd-multi-scheduler-test/
 * 1-change-plan.md §A.6 / §B for the full design rationale.
 * ==============================================================================*/
#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "batch_scheduler/batch_scheduler_test.h"
#include "batch_scheduler/event_driven_scheduler/continuous_batching_backend.h"
#include "batch_scheduler/event_driven_scheduler/event_driven_schedule_processor.h"
#include "batch_scheduler/event_driven_scheduler/global_cache_query_hook.h"
#include "batch_scheduler/event_driven_scheduler/schedule_hook_manager.h"
#include "batch_scheduler/event_driven_scheduler/workload_balance/basic_workload_balancer.h"
#include "batch_scheduler/global_cache_fetcher.h"
#include "cache_manager/cache_manager_factory.h"
#include "pd_v2/decode/pd_v2_decode_hook.h"
#include "pd_v2/public/pd_v2_config.h"
#include "pd_v2/public/pd_v2_connector.h"
#include "pd_v2/public/pd_v2_types.h"
#include "pd_v2/wire/prefill_request_wire.h"
#include "pd_v2/wire/sampling_config_wire.h"
#include "runtime/threadpool.h"
#include "utils/request.h"

namespace ksana_llm {

// Defined in batch_scheduler_test_helper.h (already fully visible above via
// batch_scheduler_test.h). Declared here only as a heads-up for readers: do
// NOT re-forward-declare this inside namespace pd_v2 below -- that would
// shadow this (outer, ksana_llm::) definition with an incomplete inner
// ksana_llm::pd_v2::FakedBlockAllocatorGroup type.
class FakedBlockAllocatorGroup;

namespace pd_v2 {

class PdV2FakeConnectorRegistry;

// Test-only PdV2ConnectorInterface implementation. One instance per
// simulated node (Decode or Prefill, selected by config_.node_role, mirroring
// how the real MooncakeEngineConnector branches on role). Pairs directly with
// another FakePdV2Connector instance registered in the same
// PdV2FakeConnectorRegistry -- no wire (de)serialization, no real transport,
// no discovery protocol. See 1-change-plan.md §A.6.
class FakePdV2Connector : public PdV2ConnectorInterface {
 public:
  explicit FakePdV2Connector(PdV2FakeConnectorRegistry* registry);
  ~FakePdV2Connector() override = default;

  // ---- PdV2ConnectorInterface ----
  Status Initialize(const PdV2ConnectorConfig& config) override;
  Status Shutdown() override;
  // [executor-role method, reused here as the Phase A "KV address" carrier]
  // Stores kv_pool_base verbatim; SubmitPrefillRequest forwards it as
  // DecodeTarget.kv_pool_base_va (see §A.6 "KV 地址简化"). Called on the
  // Decode-role connector at PdV2SimNode setup time with a pointer to this
  // node's FakedBlockAllocatorGroup.
  Status RegisterKvPool(void* kv_pool_base, size_t total_bytes, size_t block_size, size_t num_blocks,
                        size_t num_layers) override;
  Status GetKvBufferVa(int rank, uint64_t* out_kv_pool_base_va, uint64_t* out_total_bytes) override;

  bool CanAccept(uint64_t req_id, uint32_t input_token_count) override;
  Status SubmitPrefillRequest(uint64_t req_id, const std::vector<int>& input_tokens,
                              const std::vector<int>& kv_pool_block_ids, const SamplingConfigWire& sampling_config,
                              uint32_t attn_dp_group_id, uint32_t cached_block_count,
                              uint32_t decode_prefix_cache_offset, PdRequestCompleteCb on_complete,
                              void* ctx) override;
  bool HasHealthyPrefillPeer() const override;
  Status Release(uint64_t req_id) override;

  void RegisterIncomingHook(PdIncomingReqCb on_incoming, void* ctx) override;
  Status SendPrefillComplete(const std::string& decode_engine_inference_addr, uint64_t req_id,
                             PdCompleteStatus status, const std::vector<int>& generated_tokens,
                             const PeerHealth& snapshot, const PdPrefillMetrics& prefill_metrics) override;

  // ---- Test-only extensions (not part of PdV2ConnectorInterface) ----

  // Decode role: inject the Prefill peer address list. Replaces real peer
  // discovery -- see §A.6 "节点发现简化". Safe to call once, synchronously,
  // before any request flows (no async discovery window).
  void SetPeerAddrs(std::vector<std::string> peer_addrs);

  const std::string& GetInferenceAddr() const { return config_.inference_addr; }
  NodeRole GetNodeRole() const { return config_.node_role; }

  // T05: number of SubmitPrefillRequest calls observed. Used to assert a
  // prefix-cache-hit request never reaches the peer Prefill.
  std::atomic<int> submit_calls{0};

  // T09: last SubmitPrefillRequest's prefix-cache hint (Decode → Prefill).
  // PdV2DecodeHook computes these from local prefix / planning workload and
  // forwards them so Prefill can skip already-valid leading blocks.
  struct SubmitPrefillRecord {
    uint64_t req_id = 0;
    uint32_t cached_block_count = 0;
    uint32_t decode_prefix_cache_offset = 0;
    size_t input_token_num = 0;
  };
  // Returns true if at least one SubmitPrefillRequest for req_id was observed;
  // fills *out with the most recent such record.
  bool GetLastSubmitRecord(uint64_t req_id, SubmitPrefillRecord* out) const;

  // ---- Registry-facing dispatch entry points ----
  // Invoked (via PdV2FakeConnectorRegistry::Post, i.e. off the caller's
  // thread) by the *peer* FakePdV2Connector instance. Not part of
  // PdV2ConnectorInterface; public so lambdas posted from the registry's
  // thread pool can call across connector instances.
  void DispatchIncoming(const IncomingPrefillRequest& req);
  void DispatchComplete(uint64_t req_id, PdCompleteStatus status, const std::vector<int>& generated_tokens,
                        const PdPrefillMetrics& prefill_metrics, const std::string& prefill_peer);

 private:
  struct PendingDecodeReq {
    std::string peer_addr;
    PdRequestCompleteCb on_complete = nullptr;
    void* ctx = nullptr;
  };

  PdV2FakeConnectorRegistry* registry_;
  PdV2ConnectorConfig config_;

  // Decode role only: opaque pointer to this node's FakedBlockAllocatorGroup,
  // set via RegisterKvPool. Forwarded verbatim as DecodeTarget.kv_pool_base_va.
  void* local_kv_pool_ = nullptr;

  mutable std::mutex mu_;
  std::vector<std::string> peer_addrs_;
  std::atomic<size_t> rr_cursor_{0};
  // Decode role only: req_id -> {chosen peer, completion callback}. Populated
  // by CanAccept (round-robin pick), consumed by SubmitPrefillRequest /
  // DispatchComplete.
  std::unordered_map<uint64_t, PendingDecodeReq> pending_decode_reqs_;
  // Decode role only: append-only log of SubmitPrefillRequest prefix-cache
  // hints for T09 assertions (guarded by mu_).
  std::vector<SubmitPrefillRecord> submit_records_;

  // Prefill role only.
  PdIncomingReqCb on_incoming_cb_ = nullptr;
  void* on_incoming_ctx_ = nullptr;
};

// Test-only in-process address book + shared dispatch ThreadPool for
// FakePdV2Connector instances. Replaces fake_tent + Redis-based peer
// discovery (see §A.6 "节点发现简化") and preserves the "on_incoming /
// on_complete fire off the caller's thread" contract PdV2DecodeHook /
// PdV2PrefillEndpoint rely on by dispatching through a real ThreadPool
// instead of a synchronous direct call (see §A.6 "异步派发" / §G risk item).
class PdV2FakeConnectorRegistry {
 public:
  explicit PdV2FakeConnectorRegistry(size_t worker_num);
  ~PdV2FakeConnectorRegistry();

  void Register(const std::string& addr, std::weak_ptr<FakePdV2Connector> connector);
  void Unregister(const std::string& addr);
  std::shared_ptr<FakePdV2Connector> Find(const std::string& addr);

  // Post `task` onto the shared thread pool. After DrainAndStopThreadPool()
  // has flipped stopped_, further Posts are silently dropped (logged) so a
  // late OnPrefillFinished / SendPrefillComplete cannot dereference a
  // destroyed pool.
  void Post(std::function<void()> task);

  // Clear the address book only; the thread pool keeps running. Call between
  // TEST_F cases sharing one process to avoid stale inference_addr lookups.
  void Reset();

  // Waits for every Post()-ed task to actually finish executing (see .cpp
  // for why ThreadPool::Stop alone is not sufficient), then stops the
  // shared thread pool. Callers MUST first Stop() every
  // EventDrivenScheduleProcessor AND JoinDrivingThread() on every node so
  // no new Post() calls can race the shutdown, and MUST call this before
  // destroying any PdV2SimNode -- see §G risk item "共享 ThreadPool 的生命周期".
  void DrainAndStopThreadPool();

 private:
  mutable std::mutex mu_;
  std::unordered_map<std::string, std::weak_ptr<FakePdV2Connector>> addr_map_;
  // shared_ptr so Post() can retain the pool across the lock boundary while
  // DrainAndStopThreadPool() nulls the member and waits for pending tasks.
  std::shared_ptr<ThreadPool> pool_;
  // Number of Post()-ed tasks not yet finished executing. See
  // DrainAndStopThreadPool()'s doc for why this (rather than relying on
  // ThreadPool::Stop alone) is required for a genuine drain.
  std::atomic<int> pending_tasks_{0};
  // Set true at the start of DrainAndStopThreadPool(); Post() becomes a
  // no-op afterwards. Protects against driving-thread / waiter callbacks
  // that outlive the pool.
  bool stopped_ = false;
};

// Test-only stand-in for PdV2KvDispatcher/PdV2KvBarrier (the real per-layer
// RDMA write path; out of scope for Phase A, see §A.4 / §G "Phase B"). Runs
// on the Prefill side once a request's fake-generated tokens are ready;
// directly pokes the fake KV content for the freshly-prefilled prompt span
// into the Decode-side FakedBlockAllocatorGroup located via
// DecodeTarget::kv_pool_base_va (§A.6 "KV 地址简化").
//
// Only the *prompt* span [decode_prefix_cache_offset, input_tokens.size())
// is written here. The first generated token's KV entry is intentionally
// left for Decode's own next RunAStep call to record: PdV2DecodeHook sets
// kv_cached_token_num = input_tokens.size(), so Decode's own recording loop
// (`for i in [kv_cached_token_num, forwarding_tokens.size())`) naturally
// covers exactly that one position on its first post-handoff step. See
// 2-coding-snapshot.md for the full derivation.
class SimulatedKvWriteBridge {
 public:
  explicit SimulatedKvWriteBridge(std::chrono::microseconds artificial_delay = std::chrono::microseconds(0));

  // T07: inject an artificial delay between "content ready" and "write
  // complete" so the completion-vs-kv-write ordering becomes observable.
  void SetArtificialDelay(std::chrono::microseconds delay);

  // Copies the prompt span [decode_prefix_cache_offset, prompt_len) of
  // prefill_kv_snapshot into decode_block_allocator_group (addressed by
  // decode_kv_block_ids). The snapshot is what the Prefill node actually
  // recorded into its own blocks, captured by
  // PdV2PrefillIntakeAdapter::SnapshotPrefillKv while those blocks are still
  // alive -- a Prefill that recorded nothing / recorded into the wrong blocks
  // therefore fails the Decode-side RunAStep self-check instead of being
  // silently synthesised from input_tokens.
  // Returns false (without throwing) on any size/bounds mismatch, so T02 can
  // assert "never touches out-of-range blocks" without a fatal check
  // aborting the test binary.
  bool WriteBack(FakedBlockAllocatorGroup* decode_block_allocator_group, const std::vector<int>& decode_kv_block_ids,
                 const std::vector<int>& prefill_kv_snapshot, const std::vector<int>& prefill_input_tokens,
                 uint32_t decode_prefix_cache_offset, size_t block_token_num, uint64_t req_id_for_log);

  // steady_clock microseconds timestamp of the most recently completed
  // WriteBack call. T07 asserts SendPrefillComplete's dispatch is observed
  // strictly after this.
  int64_t LastWriteBackCompleteTimeUs() const;

  // T09: last successful WriteBack's transfer span. tokens_written equals
  // prompt_len - decode_prefix_cache_offset — i.e. only the miss-tail KV,
  // never the Decode-local prefix-cache hit span.
  struct WriteBackRecord {
    uint64_t req_id = 0;
    uint32_t decode_prefix_cache_offset = 0;
    size_t prompt_len = 0;
    size_t tokens_written = 0;
  };
  // Returns true if at least one successful WriteBack for req_id was observed;
  // fills *out with the most recent such record.
  bool GetLastWriteBackRecord(uint64_t req_id, WriteBackRecord* out) const;

 private:
  std::chrono::microseconds artificial_delay_;
  mutable std::mutex mu_;
  int64_t last_write_back_complete_time_us_ = 0;
  std::vector<WriteBackRecord> write_back_records_;
};

// Test-only bridge: converts an IncomingPrefillRequest (received by a
// Prefill-role FakePdV2Connector) into a Prefill-side InferRequest and
// submits it directly to that node's EventDrivenScheduleProcessor, bypassing
// LocalEndpoint::SubmitAsync / ModelDriver::Enqueue (too heavy for a
// scheduler-level test, see §A.3). Field mapping mirrors
// PdV2PrefillEndpoint::OnIncoming (csrc/endpoints/pd_v2/pd_v2_prefill_endpoint.cpp)
// and ModelDriver::Enqueue's is_pd_v2_prefill branch (csrc/engine/model_driver.cpp).
class PdV2PrefillIntakeAdapter {
 public:
  struct Deps {
    EventDrivenScheduleProcessor* prefill_eds = nullptr;
    BatchSchedulerEnvironmentSimulator* prefill_env_simulator = nullptr;
    std::shared_ptr<PdV2ConnectorInterface> prefill_connector = nullptr;
    std::shared_ptr<SimulatedKvWriteBridge> kv_bridge = nullptr;
    size_t block_token_num = 16;
  };

  explicit PdV2PrefillIntakeAdapter(Deps deps);

  static void OnIncomingTrampoline(const IncomingPrefillRequest& req, void* ctx);
  void OnIncoming(const IncomingPrefillRequest& req);

  // Captures the prompt-span KV contents the Prefill node just recorded, for
  // every scheduled request this adapter admitted. MUST be called from the
  // driving loop right after BatchSchedulerEnvironmentSimulator::RunAStep
  // (while still holding the EDS lock): by the time the request reaches its
  // terminal state, EventDrivenScheduleProcessor has already run
  // DestroyFinishedRequest + kv_cache_blocks.clear() (see
  // event_driven_schedule_processor.cpp), so reading Prefill KV from
  // OnPrefillFinished is too late.
  void SnapshotPrefillKv(const std::vector<std::shared_ptr<InferRequest>>& scheduled_reqs);

  size_t NumAdmittedForTest() const;

 private:
  struct PendingMeta {
    std::vector<DecodeTargetLite> decode_targets;
    std::string decode_engine_inference_addr;
    uint32_t input_token_num = 0;
    // Prompt-span KV contents captured by SnapshotPrefillKv; consumed by
    // SimulatedKvWriteBridge::WriteBack at completion time.
    std::vector<int> prefill_kv_snapshot;
  };

  // Ownership bundle for one admitted Prefill request. InferRequest only
  // holds C++ references into Request members, and Request (via the
  // KsanaPythonInput constructor used by InitRequest) holds references into
  // KsanaPythonInput -- neither shared_ptr extends the other's lifetime. Both
  // must stay alive until OnPrefillFinished confirms the InferRequest is done.
  struct AliveIntake {
    std::shared_ptr<Request> request;
    std::shared_ptr<KsanaPythonInput> python_input;
  };

  void OnPrefillFinished(uint64_t req_id);

  Deps deps_;
  mutable std::mutex mu_;
  size_t admitted_count_ = 0;
  // Keeps Request + KsanaPythonInput alive until the request finishes and
  // OnPrefillFinished fires. Mirrors PdV2PrefillCompletion's "alive map"
  // role without pulling in the real KV-barrier plumbing (out of scope).
  std::unordered_map<uint64_t, AliveIntake> alive_intakes_;
  std::unordered_map<uint64_t, PendingMeta> pending_meta_;
};

// Encapsulates one simulated node (Decode or Prefill): a real
// EventDrivenScheduleProcessor + FakePdV2Connector (this node's role) +
// FakedBlockAllocatorGroup + BatchSchedulerEnvironmentSimulator, plus a
// dedicated background thread driving the GetNextScheduleResult /
// RunAStep / UpdateWithGenerationResult loop (mirrors
// ParallelTester::DoParallelRequestAndCheck's inner loop, backgrounded so
// multiple nodes can run concurrently).
struct PdV2SimNode {
  NodeRole role = NodeRole::kDecode;
  std::string inference_addr;
  int tp_num = 1;

  BlockManagerConfig block_manager_config;
  BatchSchedulerConfig batch_scheduler_config;
  CacheManagerConfig cache_manager_config;
  RuntimeConfig runtime_config;

  std::shared_ptr<FakedBlockAllocatorGroup> block_allocator_group;
  std::shared_ptr<FakedBlockAllocatorGroup> swa_block_allocator_group;  // non-null only when SWA enabled
  std::shared_ptr<CacheManagerInterface> cache_manager;
  std::unique_ptr<BatchSchedulerEnvironmentSimulator> env_simulator;
  std::shared_ptr<EventDrivenScheduleProcessor> eds;
  std::shared_ptr<FakePdV2Connector> connector;

  // Decode role only.
  std::shared_ptr<GlobalCacheFetcher> global_cache_fetcher;
  std::shared_ptr<PdV2DecodeHook> decode_hook;

  // Prefill role only.
  std::shared_ptr<SimulatedKvWriteBridge> kv_bridge;
  std::shared_ptr<PdV2PrefillIntakeAdapter> prefill_intake;

  std::thread driving_thread;
  bool driving_started = false;

  ~PdV2SimNode();

  void StartDrivingLoop();
  // Stops the EventDrivenScheduleProcessor. Idempotent. MUST be called (on
  // every node) before JoinDrivingThread().
  void StopEds();
  // Joins the driving thread. MUST be called only after StopEds() (on this
  // node). MultiNodeTopology::TearDown joins every driving thread before
  // draining the shared registry pool so late Post() callers cannot race a
  // destroyed ThreadPool (Post is also stopped_-guarded).
  void JoinDrivingThread();

  // Decode role convenience: builds a fresh fake request via env_simulator's
  // InitRequest (reusing BatchSchedulerEnvironmentSimulator's fake-token
  // bookkeeping, per §B "复用 RequestInfo/GenerateRequests 风格") and submits
  // it to this node's EDS. Returns the RequestInfo (holding the InferRequest
  // group / Request / KsanaPythonInput) so the caller can keep it alive and
  // wait on infer_req_group[0]->waiter, then check via
  // env_simulator->CheckRequestOutput(...).
  //
  // IMPORTANT: pass seed == req_id (the codebase-wide convention used by
  // ParallelTester::GenerateRequests). PdV2PrefillIntakeAdapter bootstraps
  // the Prefill-side env_simulator's fake-token seed for the same req_id
  // from req_id itself (see pd_v2_multi_scheduler_test_helper.cpp), so a
  // mismatched seed here would make the two nodes' GenerateAFakeToken hash
  // chains diverge and CheckRequestOutput would fail.
  ParallelTester::RequestInfo SubmitRequest(int64_t req_id, int input_token_num, int expected_output_token_num,
                                            int seed);
};

// Parameters controlling MultiNodeTopology construction.
struct MultiNodeTopologyConfig {
  int num_prefill = 1;
  int num_decode = 1;
  int tp_num = 1;
  size_t device_block_num = 200;
  // 0 = use device_block_num for Prefill too. Set smaller to model T06
  // "Prefill busy backoff" resource pressure.
  size_t prefill_device_block_num = 0;
  size_t block_token_num = 6;
  bool enable_prefix_cache = false;
  size_t sliding_window_size = 0;  // 0 = SWA disabled
  int local_complete_threshold = 0;
  size_t registry_worker_num = 4;
};

// Builds and owns an M(Prefill)/N(Decode) in-process pd_v2 topology.
// Construction wires every Decode node's FakePdV2Connector peer list to all
// Prefill inference_addrs (round-robin target set) via a single synchronous
// SetPeerAddrs call per §B "MultiNodeTopology 辅助函数" -- no async discovery
// window. TearDown() (also invoked from the destructor) follows:
// stop every EDS -> join driving threads -> drain the shared registry pool
// (Post is also stopped_-guarded against late callbacks).
class MultiNodeTopology {
 public:
  explicit MultiNodeTopology(MultiNodeTopologyConfig config);
  ~MultiNodeTopology();

  MultiNodeTopology(const MultiNodeTopology&) = delete;
  MultiNodeTopology& operator=(const MultiNodeTopology&) = delete;

  void TearDown();

  PdV2SimNode& DecodeNode(size_t idx) { return *decode_nodes_.at(idx); }
  PdV2SimNode& PrefillNode(size_t idx) { return *prefill_nodes_.at(idx); }
  size_t NumDecodeNodes() const { return decode_nodes_.size(); }
  size_t NumPrefillNodes() const { return prefill_nodes_.size(); }
  PdV2FakeConnectorRegistry& Registry() { return *registry_; }

 private:
  std::unique_ptr<PdV2SimNode> BuildNode(NodeRole role, const std::string& addr, size_t device_block_num);

  MultiNodeTopologyConfig config_;
  std::unique_ptr<PdV2FakeConnectorRegistry> registry_;
  std::vector<std::unique_ptr<PdV2SimNode>> prefill_nodes_;
  std::vector<std::unique_ptr<PdV2SimNode>> decode_nodes_;
  bool torn_down_ = false;
};

}  // namespace pd_v2
}  // namespace ksana_llm
