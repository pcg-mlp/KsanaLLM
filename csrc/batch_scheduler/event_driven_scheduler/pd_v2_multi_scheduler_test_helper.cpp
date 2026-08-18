/* Copyright 2026 Tencent Inc.  All rights reserved.
==============================================================================*/
#include "batch_scheduler/event_driven_scheduler/pd_v2_multi_scheduler_test_helper.h"

#include <chrono>

#include "configure/environment.h"
#include "helpers/environment_test_helper.h"
#include "profiler/timer.h"
#include "utils/logger.h"
#include "utils/singleton.h"
#include "utils/status.h"

namespace ksana_llm {
namespace pd_v2 {

namespace {

SamplingConfig WireToSamplingConfig(const SamplingConfigWire& w) {
  SamplingConfig sc;
  sc.topk = w.topk;
  sc.topp = w.topp;
  sc.temperature = w.temperature;
  sc.repetition_penalty = w.repetition_penalty;
  sc.max_new_tokens = w.max_new_tokens;
  sc.stop_token_ids.assign(w.stop_token_ids.begin(), w.stop_token_ids.end());
  sc.stop_strings = w.stop_strings;
  sc.ignore_eos = w.ignore_eos;
  sc.json_schema = w.json_schema;
  sc.enable_structured_output = w.enable_structured_output;
  sc.enable_thinking = w.enable_thinking;
  sc.ptp_token_num = w.ptp_token_num;
  return sc;
}

}  // namespace

// ================= FakePdV2Connector =================

FakePdV2Connector::FakePdV2Connector(PdV2FakeConnectorRegistry* registry) : registry_(registry) {}

Status FakePdV2Connector::Initialize(const PdV2ConnectorConfig& config) {
  config_ = config;
  return Status();
}

Status FakePdV2Connector::Shutdown() { return Status(); }

Status FakePdV2Connector::RegisterKvPool(void* kv_pool_base, size_t total_bytes, size_t /*block_size*/,
                                         size_t /*num_blocks*/, size_t /*num_layers*/) {
  local_kv_pool_ = kv_pool_base;
  return Status();
}

Status FakePdV2Connector::GetKvBufferVa(int /*rank*/, uint64_t* out_kv_pool_base_va, uint64_t* out_total_bytes) {
  if (out_kv_pool_base_va != nullptr) {
    *out_kv_pool_base_va = reinterpret_cast<uint64_t>(local_kv_pool_);
  }
  if (out_total_bytes != nullptr) {
    *out_total_bytes = 0;
  }
  return Status();
}

bool FakePdV2Connector::CanAccept(uint64_t req_id, uint32_t /*input_token_count*/) {
  std::lock_guard<std::mutex> lk(mu_);
  if (peer_addrs_.empty()) {
    return false;
  }
  const size_t idx = rr_cursor_.fetch_add(1, std::memory_order_relaxed) % peer_addrs_.size();
  pending_decode_reqs_[req_id] = PendingDecodeReq{peer_addrs_[idx], nullptr, nullptr};
  return true;
}

Status FakePdV2Connector::SubmitPrefillRequest(uint64_t req_id, const std::vector<int>& input_tokens,
                                               const std::vector<int>& kv_pool_block_ids,
                                               const SamplingConfigWire& sampling_config, uint32_t attn_dp_group_id,
                                               uint32_t cached_block_count, uint32_t decode_prefix_cache_offset,
                                               PdRequestCompleteCb on_complete, void* ctx) {
  std::string peer_addr;
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = pending_decode_reqs_.find(req_id);
    if (it == pending_decode_reqs_.end()) {
      return Status(RET_INVALID_ARGUMENT, "SubmitPrefillRequest: req_id not admitted via CanAccept");
    }
    it->second.on_complete = on_complete;
    it->second.ctx = ctx;
    peer_addr = it->second.peer_addr;
  }

  auto target = registry_->Find(peer_addr);
  if (target == nullptr) {
    return Status(RET_INVALID_ARGUMENT, "SubmitPrefillRequest: unknown/dead Prefill peer addr " + peer_addr);
  }
  submit_calls.fetch_add(1, std::memory_order_relaxed);
  {
    std::lock_guard<std::mutex> lk(mu_);
    SubmitPrefillRecord record;
    record.req_id = req_id;
    record.cached_block_count = cached_block_count;
    record.decode_prefix_cache_offset = decode_prefix_cache_offset;
    record.input_token_num = input_tokens.size();
    submit_records_.push_back(record);
  }

  // Own the wire objects via shared_ptr so they outlive this call; the
  // posted lambda keeps them alive until DispatchIncoming (called
  // synchronously inside the lambda) returns. No byte-level serialization --
  // see class doc.
  auto wire = std::make_shared<SamplingConfigWire>(sampling_config);
  auto targets = std::make_shared<std::vector<DecodeTarget>>();
  DecodeTarget target_entry;
  target_entry.decode_attn_tp_rank_in_group = 0;
  target_entry.kv_pool_base_va = reinterpret_cast<uint64_t>(local_kv_pool_);
  target_entry.kv_block_ids = kv_pool_block_ids;
  target_entry.cached_block_count = cached_block_count;
  target_entry.decode_prefix_cache_offset = decode_prefix_cache_offset;
  targets->push_back(std::move(target_entry));

  const std::string self_addr = config_.inference_addr;
  const std::vector<int> tokens_copy = input_tokens;
  registry_->Post([target, self_addr, req_id, tokens_copy, wire, targets, attn_dp_group_id]() {
    IncomingPrefillRequest req;
    req.decode_engine_inference_addr = self_addr;
    req.req_id = req_id;
    req.input_tokens = tokens_copy;
    req.sampling_config_wire = wire.get();
    req.decode_targets_wire = targets->data();
    req.num_decode_targets = static_cast<uint32_t>(targets->size());
    req.decode_attn_dp_group_id = attn_dp_group_id;
    target->DispatchIncoming(req);
  });
  return Status();
}

bool FakePdV2Connector::HasHealthyPrefillPeer() const {
  std::lock_guard<std::mutex> lk(mu_);
  return !peer_addrs_.empty();
}

Status FakePdV2Connector::Release(uint64_t req_id) {
  std::lock_guard<std::mutex> lk(mu_);
  pending_decode_reqs_.erase(req_id);
  return Status();
}

void FakePdV2Connector::RegisterIncomingHook(PdIncomingReqCb on_incoming, void* ctx) {
  on_incoming_cb_ = on_incoming;
  on_incoming_ctx_ = ctx;
}

Status FakePdV2Connector::SendPrefillComplete(const std::string& decode_engine_inference_addr, uint64_t req_id,
                                              PdCompleteStatus status, const std::vector<int>& generated_tokens,
                                              const PeerHealth& /*snapshot*/, const PdPrefillMetrics& prefill_metrics) {
  auto target = registry_->Find(decode_engine_inference_addr);
  if (target == nullptr) {
    return Status(RET_INVALID_ARGUMENT,
                  "SendPrefillComplete: unknown/dead Decode peer addr " + decode_engine_inference_addr);
  }
  const std::string self_addr = config_.inference_addr;
  const std::vector<int> tokens_copy = generated_tokens;
  const PdPrefillMetrics metrics_copy = prefill_metrics;
  registry_->Post([target, req_id, status, tokens_copy, metrics_copy, self_addr]() {
    target->DispatchComplete(req_id, status, tokens_copy, metrics_copy, self_addr);
  });
  return Status();
}

void FakePdV2Connector::SetPeerAddrs(std::vector<std::string> peer_addrs) {
  std::lock_guard<std::mutex> lk(mu_);
  peer_addrs_ = std::move(peer_addrs);
}

bool FakePdV2Connector::GetLastSubmitRecord(uint64_t req_id, SubmitPrefillRecord* out) const {
  if (out == nullptr) {
    return false;
  }
  std::lock_guard<std::mutex> lk(mu_);
  for (auto it = submit_records_.rbegin(); it != submit_records_.rend(); ++it) {
    if (it->req_id == req_id) {
      *out = *it;
      return true;
    }
  }
  return false;
}

void FakePdV2Connector::DispatchIncoming(const IncomingPrefillRequest& req) {
  if (on_incoming_cb_ != nullptr) {
    on_incoming_cb_(req, on_incoming_ctx_);
  }
}

void FakePdV2Connector::DispatchComplete(uint64_t req_id, PdCompleteStatus status,
                                         const std::vector<int>& generated_tokens,
                                         const PdPrefillMetrics& prefill_metrics, const std::string& prefill_peer) {
  PdRequestCompleteCb cb = nullptr;
  void* ctx = nullptr;
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = pending_decode_reqs_.find(req_id);
    if (it == pending_decode_reqs_.end()) {
      KLLM_LOG_WARNING << "FakePdV2Connector::DispatchComplete: unknown req_id=" << req_id << ", dropping";
      return;
    }
    cb = it->second.on_complete;
    ctx = it->second.ctx;
    pending_decode_reqs_.erase(it);
  }
  if (cb != nullptr) {
    cb(req_id, status, generated_tokens, prefill_metrics, prefill_peer, ctx);
  }
}

// ================= PdV2FakeConnectorRegistry =================

PdV2FakeConnectorRegistry::PdV2FakeConnectorRegistry(size_t worker_num)
    : pool_(std::make_shared<ThreadPool>(worker_num > 0 ? worker_num : 1)) {
  pool_->Start();
}

PdV2FakeConnectorRegistry::~PdV2FakeConnectorRegistry() {
  if (pool_ || !stopped_) {
    DrainAndStopThreadPool();
  }
}

void PdV2FakeConnectorRegistry::Register(const std::string& addr, std::weak_ptr<FakePdV2Connector> connector) {
  std::lock_guard<std::mutex> lk(mu_);
  addr_map_[addr] = std::move(connector);
}

void PdV2FakeConnectorRegistry::Unregister(const std::string& addr) {
  std::lock_guard<std::mutex> lk(mu_);
  addr_map_.erase(addr);
}

std::shared_ptr<FakePdV2Connector> PdV2FakeConnectorRegistry::Find(const std::string& addr) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = addr_map_.find(addr);
  if (it == addr_map_.end()) {
    return nullptr;
  }
  return it->second.lock();
}

void PdV2FakeConnectorRegistry::Reset() {
  std::lock_guard<std::mutex> lk(mu_);
  addr_map_.clear();
}

void PdV2FakeConnectorRegistry::Post(std::function<void()> task) {
  std::shared_ptr<ThreadPool> pool;
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (stopped_ || pool_ == nullptr) {
      KLLM_LOG_WARNING << "PdV2FakeConnectorRegistry::Post: pool already stopped, dropping task";
      return;
    }
    pool = pool_;
    pending_tasks_.fetch_add(1, std::memory_order_relaxed);
  }
  auto* self = this;
  try {
    pool->Submit([self, task = std::move(task)]() {
      task();
      self->pending_tasks_.fetch_sub(1, std::memory_order_relaxed);
    });
  } catch (...) {
    pending_tasks_.fetch_sub(1, std::memory_order_relaxed);
    throw;
  }
}

void PdV2FakeConnectorRegistry::DrainAndStopThreadPool() {
  std::shared_ptr<ThreadPool> pool;
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (stopped_) {
      return;
    }
    stopped_ = true;
    pool = std::move(pool_);
  }
  if (pool == nullptr) {
    return;
  }
  // ThreadPool::Stop (runtime/threadpool.h) only lets each worker finish AT
  // MOST one already-popped task after stopped_ flips true -- its outer
  // `while (!stopped_)` loop condition is checked BEFORE the next pop, so a
  // worker that is mid-task when Stop() is called will not loop back for a
  // second queued task. With more queued tasks than workers, Stop() alone
  // can leave tasks un-executed. Busy-poll our own counter first so every
  // Post()-ed task genuinely completes before workers are joined -- see
  // 1-change-plan.md §G risk item "共享 ThreadPool 的生命周期".
  while (pending_tasks_.load(std::memory_order_relaxed) > 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  pool->Stop();
}

// ================= SimulatedKvWriteBridge =================

SimulatedKvWriteBridge::SimulatedKvWriteBridge(std::chrono::microseconds artificial_delay)
    : artificial_delay_(artificial_delay) {}

void SimulatedKvWriteBridge::SetArtificialDelay(std::chrono::microseconds delay) { artificial_delay_ = delay; }

bool SimulatedKvWriteBridge::WriteBack(FakedBlockAllocatorGroup* decode_block_allocator_group,
                                       const std::vector<int>& decode_kv_block_ids,
                                       const std::vector<int>& prefill_kv_snapshot,
                                       const std::vector<int>& prefill_input_tokens,
                                       uint32_t decode_prefix_cache_offset, size_t block_token_num,
                                       uint64_t req_id_for_log) {
  if (decode_block_allocator_group == nullptr || block_token_num == 0) {
    return false;
  }
  const size_t prompt_len = prefill_input_tokens.size();
  if (decode_prefix_cache_offset > prompt_len) {
    return false;
  }
  const size_t needed_blocks = (prompt_len + block_token_num - 1) / block_token_num;
  if (decode_kv_block_ids.size() < needed_blocks) {
    return false;
  }
  if (prefill_kv_snapshot.size() < prompt_len) {
    KLLM_LOG_ERROR << "SimulatedKvWriteBridge::WriteBack: missing/short Prefill KV snapshot, req_id=" << req_id_for_log
                   << " snapshot_size=" << prefill_kv_snapshot.size() << " prompt_len=" << prompt_len;
    return false;
  }

  // FakedBlockAllocatorGroup::RecordGeneratedToken is the only public write
  // surface it exposes (see batch_scheduler_test_helper.h, kept unmodified
  // per plan). Build a throwaway InferRequest whose kv_cache_blocks aliases
  // decode_kv_block_ids purely to reuse that addressing logic instead of
  // duplicating it.
  SamplingConfig dummy_sc;
  auto dummy_request = std::make_shared<Request>(static_cast<int64_t>(req_id_for_log), prefill_input_tokens, dummy_sc);
  auto dummy_infer_req = std::make_shared<InferRequest>(dummy_request, 0);
  dummy_infer_req->kv_cache_blocks = decode_kv_block_ids;

  if (artificial_delay_.count() > 0) {
    std::this_thread::sleep_for(artificial_delay_);
  }

  for (size_t offset = decode_prefix_cache_offset; offset < prompt_len; ++offset) {
    decode_block_allocator_group->RecordGeneratedToken(dummy_infer_req, static_cast<int>(offset),
                                                       prefill_kv_snapshot[offset]);
  }

  {
    std::lock_guard<std::mutex> lk(mu_);
    last_write_back_complete_time_us_ = ProfileTimer::GetCurrentTimeInUs();
    WriteBackRecord record;
    record.req_id = req_id_for_log;
    record.decode_prefix_cache_offset = decode_prefix_cache_offset;
    record.prompt_len = prompt_len;
    record.tokens_written = prompt_len - decode_prefix_cache_offset;
    write_back_records_.push_back(record);
  }
  return true;
}

int64_t SimulatedKvWriteBridge::LastWriteBackCompleteTimeUs() const {
  std::lock_guard<std::mutex> lk(mu_);
  return last_write_back_complete_time_us_;
}

bool SimulatedKvWriteBridge::GetLastWriteBackRecord(uint64_t req_id, WriteBackRecord* out) const {
  if (out == nullptr) {
    return false;
  }
  std::lock_guard<std::mutex> lk(mu_);
  for (auto it = write_back_records_.rbegin(); it != write_back_records_.rend(); ++it) {
    if (it->req_id == req_id) {
      *out = *it;
      return true;
    }
  }
  return false;
}

// ================= PdV2PrefillIntakeAdapter =================

PdV2PrefillIntakeAdapter::PdV2PrefillIntakeAdapter(Deps deps) : deps_(std::move(deps)) {
  KLLM_CHECK_WITH_INFO(deps_.prefill_eds != nullptr, "PdV2PrefillIntakeAdapter: prefill_eds is null");
  KLLM_CHECK_WITH_INFO(deps_.prefill_env_simulator != nullptr,
                       "PdV2PrefillIntakeAdapter: prefill_env_simulator is null");
  KLLM_CHECK_WITH_INFO(deps_.prefill_connector != nullptr, "PdV2PrefillIntakeAdapter: prefill_connector is null");
  KLLM_CHECK_WITH_INFO(deps_.kv_bridge != nullptr, "PdV2PrefillIntakeAdapter: kv_bridge is null");
}

void PdV2PrefillIntakeAdapter::OnIncomingTrampoline(const IncomingPrefillRequest& req, void* ctx) {
  auto* self = static_cast<PdV2PrefillIntakeAdapter*>(ctx);
  if (self == nullptr) {
    return;
  }
  self->OnIncoming(req);
}

void PdV2PrefillIntakeAdapter::OnIncoming(const IncomingPrefillRequest& req) {
  if (req.sampling_config_wire == nullptr) {
    KLLM_LOG_ERROR << "PdV2PrefillIntakeAdapter::OnIncoming: null sampling_config_wire, drop req_id=" << req.req_id;
    return;
  }
  const auto* wire_sc = static_cast<const SamplingConfigWire*>(req.sampling_config_wire);
  const SamplingConfig sc = WireToSamplingConfig(*wire_sc);

  // Bootstrap this node's env_simulator bookkeeping (req_output_num_map_ /
  // req_generation_seeds_) for a req_id this node did not itself originate
  // -- required so GenerateAFakeToken's hash chain matches the Decode
  // node's for the same req_id. Per the codebase-wide convention (see
  // ParallelTester::GenerateRequests: `seeds[0].second = info.req_id`),
  // each request's fake-token seed is derived from its own req_id -- NOT a
  // single topology-wide constant -- so this must match whatever seed the
  // Decode-side test harness used when it originated this same req_id (test
  // callers MUST pass seed == req_id to PdV2SimNode::SubmitRequest).
  //
  // expected_output_token_num MUST be 2 here, not 1: BatchSchedulerEnvironmentSimulator
  // treats N as "real tokens + EOS" (see RunAStep: generates while
  // output_len < N-1). With N=1 the generate branch never runs, so Prefill
  // keeps the initial GetEndId()=-1 and Decode would treat that EOS as the
  // first token and stop immediately -- breaking CheckRequestOutput for the
  // PD round-trip cases. Prefill itself is still capped to one real token by
  // the Decode-side wire max_new_tokens=1, so N=2 does not over-generate.
  //
  // InferRequest only holds references into Request / KsanaPythonInput members
  // (see InferRequest ctor), so both objects MUST stay alive for the whole
  // Prefill lifetime -- we keep them in alive_intakes_ until OnPrefillFinished.
  // InitRequest is therefore the sole source of Request + InferRequest here
  // (no throwaway + second Request), which also keeps bookkeeping keyed on
  // the same InferRequest id that EDS will process.
  AliveIntake intake;
  std::vector<std::pair<int, int>> seeds{{0, static_cast<int>(req.req_id)}};
  auto group = deps_.prefill_env_simulator->InitRequest(
      static_cast<int64_t>(req.req_id), static_cast<int>(req.input_tokens.size()),
      /*expected_output_token_num=*/2, intake.request, intake.python_input, seeds);
  if (group.empty() || intake.request == nullptr || intake.python_input == nullptr) {
    KLLM_LOG_ERROR << "PdV2PrefillIntakeAdapter::OnIncoming: InitRequest failed req_id=" << req.req_id;
    return;
  }

  const uint64_t req_id = req.req_id;
  auto& request = intake.request;
  // Wire is authoritative for the prompt span (Decode already generated these
  // tokens). InitRequest's GenerateFakeTokens should match under the seed==req_id
  // convention, but syncing from the wire avoids any drift before Prefill runs.
  request->input_tokens = req.input_tokens;
  request->output_tokens = req.input_tokens;
  // Wire sampling overrides InitRequest defaults (esp. max_new_tokens=1 from Decode).
  // InferRequest::sampling_config is a reference into request->sampling_config.
  request->sampling_config = sc;
  request->waiter = std::make_shared<Waiter>(/*num_wait=*/1, [this, req_id]() { this->OnPrefillFinished(req_id); });

  std::vector<DecodeTargetLite> targets_lite;
  if (req.decode_targets_wire != nullptr && req.num_decode_targets > 0) {
    const auto* targets_wire = static_cast<const DecodeTarget*>(req.decode_targets_wire);
    targets_lite.reserve(req.num_decode_targets);
    for (uint32_t i = 0; i < req.num_decode_targets; ++i) {
      targets_lite.push_back(targets_wire[i].ToLite());
    }
  }

  request->is_pd_v2_prefill = true;
  request->pd_v2_decode_targets = targets_lite;
  request->pd_v2_decode_peer_addr = req.decode_engine_inference_addr;
  request->pd_v2_decode_attn_dp_group_id = req.decode_attn_dp_group_id;

  auto& infer_req = group[0];
  infer_req->kv_cache_blocks.clear();
  infer_req->block_token_num = deps_.block_token_num;
  infer_req->infer_stage = InferStage::kContext;
  infer_req->step = 0;
  infer_req->kv_cached_token_num = 0;
  infer_req->pd_v2.decode_targets = request->pd_v2_decode_targets;
  infer_req->pd_v2.decode_engine_inference_addr = request->pd_v2_decode_peer_addr;
  infer_req->pd_v2.decode_attn_dp_group_id = request->pd_v2_decode_attn_dp_group_id;
  infer_req->SetReqGroup(group);

  {
    std::lock_guard<std::mutex> lk(mu_);
    alive_intakes_[req_id] = std::move(intake);
    PendingMeta meta;
    meta.decode_targets = targets_lite;
    meta.decode_engine_inference_addr = req.decode_engine_inference_addr;
    meta.input_token_num = static_cast<uint32_t>(req.input_tokens.size());
    pending_meta_[req_id] = std::move(meta);
    ++admitted_count_;
  }

  Status enqueue_status = deps_.prefill_eds->AddInferRequest(group);
  if (!enqueue_status.OK()) {
    KLLM_LOG_ERROR << "PdV2PrefillIntakeAdapter::OnIncoming: AddInferRequest failed req_id=" << req_id
                   << " err=" << enqueue_status.GetMessage();
    {
      std::lock_guard<std::mutex> lk(mu_);
      auto it = alive_intakes_.find(req_id);
      if (it != alive_intakes_.end() && it->second.request) {
        it->second.request->finish_status = enqueue_status;
      }
    }
    // InferRequest holds references into Request/KsanaPythonInput -- drop the
    // local group (and any AddInferRequest copies, of which there are none on
    // failure) BEFORE OnPrefillFinished releases the alive intake.
    group.clear();
    OnPrefillFinished(req_id);
  }
}

void PdV2PrefillIntakeAdapter::OnPrefillFinished(uint64_t req_id) {
  AliveIntake intake;
  PendingMeta meta;
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto req_it = alive_intakes_.find(req_id);
    if (req_it == alive_intakes_.end()) {
      return;
    }
    intake = std::move(req_it->second);
    alive_intakes_.erase(req_it);
    auto meta_it = pending_meta_.find(req_id);
    if (meta_it != pending_meta_.end()) {
      meta = std::move(meta_it->second);
      pending_meta_.erase(meta_it);
    }
  }

  auto& request = intake.request;
  if (request == nullptr) {
    return;
  }

  std::vector<int> generated_tokens;
  {
    std::lock_guard<std::mutex> ol(request->output_mutex);
    const size_t prompt_len = request->input_tokens.size();
    if (request->output_tokens.size() > prompt_len) {
      generated_tokens.assign(request->output_tokens.begin() + prompt_len, request->output_tokens.end());
    }
  }

  PdCompleteStatus status =
      (request->finish_status.OK() && !generated_tokens.empty()) ? PdCompleteStatus::kOk : PdCompleteStatus::kFailed;

  if (status == PdCompleteStatus::kOk) {
    // Bridge the freshly-prefilled prompt KV into the Decode node's pool
    // BEFORE acking completion, so T07's ordering assertion (write-back
    // precedes SendPrefillComplete dispatch) holds by construction. The KV
    // values come from SnapshotPrefillKv (taken while the Prefill blocks were
    // still allocated), not from a read of the now-released blocks.
    for (const auto& target : meta.decode_targets) {
      auto* decode_group = reinterpret_cast<FakedBlockAllocatorGroup*>(target.kv_pool_base_va);
      const bool ok =
          deps_.kv_bridge->WriteBack(decode_group, target.kv_block_ids, meta.prefill_kv_snapshot, request->input_tokens,
                                     target.decode_prefix_cache_offset, deps_.block_token_num, req_id);
      if (!ok) {
        KLLM_LOG_ERROR << "PdV2PrefillIntakeAdapter: SimulatedKvWriteBridge::WriteBack failed req_id=" << req_id;
        status = PdCompleteStatus::kFailed;
      }
    }
  }

  PdPrefillMetrics metrics;  // Phase A: timing not modeled, zero-valued.
  deps_.prefill_connector->SendPrefillComplete(meta.decode_engine_inference_addr, req_id, status, generated_tokens,
                                               PeerHealth{}, metrics);
  // intake (Request + KsanaPythonInput) drops here -- InferRequest is finished.
}

void PdV2PrefillIntakeAdapter::SnapshotPrefillKv(const std::vector<std::shared_ptr<InferRequest>>& scheduled_reqs) {
  auto prefill_bag = deps_.prefill_env_simulator->GetBlockAllocatorGroup();
  if (prefill_bag == nullptr) {
    return;
  }
  for (const auto& req : scheduled_reqs) {
    if (req == nullptr || req->kv_cache_blocks.empty()) {
      continue;
    }
    const uint64_t req_id = static_cast<uint64_t>(req->req_id);
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (pending_meta_.find(req_id) == pending_meta_.end()) {
        continue;  // Not one of ours (or already completed).
      }
    }

    // Skip (rather than let CollectKvCacheContent's KLLM_CHECK throw on the
    // driving thread) if this step's blocks do not cover the whole prompt yet.
    // Any previously captured snapshot is kept; if none exists, WriteBack will
    // report the miss and the request fails visibly instead of hanging.
    const size_t prompt_len = req->input_tokens.size();
    const size_t needed_blocks =
        deps_.block_token_num > 0 ? (prompt_len + deps_.block_token_num - 1) / deps_.block_token_num : 0;
    if (needed_blocks == 0 || req->kv_cache_blocks.size() < needed_blocks) {
      KLLM_LOG_WARNING << "PdV2PrefillIntakeAdapter::SnapshotPrefillKv: blocks do not cover prompt yet, req_id="
                       << req_id << " blocks=" << req->kv_cache_blocks.size() << " needed=" << needed_blocks;
      continue;
    }

    // Shim request bounding the collected range to the prompt span:
    // CollectKvCacheContent reads output_tokens.size() positions, and
    // InferRequest::output_tokens is a reference into its Request (whose
    // pd_v2 constructor seeds output_tokens = input_tokens). Reading the full
    // req->output_tokens instead would touch the not-yet-recorded position of
    // the first generated token, which still holds DEFAULT_KV_CONTENT on every
    // device and would trip the group's tp>1 cross-device consistency check.
    SamplingConfig shim_sc;
    auto shim_request = std::make_shared<Request>(req->req_id, req->input_tokens, shim_sc);
    auto shim_infer_req = std::make_shared<InferRequest>(shim_request, 0);
    shim_infer_req->kv_cache_blocks = req->kv_cache_blocks;

    std::vector<int> snapshot;
    prefill_bag->CollectKvCacheContent(shim_infer_req, snapshot);

    std::lock_guard<std::mutex> lk(mu_);
    auto it = pending_meta_.find(req_id);
    if (it != pending_meta_.end()) {
      it->second.prefill_kv_snapshot = std::move(snapshot);
    }
  }
}

size_t PdV2PrefillIntakeAdapter::NumAdmittedForTest() const {
  std::lock_guard<std::mutex> lk(mu_);
  return admitted_count_;
}

// ================= PdV2SimNode =================

PdV2SimNode::~PdV2SimNode() {
  StopEds();
  JoinDrivingThread();
}

void PdV2SimNode::StartDrivingLoop() {
  if (driving_started) {
    return;
  }
  driving_started = true;
  driving_thread = std::thread([this]() {
    while (true) {
      std::shared_ptr<ScheduleResult> schedule_result = eds->GetNextScheduleResult();
      if (!schedule_result) {
        break;  // EDS stopped.
      }
      eds->Lock();
      auto& scheduled_reqs = schedule_result->schedule_output->running_reqs;
      env_simulator->RunAStep(scheduled_reqs);
      // Prefill role: capture the just-recorded prompt KV while the blocks are
      // still allocated. UpdateWithGenerationResult below may finish the
      // request, and its cleanup releases the blocks + clears kv_cache_blocks
      // before the completion callback runs.
      if (prefill_intake) {
        prefill_intake->SnapshotPrefillKv(scheduled_reqs);
      }
      eds->Unlock();
      eds->UpdateWithGenerationResult(schedule_result->generation_output_group);
    }
  });
}

void PdV2SimNode::StopEds() {
  if (eds) {
    eds->Stop();
  }
}

void PdV2SimNode::JoinDrivingThread() {
  if (driving_thread.joinable()) {
    driving_thread.join();
  }
}

ParallelTester::RequestInfo PdV2SimNode::SubmitRequest(int64_t req_id, int input_token_num,
                                                       int expected_output_token_num, int seed) {
  ParallelTester::RequestInfo info;
  info.req_id = req_id;
  info.input_token_num = input_token_num;
  info.expect_output_token_num = expected_output_token_num;
  std::vector<std::pair<int, int>> seeds{{0, seed}};
  info.infer_req_group = env_simulator->InitRequest(req_id, input_token_num, expected_output_token_num, info.req,
                                                    info.ksana_python_input, seeds);
  Status s = eds->AddInferRequest(info.infer_req_group);
  if (!s.OK()) {
    KLLM_LOG_ERROR << "PdV2SimNode::SubmitRequest: AddInferRequest failed req_id=" << req_id
                   << " err=" << s.GetMessage();
  }
  return info;
}

// ================= MultiNodeTopology =================

std::unique_ptr<PdV2SimNode> MultiNodeTopology::BuildNode(NodeRole role, const std::string& addr,
                                                          size_t device_block_num) {
  auto node = std::make_unique<PdV2SimNode>();
  node->role = role;
  node->inference_addr = addr;
  node->tp_num = config_.tp_num;

  node->block_manager_config.host_allocator_config.blocks_num = 0;
  node->block_manager_config.device_allocator_config.blocks_num = device_block_num;
  node->block_manager_config.device_allocator_config.block_token_num = config_.block_token_num;

  node->batch_scheduler_config.schedule_strategy = static_cast<ScheduleStrategy>(0);
  node->batch_scheduler_config.waiting_timeout_in_ms = 600000;
  node->batch_scheduler_config.max_waiting_queue_len = 256;
  node->batch_scheduler_config.max_step_token_num = 4096;
  node->batch_scheduler_config.max_batch_size = 8;
  node->batch_scheduler_config.max_token_len = 1024;
  node->batch_scheduler_config.launch_block_threshold = 4.0;
  node->batch_scheduler_config.split_fuse_token_num = 0;
  node->batch_scheduler_config.enable_async = false;
  node->batch_scheduler_config.max_pp_batch_num = 1;

  node->cache_manager_config.block_token_num = config_.block_token_num;
  node->cache_manager_config.tensor_para_size = config_.tp_num;
  node->cache_manager_config.enable_prefix_caching = config_.enable_prefix_cache;
  // SWA cross-node KV bridging is not representable with the current wire
  // schema (DecodeTarget carries no SWA block ids) -- see coding-snapshot
  // "已知不确定项" for the full rationale. Left disabled here regardless of
  // config_.sliding_window_size until that gap is resolved in a future
  // ai-design pass.
  node->cache_manager_config.sliding_window_size = 0;

  node->runtime_config.parallel_basic_config.tensor_parallel_size = config_.tp_num;
  node->runtime_config.parallel_basic_config.attn_data_parallel_size = 1;
  node->runtime_config.parallel_basic_config.attn_tensor_parallel_size = config_.tp_num;
  node->runtime_config.enable_prefix_caching = node->cache_manager_config.enable_prefix_caching;

  node->block_allocator_group = std::make_shared<FakedBlockAllocatorGroup>(node->block_manager_config, config_.tp_num);
  node->cache_manager =
      CacheManagerFactory::CreateCacheManager(node->cache_manager_config, node->block_allocator_group);
  node->cache_manager->InitializeCachedBlocks();

  node->env_simulator = std::make_unique<BatchSchedulerEnvironmentSimulator>(node->block_manager_config, config_.tp_num,
                                                                             node->block_allocator_group);

  const size_t attn_dp_size = 1;
  auto eds =
      std::make_shared<EventDrivenScheduleProcessor>(attn_dp_size, node->batch_scheduler_config.max_pp_batch_num);
  eds->SetScheduleBackend(std::make_shared<ContinuousBatchingBackend>(node->batch_scheduler_config));
  eds->SetCacheManagers({node->cache_manager});
  auto hook_manager = std::make_shared<ScheduleHookManager>();

  PdV2ConnectorConfig connector_config;
  connector_config.deployment_role = DeploymentRole::kEngine;
  connector_config.node_role = role;
  connector_config.backend_type = "fake";
  connector_config.inference_addr = addr;
  connector_config.engine.local_complete_threshold = config_.local_complete_threshold;

  node->connector = std::make_shared<FakePdV2Connector>(registry_.get());
  node->connector->Initialize(connector_config);
  node->connector->RegisterKvPool(node->block_allocator_group.get(), /*total_bytes=*/0, /*block_size=*/0,
                                  device_block_num, /*num_layers=*/1);
  registry_->Register(addr, node->connector);

  if (role == NodeRole::kDecode) {
    node->global_cache_fetcher = std::make_shared<GlobalCacheFetcher>(static_cast<int>(config_.tp_num),
                                                                      /*ipc_func_wrapper=*/nullptr, "pd_v2_decode");
    node->global_cache_fetcher->Start();
    auto global_cache_hook =
        std::make_shared<GlobalCacheQueryHook>(node->global_cache_fetcher, node->batch_scheduler_config, /*hook_idx=*/0,
                                               /*defer_admission_on_block_full=*/true);
    hook_manager->AddHook(global_cache_hook);
    eds->SetPdV2DecodeMode(true);

    node->decode_hook =
        std::make_shared<PdV2DecodeHook>(node->connector, /*hook_idx=*/1, config_.local_complete_threshold);
    hook_manager->AddHook(node->decode_hook);
  } else {
    node->kv_bridge = std::make_shared<SimulatedKvWriteBridge>();
    PdV2PrefillIntakeAdapter::Deps intake_deps;
    intake_deps.prefill_eds = eds.get();
    intake_deps.prefill_env_simulator = node->env_simulator.get();
    intake_deps.prefill_connector = node->connector;
    intake_deps.kv_bridge = node->kv_bridge;
    intake_deps.block_token_num = config_.block_token_num;
    node->prefill_intake = std::make_shared<PdV2PrefillIntakeAdapter>(std::move(intake_deps));
    node->connector->RegisterIncomingHook(&PdV2PrefillIntakeAdapter::OnIncomingTrampoline, node->prefill_intake.get());
  }

  eds->SetScheduleHookManager(hook_manager);
  eds->SetWorkloadBalancer(std::make_shared<BasicWorkloadBalancer>(
      attn_dp_size, static_cast<size_t>(node->batch_scheduler_config.max_batch_size),
      static_cast<size_t>(node->batch_scheduler_config.max_decode_tokens_per_req),
      static_cast<size_t>(node->batch_scheduler_config.max_step_token_num)));
  eds->Initialize(nullptr);
  node->eds = eds;

  return node;
}

MultiNodeTopology::MultiNodeTopology(MultiNodeTopologyConfig config) : config_(std::move(config)) {
  Singleton<Environment>::GetInstance()->ParseConfig(GetTestConfigFile());

  registry_ = std::make_unique<PdV2FakeConnectorRegistry>(config_.registry_worker_num);

  for (int i = 0; i < config_.num_prefill; ++i) {
    const size_t block_num =
        config_.prefill_device_block_num > 0 ? config_.prefill_device_block_num : config_.device_block_num;
    prefill_nodes_.push_back(BuildNode(NodeRole::kPrefill, "prefill-" + std::to_string(i), block_num));
  }
  for (int i = 0; i < config_.num_decode; ++i) {
    decode_nodes_.push_back(BuildNode(NodeRole::kDecode, "decode-" + std::to_string(i), config_.device_block_num));
  }

  std::vector<std::string> prefill_addrs;
  prefill_addrs.reserve(prefill_nodes_.size());
  for (auto& n : prefill_nodes_) {
    prefill_addrs.push_back(n->inference_addr);
  }
  for (auto& n : decode_nodes_) {
    n->connector->SetPeerAddrs(prefill_addrs);
  }

  for (auto& n : prefill_nodes_) {
    n->eds->Start();
    n->StartDrivingLoop();
  }
  for (auto& n : decode_nodes_) {
    n->eds->Start();
    n->StartDrivingLoop();
  }
}

MultiNodeTopology::~MultiNodeTopology() { TearDown(); }

void MultiNodeTopology::TearDown() {
  if (torn_down_) {
    return;
  }
  torn_down_ = true;
  // Stop every EDS first so driving loops exit without scheduling new work,
  // then join driving threads (their last OnPrefillFinished → Post happens
  // before join returns), THEN drain the registry pool. Post() also guards
  // with stopped_ so a late callback cannot dereference a destroyed pool.
  for (auto& n : decode_nodes_) {
    n->StopEds();
  }
  for (auto& n : prefill_nodes_) {
    n->StopEds();
  }
  for (auto& n : decode_nodes_) {
    n->JoinDrivingThread();
  }
  for (auto& n : prefill_nodes_) {
    n->JoinDrivingThread();
  }
  if (registry_) {
    registry_->DrainAndStopThreadPool();
  }
}

}  // namespace pd_v2
}  // namespace ksana_llm
