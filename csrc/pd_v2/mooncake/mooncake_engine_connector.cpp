/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
// MooncakeEngineConnector: lifecycle (ctor/dtor/Initialize/Shutdown) and
// boot-time KV-buffer plumbing. The per-request Decode/Prefill paths and the
// notify poller / service-discovery threads live in the sibling
// mooncake_engine_connector_{decode,prefill,discovery}.cpp translation units.
#include "pd_v2/mooncake/mooncake_engine_connector.h"

#include <chrono>
#include <memory>
#include <thread>

#include "fmt/core.h"
#include "ipc/ipc_func_wrapper.h"
#include "pd_v2/mooncake/mooncake_engine_connector_internal.h"
#include "utils/logger.h"
#include "utils/ret_code.h"

namespace ksana_llm {
namespace pd_v2 {
namespace mooncake_backend {

namespace {

// Returns true if metadata_addr is a redis://... URL (the only scheme
// we plug hiredis into; etcd:// / http:// / P2PHANDSHAKE all skip Redis
// keepalive entirely).
bool IsRedisMetadata(const std::string& addr) { return addr.rfind("redis://", 0) == 0; }

}  // namespace

MooncakeEngineConnector::MooncakeEngineConnector() = default;

MooncakeEngineConnector::~MooncakeEngineConnector() {
  Shutdown().GetMessage();  // best-effort
}

Status MooncakeEngineConnector::Initialize(const PdV2ConnectorConfig& config) {
  if (config.deployment_role != DeploymentRole::kEngine) {
    return Status(RET_INVALID_ARGUMENT, "MooncakeEngineConnector requires deployment_role == kEngine");
  }
  config_ = config;

  tent_ = std::make_unique<EngineTentWrapper>();
  STATUS_CHECK_RETURN(tent_->Initialize(config.metadata_addr, config.inference_addr));

  peer_selector_ = std::make_unique<PeerSelector>(config.engine.selector_policy);

  // LinUCB LOCAL-vs-REMOTE router: only when opted in. PeerSelector keeps
  // serving kRoundRobin for the remote peer choice; the bandit only decides
  // whether to keep prefill local.
  if (config.engine.selector_policy == SelectorPolicy::kLinUCB) {
    LinUCBConfig lc;
    lc.alpha = config.engine.linucb_alpha;
    lc.tpot_target_us = config.engine.linucb_tpot_target_us;
    lc.tpot_safety_factor = config.engine.linucb_tpot_safety_factor;
    lc.ttft_threshold_us = config.engine.linucb_ttft_threshold_us;
    lc.running_ref = config.engine.linucb_running_ref;
    lc.remaining_ref_tokens = config.engine.linucb_remaining_ref_tokens;
    lc.remote_lat_ref_us = config.engine.linucb_remote_lat_ref_us;
    linucb_ = std::make_unique<LinUCBRouter>(lc);
    KLLM_LOG_INFO << "MooncakeEngineConnector: LinUCB router enabled (alpha=" << lc.alpha
                  << " tpot_target_us=" << lc.tpot_target_us << " tpot_safety_factor=" << lc.tpot_safety_factor
                  << " ttft_threshold_us=" << lc.ttft_threshold_us << " running_ref=" << lc.running_ref
                  << " remaining_ref_tokens=" << lc.remaining_ref_tokens << ")";
  }

  // Service discovery is Redis-backed (design §3.3): Decode self-registers
  // under `pd_v2/decode_alive/<cluster>/<addr>`, Prefill polls
  // `KEYS pd_v2/decode_alive/<cluster>/*`
  // every snapshot push period. Static prefill_peers seeding is gone — Decode
  // learns of Prefills via the first PrefillSnapshot push that arrives after
  // the Decode key shows up in Redis.

  poller_stop_.store(false);
  poller_ = std::thread(&MooncakeEngineConnector::NotifyPollerLoop, this);

  // §3.3 Redis-backed service discovery + keepalive. Connect a single
  // hiredis client per process when metadata_addr is redis://. Used by:
  //   - Decode: keepalive thread publishes pd_v2/decode_alive/<cluster>/<addr>
  //   - Prefill: heartbeat thread polls KEYS pd_v2/decode_alive/<cluster>/*
  // metadata schemes other than redis:// (P2PHANDSHAKE, etcd://, http://)
  // skip Redis. In that case Prefill heartbeat falls back to the S1
  // incoming_routes_ iteration and Decode never registers.
  if (IsRedisMetadata(config.metadata_addr)) {
    redis_alive_ = std::make_unique<RedisAliveStore>();
    // redis_password is forwarded into MC_REDIS_PASSWORD env at engine
    // bootstrap (see inference_engine.cpp). Pass empty string here and
    // let RedisAliveStore::Connect read the env, so we don't need a new
    // config touchpoint.
    Status rs = redis_alive_->Connect(config.metadata_addr, /*password=*/"");
    if (!rs.OK()) {
      KLLM_LOG_WARNING << "MooncakeEngineConnector: Redis connect failed (" << rs.GetMessage()
                       << ") — service discovery degraded "
                       << "(Decode won't register, Prefill heartbeat falls back "
                       << "to incoming_routes_ iteration)";
      redis_alive_.reset();
    }
  } else if (config.node_role == NodeRole::kPrefill || config.node_role == NodeRole::kDecode) {
    KLLM_LOG_INFO << "MooncakeEngineConnector: metadata_addr='" << config.metadata_addr
                  << "' is not redis://, S3 Redis service discovery disabled";
  }

  // Decode role: spawn the keepalive thread when Redis is connected.
  // boot_epoch_ms_ is captured here (system_clock now in ms) and reused
  // verbatim throughout this process's lifetime — Prefill compares it
  // against cached epoch to detect restart.
  if (config.node_role == NodeRole::kDecode && redis_alive_ != nullptr) {
    boot_epoch_ms_ = NowMs();
    keepalive_stop_.store(false);
    keepalive_ = std::thread(&MooncakeEngineConnector::RedisKeepaliveLoop, this);
    KLLM_LOG_INFO << "MooncakeEngineConnector: Decode keepalive started, epoch=" << boot_epoch_ms_;
  }

  // Decode role: allocate a registered host write-side scratch arena for serialized PrefillStagePayload writes.
  // Mooncake WRITE sources must live in registered memory, so heap buffers like std::vector are rejected.
  // This buffer is outgoing only; 1 GiB covers thousands of concurrent stage payloads with room for future growth.
  // max_peers=1 gives us one flat HostStagePool range managed by a single PeerCredits.
  if (config.node_role == NodeRole::kDecode) {
    constexpr size_t kTxPoolBytesDefault = 1ull << 30;  // 1 GiB
    tx_pool_ = std::make_unique<HostStagePool>();
    Status sp = tx_pool_->Init(kTxPoolBytesDefault, /*max_peers=*/1);
    if (!sp.OK()) {
      return Status(sp.GetCode(), fmt::format("HostStagePool::Init(write-side, {} bytes) failed: {}",
                                              kTxPoolBytesDefault, sp.GetMessage()));
    }
    Status rs = tent_->RegisterStageMemory(tx_pool_->base(), tx_pool_->total_bytes());
    if (!rs.OK()) {
      return Status(rs.GetCode(),
                    fmt::format("EngineTent::RegisterStageMemory(write-side) failed: {}", rs.GetMessage()));
    }
    tx_credits_ = std::make_unique<PeerCredits>();
    Status ts = tx_credits_->Init(/*partition_base_slot=*/0,
                                  /*partition_n_slots=*/static_cast<uint32_t>(tx_pool_->total_slots()));
    if (!ts.OK()) {
      return Status(ts.GetCode(), fmt::format("write-side PeerCredits::Init({} slots) failed: {}",
                                              tx_pool_->total_slots(), ts.GetMessage()));
    }
    KLLM_LOG_INFO << "MooncakeEngineConnector(Decode): write-side scratch ready base=" << tx_pool_->base()
                  << " bytes=" << tx_pool_->total_bytes() << " slots=" << tx_pool_->total_slots();

    // Async submit worker pool: the SubmitPrefillRequest stage handshake
    // (acquire slots + RDMA-WRITE drain) runs here, OFF the EDS scheduling
    // thread, so a stuck / full / slow peer never stalls Decode
    // scheduling. Each worker handles one request end-to-end.
    const uint32_t n_workers =
        config.engine.control_stage_async_workers == 0 ? 1u : config.engine.control_stage_async_workers;
    submit_stop_.store(false);
    submit_workers_.reserve(n_workers);
    for (uint32_t i = 0; i < n_workers; ++i) {
      submit_workers_.emplace_back(&MooncakeEngineConnector::SubmitWorkerLoop, this);
    }
    KLLM_LOG_INFO << "MooncakeEngineConnector(Decode): async submit worker pool started, threads=" << n_workers;
  }

  // Prefill role: spawn the heartbeat broadcaster. Decode side has
  // nothing to advertise (it's the consumer of PrefillSnapshot), so
  // skip on Decode.
  if (config.node_role == NodeRole::kPrefill) {
    // Allocate the control-plane HostStagePool that Decode peers
    // RDMA-WRITE the heavy prefill_req payload into. Sized from
    // EngineConfig.control_stage_*; defaults are baked into HostStagePool
    // when control_stage_pool_bytes == 0. Memory is registered with the
    // engine's tent so the RDMA transport advertises it as remote-
    // accessible. Failure here is fatal — without a stage pool, the
    // control path can't service 32K+ token prefills (its raison d'être).
    stage_pool_ = std::make_unique<HostStagePool>();
    const size_t pool_bytes = config.engine.control_stage_pool_bytes == 0
                                  ? HostStagePool::kDefaultPoolBytes
                                  : static_cast<size_t>(config.engine.control_stage_pool_bytes);
    const size_t max_peers = config.engine.control_stage_max_peers == 0 ? 128u : config.engine.control_stage_max_peers;
    Status sp = stage_pool_->Init(pool_bytes, max_peers);
    if (!sp.OK()) {
      return Status(sp.GetCode(), fmt::format("HostStagePool::Init({} bytes, {} peers) failed: {}", pool_bytes,
                                              max_peers, sp.GetMessage()));
    }
    Status rs = tent_->RegisterStageMemory(stage_pool_->base(), stage_pool_->total_bytes());
    if (!rs.OK()) {
      return Status(rs.GetCode(), fmt::format("EngineTent::RegisterStageMemory failed: {}", rs.GetMessage()));
    }
    KLLM_LOG_INFO << "MooncakeEngineConnector(Prefill): HostStagePool ready base=" << stage_pool_->base()
                  << " bytes=" << stage_pool_->total_bytes() << " slots=" << stage_pool_->total_slots()
                  << " slots_per_peer=" << stage_pool_->slots_per_peer();

    heartbeat_stop_.store(false);
    heartbeat_ = std::thread(&MooncakeEngineConnector::HeartbeatLoop, this);
  }

  KLLM_LOG_INFO << "MooncakeEngineConnector initialized inference_addr=" << config.inference_addr
                << " role=" << static_cast<int>(config.node_role);
  return Status();
}

Status MooncakeEngineConnector::Shutdown() {
  if (tent_ == nullptr) return Status();

  // Stop the async submit workers first. notify_all wakes idle workers;
  // workers parked in a blocking AcquireSlots are woken further down by
  // PeerCredits::Shutdown (before the join) so this can't hang on a
  // stuck/full peer.
  submit_stop_.store(true);
  submit_cv_.notify_all();

  poller_stop_.store(true);
  if (poller_.joinable()) poller_.join();
  heartbeat_stop_.store(true);
  if (heartbeat_.joinable()) heartbeat_.join();
  keepalive_stop_.store(true);
  if (keepalive_.joinable()) keepalive_.join();

  // Prefill side: BroadcastSnapshot never retains engine-side
  // SegmentHandles (it open+close immediately), so there is nothing to
  // close here — just drop the epoch tracking map.
  decode_epoch_cache_.clear();

  if (redis_alive_ != nullptr) {
    redis_alive_->Disconnect();
    redis_alive_.reset();
  }

  // Tear down stage pool (Prefill role) and PeerCredits rings (Decode
  // role). HostStagePool destructor unmlocks + munmaps; PeerCredits
  // shutdown unblocks any threads waiting on AcquireSlots.
  //
  // Order matters: Shutdown the credits rings FIRST (unblocks any submit
  // worker parked in a blocking AcquireSlots), THEN join the workers,
  // THEN drop the maps / pools. Workers hold shared_ptr copies of the
  // rings, so they stay alive until the worker's ProcessSubmit returns
  // even though we clear stage_peers_ below.
  {
    std::lock_guard<std::mutex> lk(stage_peers_mu_);
    for (auto& kv : stage_peers_) {
      if (kv.second.credits) kv.second.credits->Shutdown();
    }
  }
  // Decode-side write scratch: Shutdown unblocks any in-flight AcquireSlots
  // so a worker parked on the write-side credit ring doesn't wedge the join below.
  if (tx_credits_) tx_credits_->Shutdown();

  // Now that both credit rings fail fast, join the submit workers. Each
  // worker's ProcessSubmit either completes or fails (firing on_complete
  // with kFailed, erasing its in_flight_ entry) and then observes
  // submit_stop_ and exits.
  for (auto& t : submit_workers_) {
    if (t.joinable()) t.join();
  }
  submit_workers_.clear();
  {
    std::lock_guard<std::mutex> lk(submit_mu_);
    submit_queue_.clear();
  }

  {
    std::lock_guard<std::mutex> lk(stage_peers_mu_);
    stage_peers_.clear();
  }
  stage_pool_.reset();
  tx_credits_.reset();
  tx_pool_.reset();

  // Snapshot the still-pending on_complete callbacks before dropping in_flight_.
  // Each surviving entry has a Decode hook parked in kWaitAsyncResult waiting on
  // OnPrefillComplete; clearing the map without firing them would strand those
  // requests forever. Fire kCancelled so the hook can unwind (fail + end the
  // req). Invoked AFTER releasing state_mu_ — matches OnPrefillComplete's
  // snapshot-under-lock / invoke-outside pattern and keeps the hook's mutex from
  // nesting under state_mu_.
  struct PendingCancel {
    uint64_t req_id;
    PdRequestCompleteCb on_complete;
    void* on_complete_ctx;
  };
  std::vector<PendingCancel> to_cancel;
  {
    std::lock_guard<std::mutex> lk(state_mu_);
    to_cancel.reserve(in_flight_.size());
    for (auto& kv : in_flight_) {
      if (kv.second.on_complete != nullptr) {
        to_cancel.push_back({kv.first, kv.second.on_complete, kv.second.on_complete_ctx});
      }
    }
    in_flight_.clear();
    incoming_routes_.clear();
    on_incoming_ = nullptr;
    on_incoming_ctx_ = nullptr;
  }
  for (const auto& pc : to_cancel) {
    pc.on_complete(pc.req_id, PdCompleteStatus::kCancelled, /*generated_tokens=*/{}, /*prefill_metrics=*/{},
                   /*prefill_peer=*/"", pc.on_complete_ctx);
  }
  Status s = tent_->Shutdown();
  tent_.reset();
  peer_selector_.reset();
  return s;
}

// ===== Boot-time =====

Status MooncakeEngineConnector::RegisterKvPool(void* /*kv_pool_base*/, size_t /*total_bytes*/, size_t /*block_size*/,
                                               size_t /*num_blocks*/, size_t /*num_layers*/) {
  return Status(RET_INVALID_ARGUMENT, "RegisterKvPool is executor-side only; called on engine connector");
}

Status MooncakeEngineConnector::GetKvBufferVa(int rank, uint64_t* out_kv_pool_base_va, uint64_t* out_total_bytes) {
  if (out_kv_pool_base_va == nullptr || out_total_bytes == nullptr) {
    return Status(RET_INVALID_ARGUMENT, "GetKvBufferVa: null out param");
  }
  // Cache hit fast path.
  {
    std::lock_guard<std::mutex> lk(kv_va_cache_mu_);
    auto it = rank_kv_va_cache_.find(rank);
    if (it != rank_kv_va_cache_.end()) {
      *out_kv_pool_base_va = it->second.kv_pool_base_va;
      *out_total_bytes = it->second.total_bytes;
      return Status();
    }
  }

  if (ipc_func_wrapper_ == nullptr) {
    return Status(RET_INVALID_ARGUMENT,
                  "MooncakeEngineConnector::GetKvBufferVa: IpcFuncWrapper not bound — "
                  "call SetIpcFuncWrapper at boot");
  }
  uint64_t base_va = 0, total = 0, block_size = 0, num_blocks = 0, num_layers = 0;
  // First-attempt may race with executor RegisterKvPool during boot
  // (engine handles its first decode request before the executor has
  // finished registering the contiguous KV pool). Retry once with a
  // short backoff before failing the request — under steady state both
  // calls hit the cache and never enter this path.
  Status s = ipc_func_wrapper_->PdV2GetKvPoolBaseVa(rank, &base_va, &total, &block_size, &num_blocks, &num_layers);
  if (!s.OK()) {
    KLLM_LOG_WARNING << "GetKvBufferVa first attempt failed for rank=" << rank << ": " << s.GetMessage()
                     << "; retrying in 50ms";
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    s = ipc_func_wrapper_->PdV2GetKvPoolBaseVa(rank, &base_va, &total, &block_size, &num_blocks, &num_layers);
    if (!s.OK()) {
      return s;
    }
  }
  {
    std::lock_guard<std::mutex> lk(kv_va_cache_mu_);
    rank_kv_va_cache_[rank] = {base_va, total};
  }
  *out_kv_pool_base_va = base_va;
  *out_total_bytes = total;
  KLLM_LOG_INFO << "MooncakeEngineConnector::GetKvBufferVa rank=" << rank << " base_va=0x" << std::hex << base_va
                << std::dec << " total=" << total << " block_size=" << block_size << " num_blocks=" << num_blocks
                << " num_layers=" << num_layers;
  return Status();
}

void MooncakeEngineConnector::SetIpcFuncWrapper(IpcFuncWrapper* ipc_func_wrapper) {
  ipc_func_wrapper_ = ipc_func_wrapper;
}

void MooncakeEngineConnector::SetSnapshotProvider(std::function<PeerHealth()> provider) {
  snapshot_provider_ = std::move(provider);
}

}  // namespace mooncake_backend
}  // namespace pd_v2
}  // namespace ksana_llm
