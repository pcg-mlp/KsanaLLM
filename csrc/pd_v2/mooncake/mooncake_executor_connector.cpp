/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "pd_v2/mooncake/mooncake_executor_connector.h"

#include <cuda_runtime.h>
#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <set>
#include <thread>

#include "device/device_utils.h"

#include "fmt/core.h"
#include "ipc/ipc_func_wrapper.h"
#include "transfer_engine.h"
#include "utils/logger.h"
#include "utils/ret_code.h"

namespace ksana_llm {
namespace pd_v2 {
namespace mooncake_backend {

namespace {
constexpr const char* kInvalidEngineSide = "method is engine-only; called on executor connector";

// Service-discovery key namespace, scoped by cluster_name so multiple
// independent pd_v2 clusters sharing one Redis don't discover each
// other's Decodes (and openSegment their .exec.r{N} cross-cluster).
// Must match MooncakeEngineConnector::AliveKeyPrefix exactly. Empty
// cluster_name falls back to "default" (matches python request_id.py).
std::string AliveKeyPrefix(const std::string& cluster_name) {
  return "pd_v2/decode_alive/" + (cluster_name.empty() ? std::string("default") : cluster_name) + "/";
}

// Poll interval for the executor-side epoch tracker. Engine polls at
// `snapshot_push_interval_ms` (default 1s); executor uses the same
// cadence by default — Decode restart latency to data-plane refresh is
// bounded by 2 * poll_interval (one cycle to notice + one to close+open).
constexpr uint32_t kExecutorEpochPollMs = 1000;

// Per Decode node we don't know its `attn_tp_decode` from the alive
// value (Decode's keepalive doesn't publish it). Probe up to this many
// `.exec.r{N}` segments per addr on epoch change; closeSegment on a
// segment we never opened is a cheap no-op in Mooncake.
constexpr uint32_t kMaxProbeRanksPerPeer = 16;

bool IsRedisMetadata(const std::string& addr) { return addr.rfind("redis://", 0) == 0; }

uint64_t ParseEpochFromJson(const std::string& v) {
  size_t k = v.find("\"epoch\"");
  if (k == std::string::npos) return 0;
  k = v.find(':', k);
  if (k == std::string::npos) return 0;
  ++k;
  while (k < v.size() && (v[k] == ' ' || v[k] == '\t' || v[k] == '"')) ++k;
  uint64_t epoch = 0;
  bool any = false;
  while (k < v.size() && v[k] >= '0' && v[k] <= '9') {
    epoch = epoch * 10 + static_cast<uint64_t>(v[k] - '0');
    ++k;
    any = true;
  }
  return any ? epoch : 0;
}
}  // namespace

MooncakeExecutorConnector::MooncakeExecutorConnector() = default;

MooncakeExecutorConnector::~MooncakeExecutorConnector() {
  Shutdown().GetMessage();  // best-effort
}

Status MooncakeExecutorConnector::Initialize(const PdV2ConnectorConfig& config) {
  KLLM_LOG_INFO << "MooncakeExecutorConnector::Initialize enter";
  if (config.deployment_role != DeploymentRole::kExecutor) {
    return Status(RET_INVALID_ARGUMENT, "MooncakeExecutorConnector requires deployment_role == kExecutor");
  }
  {
    KLLM_LOG_DEBUG << "MooncakeExecutorConnector config sizes:"
                   << " backend_type=" << config.backend_type.size() << " cluster_name=" << config.cluster_name.size()
                   << " inference_addr=" << config.inference_addr.size() << " router_addr=" << config.router_addr.size()
                   << " metadata_addr=" << config.metadata_addr.size()
                   << " protocol=" << config.executor.protocol.size()
                   << " nic_json=" << config.executor.rdma_nic_priority_matrix_json.size();
    struct rlimit rl{};
    if (getrlimit(RLIMIT_AS, &rl) == 0) {
      KLLM_LOG_DEBUG << "MooncakeExecutorConnector rlimit AS cur=" << static_cast<int64_t>(rl.rlim_cur)
                     << " max=" << static_cast<int64_t>(rl.rlim_max);
    }
    if (getrlimit(RLIMIT_DATA, &rl) == 0) {
      KLLM_LOG_DEBUG << "MooncakeExecutorConnector rlimit DATA cur=" << static_cast<int64_t>(rl.rlim_cur)
                     << " max=" << static_cast<int64_t>(rl.rlim_max);
    }
  }
  try {
    config_ = config;
  } catch (const std::exception& e) {
    KLLM_LOG_ERROR << "MooncakeExecutorConnector: config copy threw: " << e.what();
    return Status(RET_RUNTIME_FAILED, std::string("config copy threw: ") + e.what());
  }
  try {
    tent_ = std::make_unique<ExecutorTentWrapper>();
  } catch (const std::exception& e) {
    KLLM_LOG_ERROR << "MooncakeExecutorConnector: tent make_unique threw: " << e.what();
    return Status(RET_RUNTIME_FAILED, std::string("tent make_unique threw: ") + e.what());
  }
  // Engine and executor coexist on the same node and would otherwise
  // both try to register the same Mooncake server_name
  // (config.inference_addr), colliding in the Redis metadata store.
  // Suffix the executor's name so peers can address them independently:
  //  - engine:    <inference_addr>                  (control-plane notifies)
  //  - executor r: <inference_addr>.exec.r<rank>    (KV WRITE target,
  //                                                  per-TP-rank distinct)
  // The per-rank suffix is required under TP > 1 because every executor
  // process on this Decode node owns a different KV pool VA range. If
  // they all registered under the same name, only the last writer to
  // Redis would be visible, and peers resolving target_offset against
  // that stale base_va would miss findBuffer().
  const std::string executor_server_name =
      config.inference_addr + ".exec.r" + std::to_string(config.executor.executor_rank);
  STATUS_CHECK_RETURN(tent_->Initialize(config.metadata_addr, executor_server_name, config.executor.rpc_port));
  KLLM_LOG_INFO << "MooncakeExecutorConnector initialized inference_addr=" << config.inference_addr
                << " segment_name=" << executor_server_name << " rpc_port=" << config.executor.rpc_port;

  // §3.3 executor-side epoch tracker. Independent of engine — each
  // executor process opens its own Redis connection and polls. The
  // engine TransferEngine's segment cache and the executor
  // TransferEngine's segment cache are in different processes; engine's
  // close+open does not invalidate executor's. We need our own loop.
  if (IsRedisMetadata(config.metadata_addr)) {
    redis_alive_ = std::make_unique<RedisAliveStore>();
    Status rs = redis_alive_->Connect(config.metadata_addr, /*password=*/"");
    if (!rs.OK()) {
      KLLM_LOG_WARNING << "MooncakeExecutorConnector: Redis connect failed (" << rs.GetMessage()
                       << ") — executor epoch tracker disabled. "
                       << "Decode restart will not auto-invalidate this executor's "
                       << "Mooncake segment cache; manual restart needed.";
      redis_alive_.reset();
    } else {
      epoch_poll_stop_.store(false);
      epoch_poll_ = std::thread(&MooncakeExecutorConnector::EpochPollLoop, this);
      KLLM_LOG_INFO << "MooncakeExecutorConnector: epoch poll thread started "
                    << "(rank=" << config.executor.executor_rank << ")";
    }
  }
  return Status();
}

Status MooncakeExecutorConnector::Shutdown() {
  if (tent_ == nullptr) return Status();
  epoch_poll_stop_.store(true);
  if (epoch_poll_.joinable()) epoch_poll_.join();
  if (redis_alive_ != nullptr) {
    redis_alive_->Disconnect();
    redis_alive_.reset();
  }
  decode_epoch_cache_.clear();

  Status s = tent_->Shutdown();
  tent_.reset();
  std::lock_guard<std::mutex> lk(kv_state_mu_);
  kv_pool_registered_ = false;
  return s;
}

void MooncakeExecutorConnector::SetIpcFuncWrapper(IpcFuncWrapper* ipc_func_wrapper) {
  if (ipc_func_wrapper == nullptr) return;
  // Bind a type-erased query fn so ipc/ doesn't depend on the mooncake
  // backend at link time. Capturing `this` is safe — the connector
  // outlives the IpcFuncWrapper (both owned by InferenceExecutor and
  // torn down LIFO in its destructor).
  ipc_func_wrapper->SetPdV2KvPoolQueryFn([this](uint64_t* base_va, uint64_t* total_bytes, uint64_t* block_size,
                                                uint64_t* num_blocks, uint64_t* num_layers) -> bool {
    size_t bs = 0, nb = 0, nl = 0;
    bool ok = this->GetLocalKvPool(base_va, total_bytes, &bs, &nb, &nl);
    if (block_size) *block_size = bs;
    if (num_blocks) *num_blocks = nb;
    if (num_layers) *num_layers = nl;
    return ok;
  });
}

Status MooncakeExecutorConnector::RegisterKvPool(void* kv_pool_base, size_t total_bytes, size_t block_size,
                                                 size_t num_blocks, size_t num_layers) {
  if (tent_ == nullptr) {
    return Status(RET_INVALID_ARGUMENT, "RegisterKvPool before Initialize");
  }
  if (kv_pool_base == nullptr || total_bytes == 0) {
    return Status(RET_INVALID_ARGUMENT, "RegisterKvPool: invalid args");
  }
  // location 必须匹配 KV pool 所在的真实 CUDA 设备。每个 executor rank 绑定一个 logical device，
  // Mooncake 接收侧会按该 location 执行 cudaMemcpy。
  const std::string location = "cuda:" + std::to_string(config_.executor.executor_rank);
  STATUS_CHECK_RETURN(tent_->RegisterKvPool(kv_pool_base, total_bytes, location));

  std::lock_guard<std::mutex> lk(kv_state_mu_);
  kv_pool_base_va_ = reinterpret_cast<uint64_t>(kv_pool_base);
  kv_pool_total_bytes_ = total_bytes;
  block_size_ = block_size;
  num_blocks_ = num_blocks;
  num_layers_ = num_layers;
  kv_pool_registered_ = true;
  KLLM_LOG_INFO << "MooncakeExecutorConnector::RegisterKvPool stored base_va=0x" << std::hex << kv_pool_base_va_
                << std::dec << " bytes=" << total_bytes << " block_size=" << block_size << " num_blocks=" << num_blocks
                << " num_layers=" << num_layers;
  return Status();
}

bool MooncakeExecutorConnector::GetLocalKvPool(uint64_t* out_kv_pool_base_va, uint64_t* out_total_bytes,
                                               size_t* out_block_size, size_t* out_num_blocks,
                                               size_t* out_num_layers) const {
  std::lock_guard<std::mutex> lk(kv_state_mu_);
  if (!kv_pool_registered_) return false;
  if (out_kv_pool_base_va) *out_kv_pool_base_va = kv_pool_base_va_;
  if (out_total_bytes) *out_total_bytes = kv_pool_total_bytes_;
  if (out_block_size) *out_block_size = block_size_;
  if (out_num_blocks) *out_num_blocks = num_blocks_;
  if (out_num_layers) *out_num_layers = num_layers_;
  return true;
}

bool MooncakeExecutorConnector::GetLocalKvPoolShape(uint64_t* out_kv_pool_base_va, uint64_t* out_total_bytes,
                                                    size_t* out_block_size, size_t* out_num_blocks,
                                                    size_t* out_num_layers) const {
  return GetLocalKvPool(out_kv_pool_base_va, out_total_bytes, out_block_size, out_num_blocks, out_num_layers);
}

Status MooncakeExecutorConnector::GetKvBufferVa(int /*rank*/, uint64_t* /*out_kv_pool_base_va*/,
                                                uint64_t* /*out_total_bytes*/) {
  return Status(RET_INVALID_ARGUMENT, kInvalidEngineSide);
}

Status MooncakeExecutorConnector::SubmitPrefillRequest(uint64_t /*req_id*/, const std::vector<int>& /*input_tokens*/,
                                                       const std::vector<int>& /*kv_pool_block_ids*/,
                                                       const SamplingConfigWire& /*sampling_config*/,
                                                       uint32_t /*attn_dp_group_id*/, uint32_t /*cached_block_count*/,
                                                       uint32_t /*decode_prefix_cache_offset*/,
                                                       PdRequestCompleteCb /*on_complete*/, void* /*ctx*/) {
  return Status(RET_INVALID_ARGUMENT, kInvalidEngineSide);
}

Status MooncakeExecutorConnector::Release(uint64_t /*req_id*/) {
  return Status(RET_INVALID_ARGUMENT, kInvalidEngineSide);
}

Status MooncakeExecutorConnector::SendPrefillComplete(const std::string& /*decode_engine_inference_addr*/,
                                                      uint64_t /*req_id*/, PdCompleteStatus /*status*/,
                                                      const std::vector<int>& /*generated_tokens*/,
                                                      const PeerHealth& /*snapshot*/,
                                                      const PdPrefillMetrics& /*prefill_metrics*/) {
  return Status(RET_INVALID_ARGUMENT, kInvalidEngineSide);
}

void MooncakeExecutorConnector::TrackInflightBatch(uint64_t req_id, uint64_t batch_id) {
  std::lock_guard<std::mutex> lk(inflight_mu_);
  inflight_batches_[req_id].push_back(batch_id);
}

void MooncakeExecutorConnector::EraseInflightBatch(uint64_t req_id, uint64_t batch_id) {
  // Free Mooncake's per-batch resources. Each submitTransfer batch
  // backs a SubBatch object which (in tent's CUDA path) owns its own
  // CUDA stream allocated via cudaStreamCreateWithFlags. Without
  // freeing, every per-layer KV write leaks a stream; under the §6
  // routing where all attn_TP Prefill ranks send per request × N
  // layers × M warmup requests, the CUDA stream pool exhausts within
  // seconds and Mooncake reports
  //   "Failed to allocate SubBatch: cudaStreamCreateWithFlags ... out of memory"
  // which surfaces as PrefillRequestComplete{status=kFailed} on the
  // Decode side. Free here, before dropping the tracking entry, so
  // the barrier's COMPLETED → erase path always releases the stream.
  if (tent_ != nullptr) {
    Status fs = tent_->FreeBatchID(batch_id);
    if (!fs.OK()) {
      KLLM_LOG_WARNING << "MooncakeExecutorConnector::EraseInflightBatch FreeBatchID(" << batch_id
                       << ") failed: " << fs.GetMessage();
    }
  }
  std::lock_guard<std::mutex> lk(inflight_mu_);
  auto it = inflight_batches_.find(req_id);
  if (it == inflight_batches_.end()) return;
  auto& v = it->second;
  v.erase(std::remove(v.begin(), v.end(), batch_id), v.end());
  if (v.empty()) inflight_batches_.erase(it);
}

Status MooncakeExecutorConnector::SubmitWriteAsyncForRequest(const std::string& target_server_name, uint64_t target_va,
                                                             void* source, size_t length, uint64_t req_id,
                                                             uint64_t* out_batch_id) {
  if (out_batch_id) *out_batch_id = 0;
  if (tent_ == nullptr) {
    return Status(RET_INVALID_ARGUMENT, "SubmitWriteAsyncForRequest before Initialize");
  }
  // target_server_name MUST already include the per-rank suffix
  // (".exec.r<N>") set by the Prefill-side dispatcher. The executor
  // connector registers its KV pool under that name in Initialize();
  // peers must address the same name back. Caller responsibility, no
  // re-suffixing here — see pd_v2_kv_dispatcher.cpp.
  uint64_t batch_id = 0;
  Status s = tent_->SubmitWriteAsync(target_server_name, target_va, source, length, &batch_id);
  if (!s.OK()) return s;
  TrackInflightBatch(req_id, batch_id);
  if (out_batch_id) *out_batch_id = batch_id;
  return Status();
}

Status MooncakeExecutorConnector::SubmitWriteBatchAsyncForRequest(const std::string& target_server_name,
                                                                  const std::vector<KvWriteEntry>& entries,
                                                                  uint64_t req_id, uint64_t* out_batch_id) {
  if (out_batch_id) *out_batch_id = 0;
  if (tent_ == nullptr) {
    return Status(RET_INVALID_ARGUMENT, "SubmitWriteBatchAsyncForRequest before Initialize");
  }
  if (entries.empty()) {
    // Nothing to write — caller treats this as "no batch tracked".
    return Status();
  }
  std::vector<ExecutorTentWrapper::BatchWriteEntry> tent_entries;
  tent_entries.reserve(entries.size());
  for (const auto& e : entries) {
    tent_entries.push_back({e.target_va, e.source, e.length});
  }
  uint64_t batch_id = 0;
  Status s = tent_->SubmitWriteBatchAsync(target_server_name, tent_entries, &batch_id);
  if (!s.OK()) return s;
  TrackInflightBatch(req_id, batch_id);
  if (out_batch_id) *out_batch_id = batch_id;
  return Status();
}

std::vector<uint64_t> MooncakeExecutorConnector::GetInflightBatchIds(uint64_t req_id) const {
  std::lock_guard<std::mutex> lk(inflight_mu_);
  auto it = inflight_batches_.find(req_id);
  if (it == inflight_batches_.end()) return {};
  return it->second;
}

Status MooncakeExecutorConnector::PollBatchStatus(uint64_t batch_id, bool* out_completed, bool* out_failed) {
  if (tent_ == nullptr) {
    return Status(RET_INVALID_ARGUMENT, "PollBatchStatus before Initialize");
  }
  return tent_->GetBatchStatus(batch_id, out_completed, out_failed);
}

void MooncakeExecutorConnector::EpochPollLoop() {
  if (redis_alive_ == nullptr) return;
  // 轮询线程需要切到 executor 的 logical device，确保从本 rank KV pool 做 cudaMemcpy 时设备上下文正确。
  SetDevice(config_.executor.executor_rank);
  while (!epoch_poll_stop_.load(std::memory_order_acquire)) {
    std::vector<std::string> alive_keys;
    const std::string alive_prefix = AliveKeyPrefix(config_.cluster_name);
    Status ks = redis_alive_->Keys(alive_prefix + "*", &alive_keys);
    if (ks.OK()) {
      std::set<std::string> current_addrs;
      const size_t kPrefixLen = alive_prefix.size();
      for (const auto& key : alive_keys) {
        if (key.size() <= kPrefixLen) continue;
        if (key.compare(0, kPrefixLen, alive_prefix) != 0) continue;
        std::string addr = key.substr(kPrefixLen);
        if (addr.empty()) continue;
        std::string value;
        if (!redis_alive_->Get(key, &value).OK() || value.empty()) continue;
        uint64_t new_epoch = ParseEpochFromJson(value);
        if (new_epoch == 0) continue;
        current_addrs.insert(addr);

        auto it = decode_epoch_cache_.find(addr);
        if (it == decode_epoch_cache_.end()) {
          // First time we see this Decode — record and move on. No
          // segments to invalidate yet (executor hasn't transferred to
          // it). If it had been alive earlier in this executor's life
          // we would have already cached it.
          decode_epoch_cache_[addr] = new_epoch;
          KLLM_LOG_INFO << "MooncakeExecutorConnector::EpochPollLoop discovered "
                        << "Decode=" << addr << " epoch=" << new_epoch << " (rank=" << config_.executor.executor_rank
                        << ")";
        } else if (it->second != new_epoch) {
          KLLM_LOG_INFO << "MooncakeExecutorConnector::EpochPollLoop epoch mismatch "
                        << "Decode=" << addr << " old=" << it->second << " new=" << new_epoch
                        << " — invalidating cached segments";
          InvalidatePeerSegments(addr);
          it->second = new_epoch;
        }
      }
      // Decode that disappeared from Redis: also invalidate so a stale
      // SubmitWrite doesn't reuse it (the in-flight batch already
      // submitted will complete or fail on its own; this prevents the
      // NEXT write).
      for (auto it = decode_epoch_cache_.begin(); it != decode_epoch_cache_.end();) {
        if (current_addrs.count(it->first) == 0) {
          KLLM_LOG_INFO << "MooncakeExecutorConnector::EpochPollLoop Decode " << it->first
                        << " disappeared, invalidating cached segments";
          InvalidatePeerSegments(it->first);
          it = decode_epoch_cache_.erase(it);
        } else {
          ++it;
        }
      }
    } else {
      KLLM_LOG_WARNING << "MooncakeExecutorConnector::EpochPollLoop: Redis KEYS failed: " << ks.GetMessage();
    }

    // Sleep in 100ms slices for prompt shutdown.
    for (uint32_t slept = 0; slept < kExecutorEpochPollMs && !epoch_poll_stop_.load(std::memory_order_acquire);
         slept += 100) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }
}

void MooncakeExecutorConnector::InvalidatePeerSegments(const std::string& decode_addr) {
  if (tent_ == nullptr || tent_->raw() == nullptr) return;
  // We don't know Decode's actual attn_tp, so probe up to a bounded N
  // (kMaxProbeRanksPerPeer). For each `<addr>.exec.r{N}` the dance
  // openSegment+closeSegment forces a refresh:
  //   - if Mooncake had it cached → closeSegment drops it; next
  //     SubmitWrite re-fetches fresh SegmentDesc from Redis
  //   - if Mooncake didn't (rank doesn't exist, or never written here
  //     yet) → openSegment may succeed (Mooncake fetches metadata
  //     anyway since Redis has it for live ranks) or fail (rank gone);
  //     either way closeSegment is harmless
  // Cost: 16 RPC pairs on an epoch change (rare). Bounded and amortized.
  uint32_t closed = 0;
  for (uint32_t r = 0; r < kMaxProbeRanksPerPeer; ++r) {
    const std::string seg_name = decode_addr + ".exec.r" + std::to_string(r);
    mooncake::SegmentHandle h = tent_->raw()->openSegment(seg_name);
    if (static_cast<int64_t>(h) < 0) {
      // No such rank registered in metadata — stop probing higher ranks.
      break;
    }
    tent_->raw()->closeSegment(h);
    ++closed;
  }
  KLLM_LOG_INFO << "MooncakeExecutorConnector::InvalidatePeerSegments addr=" << decode_addr
                << " probed_ranks=" << closed << " (executor_rank=" << config_.executor.executor_rank << ")";
}

}  // namespace mooncake_backend
}  // namespace pd_v2
}  // namespace ksana_llm
