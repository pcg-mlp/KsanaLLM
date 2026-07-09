/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "pd_v2/mooncake/engine_tent_wrapper.h"

#include "base64.hpp"
#include "common/base/status.h"
#include "fmt/core.h"
#include "nlohmann/json.hpp"
#include "transfer_engine.h"
#include "transfer_metadata.h"
#include "transport/transport.h"
#include "utils/logger.h"
#include "utils/ret_code.h"

namespace ksana_llm {
namespace pd_v2 {
namespace mooncake_backend {

namespace {

// Engine TransferEngine ports start here. Mooncake's metadata server
// gates server_name uniqueness, but the local rpc_port is still bound;
// engine and executor must use distinct ports within the same process.
// 12345 is the upstream default but we move engine off it so executor
// (rare same-process case) can keep the default.
constexpr uint64_t kEngineRpcPortDefault = 17070;

}  // namespace

EngineTentWrapper::EngineTentWrapper() = default;

EngineTentWrapper::~EngineTentWrapper() {
  if (initialized_) {
    Shutdown().GetMessage();  // best-effort
  }
}

Status EngineTentWrapper::Initialize(const std::string& metadata_addr, const std::string& local_server_name) {
  if (initialized_) {
    return Status(RET_INVALID_ARGUMENT, "EngineTentWrapper already initialized");
  }
  local_server_name_ = local_server_name;

  // Force Mooncake into the tent backend. Without tent we get the
  // legacy TransferEngineImpl path, where sendNotifyByName depends on a
  // handshake daemon that doesn't get started under Redis-backed
  // metadata (notifies are silently dropped). Tent's sendNotifyByName
  // is implemented via openSegment+sendNotification and works out of
  // the box. This matches the upstream tebench reference scripts that
  // run RDMA transfers reliably. pd_v2 always uses tent — we override
  // any pre-existing env so callers can't accidentally fall back.
  setenv("MC_USE_TENT", "1", /*overwrite=*/1);

  // Tent's RPC bind host, RDMA HCA whitelist, transport flags etc. are
  // all configured via the upstream MC_TENT_CONF env (JSON). HCA names
  // are platform-specific (mlx5_*, hns_*, ...) and the GPU↔HCA mapping
  // depends on host PCI topology, so we ship no default — callers must
  // export MC_TENT_CONF before launch. See your tebench reference scripts
  // and docs/technology/design/pd_v2_smoke_runbook.md for examples.
  if (getenv("MC_TENT_CONF") == nullptr) {
    KLLM_LOG_WARNING << "EngineTentWrapper: MC_TENT_CONF env not set — tent will "
                     << "auto-detect RPC bind IP and RDMA HCAs, which may not "
                     << "match the configured inference_addr/GPU. Export "
                     << "MC_TENT_CONF JSON before launch.";
  }

  // Control-plane NIC isolation: EngineConnector does no KV-cache RDMA (that's
  // ExecutorConnector's job in its own subprocess). All EngineConnector traffic
  // is small control-plane messages (notify + stage payload), so a single NIC
  // suffices. Using the full multi-NIC whitelist here lets mooncake's stage
  // WRITE go multi-rail, which in turn spawns notify QPs on every NIC pair —
  // and on mlx5 the qp_num is per-context, so multi-NIC same-number collisions
  // corrupt mooncake's `notify_qp_to_endpoint_` flat map (CQEs get dispatched
  // to the wrong EP, recv buffer reads as all zeros — manifests as Prefill
  // receiving prefill_req with byte_len=1629 but parsed name='' on long
  // requests).
  //
  // Fix: temporarily rewrite MC_TENT_CONF.topology.rdma_whitelist to a single
  // NIC for this engine's TransferEngine::init only, then restore the original
  // env so the executor subprocesses (forked AFTER this point) continue to
  // inherit the full multi-NIC whitelist for KV-cache parallel rail.
  std::string mc_tent_conf_backup;
  bool mc_tent_conf_restore = false;
  const char* mc_tent_conf_env = std::getenv("MC_TENT_CONF");
  if (mc_tent_conf_env != nullptr && *mc_tent_conf_env != '\0') {
    mc_tent_conf_backup.assign(mc_tent_conf_env);
    try {
      nlohmann::json conf = nlohmann::json::parse(mc_tent_conf_backup);
      auto& wl = conf["topology"]["rdma_whitelist"];
      if (wl.is_array() && wl.size() > 1) {
        std::string first_nic = wl[0].get<std::string>();
        wl = nlohmann::json::array({first_nic});
        std::string engine_conf = conf.dump();
        setenv("MC_TENT_CONF", engine_conf.c_str(), /*overwrite=*/1);
        mc_tent_conf_restore = true;
        KLLM_LOG_INFO << "EngineTentWrapper: control-plane NIC isolation — engine TransferEngine "
                      << "init will use single NIC '" << first_nic << "' (full whitelist restored "
                      << "after init so executor subprocesses see multi-NIC).";
      }
    } catch (const std::exception& e) {
      KLLM_LOG_WARNING << "EngineTentWrapper: failed to rewrite MC_TENT_CONF for single-NIC engine "
                       << "init (" << e.what() << ") — falling back to original env. If this is a "
                       << "multi-NIC host the notify QP collision bug may reappear.";
    }
  }

  engine_ = std::make_unique<mooncake::TransferEngine>(/*auto_discover=*/false);
  const std::string md = metadata_addr.empty() ? "P2PHANDSHAKE" : metadata_addr;
  int init_rc = engine_->init(md, local_server_name, /*ip_or_host_name=*/"", kEngineRpcPortDefault);

  // Restore original MC_TENT_CONF immediately after mooncake has loaded its
  // internal Config from env. Mooncake reads env exactly once at init
  // (`ConfigHelper().loadFromEnv(*conf_)`) and from then on uses the in-memory
  // `conf_` object, so restoring here is safe for this engine. The executor
  // subprocesses are forked later and will inherit the restored full
  // multi-NIC env.
  if (mc_tent_conf_restore) {
    setenv("MC_TENT_CONF", mc_tent_conf_backup.c_str(), /*overwrite=*/1);
  }

  if (init_rc != 0) {
    return Status(RET_RUNTIME_FAILED, fmt::format("mooncake TransferEngine::init failed metadata={}", md));
  }
  // Install both transports. Tent-mode installTransport returns nullptr
  // — that's not an error; transports are configured via MC_TENT_CONF.
  // RDMA carries the control-stage WRITEs (prefill_req payload); TCP
  // remains for environments without RDMA HCAs (dev / smoke). Mooncake
  // picks the transport based on what the remote registered.
  void* args = nullptr;
  (void)engine_->installTransport("rdma", &args);
  (void)engine_->installTransport("tcp", &args);

  KLLM_LOG_INFO << "EngineTentWrapper::Initialize metadata=" << md << " local_server_name=" << local_server_name
                << " rpc_port=" << kEngineRpcPortDefault;
  initialized_ = true;
  return Status();
}

Status EngineTentWrapper::Shutdown() {
  if (!initialized_) return Status();
  // mooncake::TransferEngine destructor releases all transports; freeEngine
  // is the explicit teardown but the destructor calls it for us.
  KLLM_LOG_INFO << "EngineTentWrapper::Shutdown local_server_name=" << local_server_name_;
  engine_.reset();
  initialized_ = false;
  return Status();
}

Status EngineTentWrapper::SendNotification(const std::string& peer_server_name, const Notification& notify) {
  if (!initialized_) {
    return Status(RET_INVALID_ARGUMENT, "EngineTentWrapper not initialized");
  }
  // Mooncake serializes NotifyDesc::notify_msg via JSON (jsoncpp), which
  // mangles non-UTF-8 bytes. pd_v2 wire frames are msgpack — pure binary
  // — so wrap them in base64 for transport. The receiver decodes back
  // before deserialize. Cost: ~33% size; alternative would be patching
  // Mooncake to use a binary-safe encoding, which we don't own.
  mooncake::TransferMetadata::NotifyDesc desc;
  desc.name = notify.name;
  desc.notify_msg = base64::to_base64(notify.msg);

  // Resolve target SegmentHandle once per peer and cache it. sendNotifyByName
  // would do open+send+close on every call, and the close races other
  // concurrent calls — see the comment on peer_handle_cache_ in the header.
  mooncake::SegmentHandle handle;
  {
    std::lock_guard<std::mutex> lock(peer_handle_mutex_);
    auto it = peer_handle_cache_.find(peer_server_name);
    if (it != peer_handle_cache_.end()) {
      handle = static_cast<mooncake::SegmentHandle>(it->second);
    } else {
      handle = engine_->openSegment(peer_server_name);
      if (static_cast<int64_t>(handle) < 0) {
        return Status(RET_RUNTIME_FAILED, fmt::format("openSegment({}) failed", peer_server_name));
      }
      peer_handle_cache_[peer_server_name] = static_cast<int64_t>(handle);
    }
  }

  if (engine_->sendNotifyByID(static_cast<mooncake::SegmentID>(handle), desc) != 0) {
    // Send failed — the cached handle may point at a torn-down QP (peer
    // restarted). Drop it so the next call re-opens the segment and rebuilds
    // the notify QP; self-heals even if the epoch-poll invalidation lags.
    {
      std::lock_guard<std::mutex> lock(peer_handle_mutex_);
      peer_handle_cache_.erase(peer_server_name);
    }
    return Status(RET_RUNTIME_FAILED, fmt::format("sendNotifyByID({},{}) failed", peer_server_name, notify.name));
  }
  KLLM_LOG_DEBUG << "EngineTentWrapper::SendNotification peer=" << peer_server_name << " name=" << notify.name
                 << " msg.size=" << notify.msg.size() << " (b64=" << desc.notify_msg.size() << ")";
  return Status();
}

void EngineTentWrapper::InvalidatePeerHandle(const std::string& peer_server_name) {
  std::lock_guard<std::mutex> lock(peer_handle_mutex_);
  if (peer_handle_cache_.erase(peer_server_name) > 0) {
    KLLM_LOG_INFO << "EngineTentWrapper::InvalidatePeerHandle dropped cached notify handle for peer="
                  << peer_server_name;
  }
}

Status EngineTentWrapper::ReceiveNotifications(std::vector<Notification>* out) {
  if (out == nullptr) return Status(RET_INVALID_ARGUMENT, "out is null");
  out->clear();
  if (!initialized_) return Status();
  std::vector<mooncake::TransferMetadata::NotifyDesc> raw;
  if (engine_->getNotifies(raw) != 0) {
    return Status(RET_RUNTIME_FAILED, "mooncake getNotifies failed");
  }
  out->reserve(raw.size());
  if (!raw.empty()) {
    KLLM_LOG_DEBUG << "EngineTentWrapper::ReceiveNotifications got " << raw.size() << " entries on "
                   << local_server_name_ << ", first: name=" << raw[0].name << " msg_size=" << raw[0].notify_msg.size();
  }
  for (auto& n : raw) {
    // Reverse of SendNotification: incoming notify_msg is base64-wrapped
    // (see SendNotification for the rationale). Decode back to raw bytes
    // before handing to the wire-deserializer.
    std::string decoded;
    try {
      decoded = base64::from_base64(n.notify_msg);
    } catch (const std::exception& e) {
      KLLM_LOG_WARNING << "EngineTentWrapper::ReceiveNotifications: base64 decode "
                       << "failed for name=" << n.name << " err=" << e.what() << " — dropping";
      continue;
    }
    out->push_back({std::move(n.name), std::move(decoded)});
  }
  return Status();
}

Status EngineTentWrapper::RegisterStageMemory(void* base, size_t total_bytes) {
  if (!initialized_) {
    return Status(RET_INVALID_ARGUMENT, "EngineTentWrapper not initialized");
  }
  if (base == nullptr || total_bytes == 0) {
    return Status(RET_INVALID_ARGUMENT,
                  fmt::format("RegisterStageMemory invalid base={} total_bytes={}", base, total_bytes));
  }
  // "cpu:0" because HostStagePool is mmap'd anonymous host memory. Same
  // location string Mooncake's own examples use for host buffers.
  // remote_accessible=true is what makes this segment WRITE-able by
  // remote engines (the Decode side, in our case).
  if (engine_->registerLocalMemory(base, total_bytes, /*location=*/"cpu:0",
                                   /*remote_accessible=*/true,
                                   /*update_metadata=*/true) != 0) {
    return Status(RET_RUNTIME_FAILED,
                  fmt::format("registerLocalMemory(stage) failed base={} bytes={}", base, total_bytes));
  }
  KLLM_LOG_INFO << "EngineTentWrapper::RegisterStageMemory base=" << base << " bytes=" << total_bytes;
  return Status();
}

Status EngineTentWrapper::SubmitStageWriteAsync(const std::string& target_server_name, uint64_t target_offset,
                                                void* source, size_t length, uint64_t* out_batch_id) {
  if (out_batch_id) *out_batch_id = 0;
  if (!initialized_) {
    return Status(RET_INVALID_ARGUMENT, "EngineTentWrapper not initialized");
  }
  if (source == nullptr || length == 0) {
    return Status(RET_INVALID_ARGUMENT,
                  fmt::format("SubmitStageWriteAsync invalid source={} length={}", source, length));
  }
  mooncake::SegmentHandle seg = engine_->openSegment(target_server_name);
  if (seg == mooncake::Transport::SegmentID(-1) || static_cast<int64_t>(seg) < 0) {
    return Status(RET_RUNTIME_FAILED, fmt::format("openSegment({}) failed", target_server_name));
  }
  mooncake::BatchID batch = engine_->allocateBatchID(/*batch_size=*/1);
  if (batch == mooncake::INVALID_BATCH_ID) {
    return Status(RET_RUNTIME_FAILED, "allocateBatchID failed");
  }
  mooncake::TransferRequest req;
  req.opcode = mooncake::TransferRequest::WRITE;
  req.source = source;
  req.target_id = static_cast<mooncake::SegmentID>(seg);
  req.target_offset = target_offset;
  req.length = length;
  mooncake::Status s = engine_->submitTransfer(batch, {req});
  if (!s.ok()) {
    engine_->freeBatchID(batch);
    return Status(RET_RUNTIME_FAILED, fmt::format("submitTransfer(stage) failed code={}", static_cast<int>(s.code())));
  }
  *out_batch_id = static_cast<uint64_t>(batch);
  return Status();
}

Status EngineTentWrapper::GetBatchStatus(uint64_t batch_id, bool* out_completed, bool* out_failed) {
  if (out_completed) *out_completed = false;
  if (out_failed) *out_failed = false;
  if (!initialized_) {
    return Status(RET_INVALID_ARGUMENT, "EngineTentWrapper not initialized");
  }
  mooncake::TransferStatus ts;
  mooncake::Status gs = engine_->getBatchTransferStatus(static_cast<mooncake::BatchID>(batch_id), ts);
  if (!gs.ok()) {
    return Status(RET_RUNTIME_FAILED,
                  fmt::format("getBatchTransferStatus failed code={}", static_cast<int>(gs.code())));
  }
  if (ts.s == mooncake::Transport::COMPLETED) {
    if (out_completed) *out_completed = true;
  } else if (ts.s == mooncake::Transport::FAILED || ts.s == mooncake::Transport::TIMEOUT ||
             ts.s == mooncake::Transport::CANCELED || ts.s == mooncake::Transport::INVALID) {
    if (out_failed) *out_failed = true;
    return Status(RET_RUNTIME_FAILED, fmt::format("transfer terminal state={}", static_cast<int>(ts.s)));
  }
  return Status();
}

Status EngineTentWrapper::FreeBatchID(uint64_t batch_id) {
  if (!initialized_) {
    return Status(RET_INVALID_ARGUMENT, "EngineTentWrapper not initialized");
  }
  mooncake::Status s = engine_->freeBatchID(static_cast<mooncake::BatchID>(batch_id));
  if (!s.ok()) {
    KLLM_LOG_WARNING << "EngineTentWrapper::FreeBatchID(" << batch_id << ") failed code=" << static_cast<int>(s.code());
    return Status(RET_RUNTIME_FAILED, "freeBatchID failed");
  }
  return Status();
}

}  // namespace mooncake_backend
}  // namespace pd_v2
}  // namespace ksana_llm
