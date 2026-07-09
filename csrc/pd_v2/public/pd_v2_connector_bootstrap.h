/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>

#include "pd_v2/public/pd_v2_config.h"
#include "pd_v2/public/pd_v2_connector.h"
#include "pd_v2/public/pd_v2_connector_loader.h"
#include "utils/logger.h"
#include "utils/status.h"

namespace ksana_llm {

class IpcFuncWrapper;

namespace pd_v2 {

// dlopen the backend .so, create the connector via its factory, bind the
// engine-process IpcFuncWrapper (no-op on backends / roles that don't need
// it), and Initialize it. Shared by inference_engine (kEngine) and
// inference_executor (kExecutor) so the boilerplate lives in one place.
//   - *out_handle receives the dlopen handle the caller must dlclose on
//     teardown (may be nullptr to ignore).
//   - redis_password (if non-empty) is exported to MC_REDIS_PASSWORD for
//     Mooncake's RedisStoragePlugin; no-op when metadata_addr is not redis://.
//   - context_hint ("engine" / "executor") is embedded in log + error messages.
// Throws (KLLM_THROW) on dlopen/dlsym/factory/Initialize failure.
inline std::shared_ptr<PdV2ConnectorInterface> CreateAndInitConnector(const PdV2ConnectorConfig& cfg,
                                                                      const std::string& redis_password,
                                                                      IpcFuncWrapper* ipc_func_wrapper,
                                                                      void** out_handle,
                                                                      const std::string& context_hint) {
  const PdV2ConnectorLibrary lib = LoadPdV2ConnectorLibrary(cfg.backend_type, context_hint);
  if (out_handle != nullptr) {
    *out_handle = lib.handle;
  }

  if (!redis_password.empty()) {
    setenv("MC_REDIS_PASSWORD", redis_password.c_str(), /*overwrite=*/1);
  }

  std::shared_ptr<PdV2ConnectorInterface> connector = lib.factory(cfg);
  if (connector == nullptr) {
    KLLM_THROW("pd_v2 factory returned nullptr (" + context_hint + ")");
  }
  // Engine-only: bind IpcFuncWrapper for cross-process GetKvBufferVa.
  connector->SetIpcFuncWrapper(ipc_func_wrapper);
  const Status init = connector->Initialize(cfg);
  if (!init.OK()) {
    KLLM_THROW("pd_v2 connector Initialize failed (" + context_hint + "): " + init.GetMessage());
  }
  KLLM_LOG_INFO << "pd_v2 connector created and initialized, context=" << context_hint << " so=" << lib.so_name
                << " cluster=" << cfg.cluster_name << " inference_addr=" << cfg.inference_addr;
  return connector;
}

// pd_v2 Decode warmup gate: poll CanAccept(probe)+Release until a healthy
// Prefill peer appears in the local PeerSelector (which only populates after
// Prefill's first heartbeat), capped at wait_ms. Returns true iff a peer
// became available. Uses a reserved probe req_id so it never collides with
// real requests.
inline bool WaitForHealthyPeer(const std::shared_ptr<PdV2ConnectorInterface>& connector, int wait_ms) {
  if (connector == nullptr) {
    return false;
  }
  constexpr uint64_t kProbeReqId = static_cast<uint64_t>(-2);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(wait_ms);
  while (std::chrono::steady_clock::now() < deadline) {
    if (connector->CanAccept(kProbeReqId, /*input_token_count=*/1)) {
      connector->Release(kProbeReqId);
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  return false;
}

}  // namespace pd_v2
}  // namespace ksana_llm
