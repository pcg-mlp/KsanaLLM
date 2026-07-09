/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstdint>
#include <functional>
#include <memory>

#include "endpoints/pd_v2/pd_v2_prefill_completion.h"
#include "pd_v2/public/pd_v2_connector.h"
#include "pd_v2/public/pd_v2_types.h"

namespace ksana_llm {

class LocalEndpoint;

namespace pd_v2 {

// Prefill-side admission: IncomingPrefillRequest → Request →
// LocalEndpoint::SubmitAsync → ModelDriver::Enqueue.
class PdV2PrefillEndpoint {
 public:
  struct Deps {
    std::shared_ptr<LocalEndpoint> local_endpoint = nullptr;
    std::shared_ptr<PdV2ConnectorInterface> connector = nullptr;
    std::function<PeerHealth()> snapshot_fn = {};
    uint32_t prefill_attn_tp = 0;
  };

  explicit PdV2PrefillEndpoint(Deps deps);
  ~PdV2PrefillEndpoint();

  PdV2PrefillEndpoint(const PdV2PrefillEndpoint&) = delete;
  PdV2PrefillEndpoint& operator=(const PdV2PrefillEndpoint&) = delete;

  void Register();

  static void OnIncomingTrampoline(const IncomingPrefillRequest& req, void* ctx);
  void OnIncoming(const IncomingPrefillRequest& req);

  void OnBarrierDone(int64_t req_id, uint8_t status);

  size_t NumIncomingAdmittedForTest() const;
  size_t NumCompletedForTest() const;
  size_t NumAliveForTest() const;

 private:
  Deps deps_;
  std::shared_ptr<PdV2PrefillCompletion> completion_;
};

}  // namespace pd_v2
}  // namespace ksana_llm
