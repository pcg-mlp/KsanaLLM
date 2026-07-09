/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "pd_v2/mooncake/mooncake_engine_connector.h"
#include "pd_v2/mooncake/mooncake_executor_connector.h"
#include "pd_v2/public/pd_v2_connector.h"
#include "utils/logger.h"

namespace ksana_llm {
namespace pd_v2 {

extern "C" std::shared_ptr<PdV2ConnectorInterface> ksana_pd_v2_create_connector(const PdV2ConnectorConfig& cfg) {
  KLLM_LOG_INFO << "ksana_pd_v2_create_connector deployment_role=" << static_cast<int>(cfg.deployment_role)
                << " backend_type=" << cfg.backend_type;
  switch (cfg.deployment_role) {
    case DeploymentRole::kEngine:
      return std::make_shared<mooncake_backend::MooncakeEngineConnector>();
    case DeploymentRole::kExecutor:
      return std::make_shared<mooncake_backend::MooncakeExecutorConnector>();
  }
  KLLM_LOG_ERROR << "ksana_pd_v2_create_connector: unknown deployment_role=" << static_cast<int>(cfg.deployment_role);
  return nullptr;
}

}  // namespace pd_v2
}  // namespace ksana_llm
