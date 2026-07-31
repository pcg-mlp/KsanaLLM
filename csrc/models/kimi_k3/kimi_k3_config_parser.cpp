/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/kimi_k3/kimi_k3_config_parser.h"

#include "utils/logger.h"

namespace ksana_llm {

Status KimiK3ConfigParser::ParseModelConfig(const nlohmann::json &config_json,
                                            const ParallelismBasicConfig &parallel_basic_config,
                                            const std::string &model_dir,
                                            std::shared_ptr<BaseModelConfig> &model_config) {
  KLLM_LOG_INFO << "Parse config using kimi_k3 config parser";
  std::shared_ptr<KimiK3Config> kimi_k3_config = std::make_shared<KimiK3Config>();
  model_config = kimi_k3_config;

  kimi_k3_config->model_dir = model_dir;

  // Parse mla config
  kimi_k3_config->mla_config.mla_use_nope = config_json.value("mla_use_nope", true);
  kimi_k3_config->mla_config.mla_use_output_gate = config_json.value("mla_use_output_gate", true);

  // Parse attn_res config
  kimi_k3_config->attn_res_config.attn_res_block_size = config_json.value("attn_res_block_size", 0u);
  KLLM_LOG_INFO << fmt::format("AttnRes config: attn_res_block_size: {}",
                               kimi_k3_config->attn_res_config.attn_res_block_size);

  return Status();
}

}  // namespace ksana_llm
