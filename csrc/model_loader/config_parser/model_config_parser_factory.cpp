/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "model_loader/config_parser/model_config_parser_factory.h"

#include "models/arc_hunyuan_video/arc_hunyuan_video_config_parser.h"
#include "models/bge_reranker_minicpm/bge_reranker_minicpm_config_parser.h"
#include "models/chatglm/chatglm_config_parser.h"
#include "models/deepseek_v3/deepseek_v3_config_parser.h"
#include "models/deepseek_v4/deepseek_v4_config_parser.h"
#include "models/glm4_moe/glm4_moe_config_parser.h"
#include "models/gpt_moe/gpt_moe_config_parser.h"
#include "models/hy_v3/hy_v3_config_parser.h"
#include "models/hunyuan_turbo/hunyuan_turbo_config_parser.h"
#include "models/internlm2/new_internlm2_config_parser.h"
#include "models/internlmxcomposer2/new_internlmxcomposer2_config_parser.h"
#include "models/kimi_k3/kimi_k3_config_parser.h"
#include "models/llama/llama_model_config_parser.h"
#include "models/mixtral/mixtral_config_parser.h"
#include "models/qwen/qwen_config_parser.h"
#include "models/qwen2_moe/new_qwen2_moe_config_parser.h"
#include "models/qwen3_moe/qwen3_moe_config_parser.h"
#include "utils/string_utils.h"

namespace ksana_llm {

Status ModelConfigParserFactory::CreateModelConfigParser(ModelArchitecture model_arch,
                                                         std::shared_ptr<BaseModelConfigParser>& model_config_parser) {
  switch (model_arch) {
    case ModelArchitecture::ARCH_LLAMA: {
      model_config_parser = std::make_shared<LlamaModelConfigParser>();
      break;
    }
    case ModelArchitecture::ARCH_DEEPSEEK: {
      model_config_parser = std::make_shared<DeepSeekV3ConfigParser>();
      break;
    }
    case ModelArchitecture::ARCH_DEEPSEEK_V4: {
      model_config_parser = std::make_shared<DeepSeekV4ConfigParser>();
      break;
    }
    case ModelArchitecture::ARCH_KIMI_K3: {
      model_config_parser = std::make_shared<KimiK3ConfigParser>();
      break;
    }
    case ModelArchitecture::ARCH_QWEN: {
      model_config_parser = std::make_shared<QwenConfigParser>();
      break;
    }
    case ModelArchitecture::ARCH_INTERNLM2: {
      model_config_parser = std::make_shared<Internlm2ConfigParser>();
      break;
    }
    case ModelArchitecture::ARCH_INTERNLMXCOMPOSER2: {
      model_config_parser = std::make_shared<InternlmxComposer2ConfigParser>();
      break;
    }
    case ModelArchitecture::ARCH_QWEN2_MOE: {
      model_config_parser = std::make_shared<NewQwen2MoeConfigParser>();
      break;
    }
    case ModelArchitecture::ARCH_QWEN3_MOE: {
      model_config_parser = std::make_shared<Qwen3MoeConfigParser>();
      break;
    }
    case ModelArchitecture::ARCH_MIXTRAL: {
      model_config_parser = std::make_shared<MixtralConfigParser>();
      break;
    }
    case ModelArchitecture::ARCH_ARC_HUNYUAN_VIDEO: {
      model_config_parser = std::make_shared<ArcHunyuanVideoConfigParser>();
      break;
    }
    case ModelArchitecture::ARCH_HUNYUAN_TURBO: {
      model_config_parser = std::make_shared<HunyuanTurboConfigParser>();
      break;
    }
    case ModelArchitecture::ARCH_GLM4_MOE: {
      model_config_parser = std::make_shared<Glm4MoeConfigParser>();
      break;
    }
    case ModelArchitecture::ARCH_HY_V3: {
      model_config_parser = std::make_shared<HyV3ConfigParser>();
      break;
    }
    case ModelArchitecture::ARCH_MINICPM: {
      model_config_parser = std::make_shared<BgeRerankerMinicpmConfigParser>();
      break;
    }
    case ModelArchitecture::ARCH_CHATGLM: {
      model_config_parser = std::make_shared<ChatglmConfigParser>();
      break;
    }
    case ModelArchitecture::ARCH_GPT_MOE: {
      model_config_parser = std::make_shared<GptMoeConfigParser>();
      break;
    }
    default: {
      model_config_parser = nullptr;
      return Status(RET_INVALID_ARGUMENT, FormatStr("Not supported model arch %d", static_cast<int>(model_arch)));
    }
  }

  return Status();
}

}  // namespace ksana_llm
