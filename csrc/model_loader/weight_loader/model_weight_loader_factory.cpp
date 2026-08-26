/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "model_loader/weight_loader/model_weight_loader_factory.h"

#include <memory>

#include "utils/status.h"
#include "utils/string_utils.h"

#include "models/arc_hunyuan_video/arc_hunyuan_video_weight_loader.h"
#include "models/bge_reranker_minicpm/bge_reranker_minicpm_weight_loader.h"
#include "models/chatglm/chatglm_model_weight_loader.h"
#include "models/deepseek_v3/deepseek_v3_weight_loader.h"
#include "models/deepseek_v4/deepseek_v4_weight_loader.h"
#include "models/glm4_moe/glm4_moe_weight_loader.h"
#include "models/gpt_moe/gpt_moe_weight_loader.h"
#include "models/hy_v3/hy_v3_weight_loader.h"
#include "models/hunyuan_turbo/hunyuan_turbo_weight_loader.h"
#include "models/internlm2/new_internlm2_weight_loader.h"
#include "models/internlmxcomposer2/new_internlmxcomposer2_weight_loader.h"
#include "models/kimi_k3/kimi_k3_weight_loader.h"
#include "models/llama/llama_model_weight_loader.h"
#include "models/mixtral/mixtral_weight_loader.h"
#include "models/qwen/qwen_weight_loader.h"
#include "models/qwen2_moe/new_qwen2_moe_weight_loader.h"
#include "models/qwen3_moe/qwen3_moe_weight_loader.h"

namespace ksana_llm {

Status ModelWeightLoaderFactory::CreateModelWeightLoader(ModelArchitecture model_arch,
                                                         std::shared_ptr<BaseModelConfig> model_config,
                                                         std::shared_ptr<Environment> env,
                                                         std::shared_ptr<Context> context,
                                                         std::shared_ptr<BaseModelWeightLoader>& model_weight_loader) {
  switch (model_arch) {
    case ModelArchitecture::ARCH_CHATGLM: {
      model_weight_loader = std::make_shared<ChatglmModelWeightLoader>(model_config, env, context);
      break;
    }
    case ModelArchitecture::ARCH_LLAMA: {
      model_weight_loader = std::make_shared<LlamaModelWeightLoader>(model_config, env, context);
      break;
    }
    case ModelArchitecture::ARCH_DEEPSEEK: {
      model_weight_loader = std::make_shared<DeepSeekV3WeightLoader>(model_config, env, context);
      break;
    }
    case ModelArchitecture::ARCH_QWEN: {
      model_weight_loader = std::make_shared<QwenWeightLoader>(model_config, env, context);
      break;
    }
    case ModelArchitecture::ARCH_INTERNLM2: {
      model_weight_loader = std::make_shared<NewInternlm2WeightLoader>(model_config, env, context);
      break;
    }
    case ModelArchitecture::ARCH_INTERNLMXCOMPOSER2: {
      model_weight_loader = std::make_shared<NewInternlmxComposer2WeightLoader>(model_config, env, context);
      break;
    }
    case ModelArchitecture::ARCH_QWEN2_MOE: {
      model_weight_loader = std::make_shared<NewQwen2MoeWeightLoader>(model_config, env, context);
      break;
    }
    case ModelArchitecture::ARCH_QWEN3_MOE: {
      model_weight_loader = std::make_shared<Qwen3MoeWeightLoader>(model_config, env, context);
      break;
    }
    case ModelArchitecture::ARCH_MIXTRAL: {
      model_weight_loader = std::make_shared<MixtralWeightLoader>(model_config, env, context);
      break;
    }
    case ModelArchitecture::ARCH_ARC_HUNYUAN_VIDEO: {
      model_weight_loader = std::make_shared<ArcHunyuanVideoWeightLoader>(model_config, env, context);
      break;
    }
    case ModelArchitecture::ARCH_HUNYUAN_TURBO: {
      model_weight_loader = std::make_shared<HunyuanTurboWeightLoader>(model_config, env, context);
      break;
    }
    case ModelArchitecture::ARCH_GLM4_MOE: {
      model_weight_loader = std::make_shared<Glm4MoeWeightLoader>(model_config, env, context);
      break;
    }
    case ModelArchitecture::ARCH_HY_V3: {
      model_weight_loader = std::make_shared<HyV3WeightLoader>(model_config, env, context);
      break;
    }
    case ModelArchitecture::ARCH_MINICPM: {
      model_weight_loader = std::make_shared<BgeRerankerMinicpmWeightLoader>(model_config, env, context);
      break;
    }
    case ModelArchitecture::ARCH_DEEPSEEK_V4: {
      model_weight_loader = std::make_shared<DeepSeekV4WeightLoader>(model_config, env, context);
      break;
    }
    case ModelArchitecture::ARCH_KIMI_K3: {
      model_weight_loader = std::make_shared<KimiK3WeightLoader>(model_config, env, context);
      break;
    }
    case ModelArchitecture::ARCH_GPT_MOE: {
      model_weight_loader = std::make_shared<GptMoeWeightLoader>(model_config, env, context);
      break;
    }
    default: {
      model_weight_loader = nullptr;
      return Status(RET_INVALID_ARGUMENT, FormatStr("Not supported model arch %d", static_cast<int>(model_arch)));
    }
  }

  return Status();
}

}  // namespace ksana_llm
