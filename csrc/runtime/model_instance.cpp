/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "runtime/model_instance.h"

#include <future>
#include <memory>
#include <vector>

#include "device/device_types.h"
#include "nlohmann/json.hpp"
#include "utils/get_custom_weight_name.h"
#include "utils/logger.h"
#include "utils/optional_file.h"
#include "utils/status.h"

#include "models/arc_hunyuan_video/arc_hunyuan_video_model.h"
#include "models/baichuan/baichuan_model.h"
#include "models/bge_reranker_minicpm/bge_reranker_minicpm_model.h"
#include "models/chatglm/chatglm_model.h"
#include "models/deepseek_v3/deepseek_v3_model.h"
#include "models/deepseek_v4/deepseek_v4_model.h"
#include "models/glm4_moe/glm4_moe_model.h"
#include "models/gpt/gpt_model.h"
#include "models/hunyuan_large/hunyuan_large_model.h"
#include "models/hunyuan_turbo/hunyuan_turbo_model.h"
#include "models/internlm2/internlm2_model.h"
#include "models/internlmxcomposer2/internlmxcomposer2_model.h"
#include "models/llama/llama_model.h"
#include "models/llama4/llama4_model.h"
#include "models/mixtral/mixtral_model.h"
#include "models/qwen/qwen_embedding_model.h"
#include "models/qwen/qwen_model.h"
#include "models/qwen2_moe/qwen2_moe_model.h"
#include "models/qwen3_moe/qwen3_moe_model.h"

namespace ksana_llm {

template <class ModelType>
void CreateModelInstance(const ModelConfig& model_config, RuntimeConfig& runtime_config,
                         std::shared_ptr<Context>& context, std::shared_ptr<BaseModel>& model_instance,
                         std::shared_ptr<WeightInstanceInterface>& weight_instance,
                         std::shared_ptr<BaseModelCacheLayout> block_cache_layout) {
  size_t worker_id = context->device->GetDeviceId();
  KLLM_LOG_INFO << "Start to create model on device " << worker_id;
  model_instance = std::make_shared<ModelType>(model_config, runtime_config, worker_id, context,
                                               weight_instance->GetWeight(), block_cache_layout);
}

void ModelInstance::Load() {
  std::string unified_model_type = model_config_.type;
  // unify it to lower case
  std::transform(unified_model_type.begin(), unified_model_type.end(), unified_model_type.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  if (unified_model_type.find("llama4") != std::string::npos) {
    type = "llama4";
    CreateModelInstance<Llama4Model>(model_config_, runtime_config_, context_, model_instance_, weight_instance_,
                                     block_cache_layout_);
  } else if (unified_model_type.find("llama") != std::string::npos) {
    type = "llama";
    CreateModelInstance<LlamaModel>(model_config_, runtime_config_, context_, model_instance_, weight_instance_,
                                    block_cache_layout_);
  } else if (unified_model_type.find("qwen3_moe") != std::string::npos) {
    type = "qwen3_moe";
    CreateModelInstance<Qwen3MoeModel>(model_config_, runtime_config_, context_, model_instance_, weight_instance_,
                                       block_cache_layout_);
  } else if (unified_model_type.find("qwen3") != std::string::npos) {
    type = "qwen3";
    model_config_.enable_qk_pre_norm_before_rotary_pos = true;
    if (model_config_.is_embedding_model) {
      CreateModelInstance<QwenEmbeddingModel>(model_config_, runtime_config_, context_, model_instance_,
                                              weight_instance_, block_cache_layout_);
    } else {
      CreateModelInstance<QwenModel>(model_config_, runtime_config_, context_, model_instance_, weight_instance_,
                                     block_cache_layout_);
    }
  } else if (unified_model_type.find("qwen2_moe") != std::string::npos) {
    type = "qwen2_moe";
    CreateModelInstance<Qwen2MoeModel>(model_config_, runtime_config_, context_, model_instance_, weight_instance_,
                                       block_cache_layout_);
  } else if (unified_model_type.find("qwen") != std::string::npos) {
    type = "qwen";
    // or qwen2_vl
    model_config_.enable_add_qkv_bias = true;
    CreateModelInstance<QwenModel>(model_config_, runtime_config_, context_, model_instance_, weight_instance_,
                                   block_cache_layout_);
  } else if (unified_model_type.find("baichuan") != std::string::npos) {
    type = "baichuan";
    CreateModelInstance<BaichuanModel>(model_config_, runtime_config_, context_, model_instance_, weight_instance_,
                                       block_cache_layout_);
  } else if (unified_model_type.find("chatglm") != std::string::npos) {
    type = "chatglm";
    CreateModelInstance<ChatglmModel>(model_config_, runtime_config_, context_, model_instance_, weight_instance_,
                                      block_cache_layout_);
  } else if (unified_model_type.find("gpt") != std::string::npos ||
             unified_model_type.find("fairseq-transformer") != std::string::npos) {
    CreateModelInstance<GptModel>(model_config_, runtime_config_, context_, model_instance_, weight_instance_,
                                  block_cache_layout_);
  } else if (unified_model_type.find("internlm2") != std::string::npos ||
             unified_model_type.find("internvl_chat") != std::string::npos) {
    type = "internlm2";
    CreateModelInstance<Internlm2Model>(model_config_, runtime_config_, context_, model_instance_, weight_instance_,
                                        block_cache_layout_);
  } else if (unified_model_type.find("internlmxcomposer2") != std::string::npos) {
    type = "internlm2";
    CreateModelInstance<InternlmxComposer2Model>(model_config_, runtime_config_, context_, model_instance_,
                                                 weight_instance_, block_cache_layout_);
  } else if (unified_model_type.find("arc_hunyuan_video") !=
             std::string::npos) {  // 要放在普通hunyuan的前面，避免错误匹配
    type = "arc_hunyuan_video";
    CreateModelInstance<ArcHunyuanVideoModel>(model_config_, runtime_config_, context_, model_instance_,
                                              weight_instance_, block_cache_layout_);
  } else if (unified_model_type.find("hunyuan") != std::string::npos && !model_config_.is_moe) {
    type = "hunyuan";
    CreateModelInstance<HunyuanTurboModel>(model_config_, runtime_config_, context_, model_instance_, weight_instance_,
                                           block_cache_layout_);
  } else if (unified_model_type.find("hunyuan") != std::string::npos && model_config_.is_moe) {
    type = "hunyuan";
    CreateModelInstance<HunyuanLargeModel>(model_config_, runtime_config_, context_, model_instance_, weight_instance_,
                                           block_cache_layout_);
  } else if (unified_model_type.find("mixtral") != std::string::npos) {
    type = "mixtral";
    CreateModelInstance<MixtralModel>(model_config_, runtime_config_, context_, model_instance_, weight_instance_,
                                      block_cache_layout_);
  } else if (unified_model_type.find("glm4_moe") != std::string::npos) {
    type = "glm4_moe";
    model_config_.enable_qk_pre_norm_before_rotary_pos = true;
    model_config_.use_qk_norm = false;  // Disable post-RoPE QK Norm since we use pre-RoPE
    CreateModelInstance<Glm4MoeModel>(model_config_, runtime_config_, context_, model_instance_, weight_instance_,
                                      block_cache_layout_);
  } else if (unified_model_type.find("deepseek_v4") != std::string::npos) {
    type = "deepseek_v4";
    CreateModelInstance<DeepSeekV4Model>(model_config_, runtime_config_, context_, model_instance_, weight_instance_,
                                         block_cache_layout_);
  } else if (unified_model_type.find("deepseek") != std::string::npos ||
             unified_model_type.find("kimi_k2") != std::string::npos ||
             unified_model_type.find("glm_moe_dsa") != std::string::npos) {
    type = "deepseek_v3";
    CreateModelInstance<DeepSeekV3Model>(model_config_, runtime_config_, context_, model_instance_, weight_instance_,
                                         block_cache_layout_);
  } else if (unified_model_type.find("minicpm") != std::string::npos) {
    type = "minicpm";
    CreateModelInstance<BgeRerankerMinicpmModel>(model_config_, runtime_config_, context_, model_instance_,
                                                 weight_instance_, block_cache_layout_);
  } else {
    // Optional weights map
    auto optional_file = Singleton<OptionalFile>::GetInstance();
    std::string& weight_map =
        optional_file->GetOptionalFile(model_config_.path, "weight_map", unified_model_type + "_weight_map.json");
    if (weight_map != "") {
      type = "llama";
      CreateModelInstance<LlamaModel>(model_config_, runtime_config_, context_, model_instance_, weight_instance_,
                                      block_cache_layout_);
    } else {
      KLLM_THROW(fmt::format("Model type {} is not supported.", unified_model_type));
    }
  }
}

void ModelInstance::SetBlockCacheLayout(std::shared_ptr<BaseModelCacheLayout> block_cache_layout) {
  block_cache_layout_ = block_cache_layout;
}

float* ModelInstance::GetLogitsPtr() { return model_instance_->GetLogitsPtr(); }

Status ModelInstance::Forward(std::shared_ptr<BaseWeight> weight, InferStage stage,
                              std::vector<ForwardRequest>& forward_reqs, HiddenUnitDeviceBuffer* hidden_unit,
                              RunMode run_mode, size_t model_input_slot, const int64_t* prev_ring_dev,
                              bool gpu_input_already_prepared) {
  return model_instance_->Forward(weight, forward_reqs, hidden_unit, run_mode, model_input_slot, prev_ring_dev,
                                  gpu_input_already_prepared);
}

Status ModelInstance::PrepareModelInputGpu(std::vector<ForwardRequest>& forward_reqs, RunMode run_mode,
                                           size_t model_input_slot) {
  return model_instance_->PrepareModelInputGpu(forward_reqs, run_mode, model_input_slot);
}

void ModelInstance::GetCurBatchPairs(size_t model_input_slot, const int64_t** out_pairs_dev, int* out_batch_size) {
  if (model_instance_ == nullptr) {
    if (out_pairs_dev) *out_pairs_dev = nullptr;
    if (out_batch_size) *out_batch_size = 0;
    return;
  }
  model_instance_->GetCurBatchPairs(model_input_slot, out_pairs_dev, out_batch_size);
}

void ModelInstance::EvictFinishedRequests(const std::vector<int64_t>& finished_req_ids) {
  if (finished_req_ids.empty() || !model_instance_) {
    return;
  }
  model_instance_->EvictFinishedRequests(finished_req_ids);
}

}  // namespace ksana_llm
