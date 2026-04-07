/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/arc_hunyuan_video/arc_hunyuan_video_weight_loader.h"

#include "model_loader/model_loader_utils.h"

namespace ksana_llm {
ArcHunyuanVideoWeightLoader::ArcHunyuanVideoWeightLoader(std::shared_ptr<BaseModelConfig> model_config,
                                                         std::shared_ptr<Environment> env,
                                                         std::shared_ptr<Context> context)
    : BaseModelWeightLoader(model_config, env, context),
      common_weight_loader_(std::make_shared<CommonModelWeightLoader>(model_config, env, context)) {
  env->GetPipelineConfig(pipeline_config_);
  env->GetRuntimeConfig(runtime_config_);
  tp_ = runtime_config_.parallel_basic_config.tensor_parallel_size;
  arc_hunyuan_video_config_ = std::dynamic_pointer_cast<ArcHunyuanVideoConfig>(model_config_);
  KLLM_CHECK_WITH_INFO(arc_hunyuan_video_config_->num_key_value_heads % tp_ == 0,
                       "num_key_value_heads must be divisible by tensor_parallel_size");

  if (arc_hunyuan_video_config_->is_quant) {
    weight_method_ =
        std::make_shared<WeightMethod>(common_weight_loader_, tp_, arc_hunyuan_video_config_->quant_config);
  } else {
    weight_method_ = std::make_shared<WeightMethod>(common_weight_loader_, tp_);
  }
}

ArcHunyuanVideoWeightLoader::~ArcHunyuanVideoWeightLoader() {}

Status ArcHunyuanVideoWeightLoader::FilterWeightNames(std::vector<std::string>& weight_names) {
  std::vector<std::string> skip_list = {"speech_encoder.", "vision_model.", "mlp2.0", "mlp2.1", "mlp2.3"};
  std::vector<std::string> master_only_list = {"language_model.model.embed_tokens.weight"};
  std::vector<std::string> tail_only_list = {"language_model.lm_head"};

  int lower_layer_idx = pipeline_config_.lower_layer_idx;
  int upper_layer_idx = pipeline_config_.upper_layer_idx;
  int lower_nextn_layer_idx = pipeline_config_.lower_nextn_layer_idx;
  int upper_nextn_layer_idx = pipeline_config_.upper_nextn_layer_idx;

  for (auto it = weight_names.begin(); it != weight_names.end();) {
    if (CheckWeightNameMatched(*it, skip_list, false)) {
      weight_names.erase(it);
      continue;
    }

    // Skip some layers in distributed mode.
    if (lower_layer_idx >= 0 && upper_layer_idx >= 0) {
      int layer_idx = GetLayerIdxFromName(*it);
      if (layer_idx >= 0 && ((layer_idx < lower_layer_idx || layer_idx > upper_layer_idx) &&
                             (layer_idx < lower_nextn_layer_idx || layer_idx > upper_nextn_layer_idx))) {
        weight_names.erase(it);
        continue;
      }
    }

    if (!context_->global->IsStandalone()) {
      // Skip embedding on if not master mode.
      if (context_->global->IsDistributedWorker()) {
        if (CheckWeightNameMatched(*it, master_only_list, false)) {
          weight_names.erase(it);
          continue;
        }
      }

      // Skip lm_head if not tail mode.
      if (!context_->global->IsDistributedLastWorker()) {
        if (CheckWeightNameMatched(*it, tail_only_list, false)) {
          weight_names.erase(it);
          continue;
        }
      }
    }
    ++it;
  }
  return Status();
}

Status ArcHunyuanVideoWeightLoader::PreProcessModelWeights(
    const std::unordered_map<std::string, Tensor>& host_model_weights) {
  // V1.1 has separate lm_head; V2 uses tied embed_tokens. Called per safetensors file, use OR to keep true once set.
  if (host_model_weights.count("language_model.lm_head.weight") > 0) {
    has_separate_lm_head_ = true;
    KLLM_LOG_INFO << "ArcHunyuanVideo: detected separate lm_head weight (V1.1 style).";
  } else if (!has_separate_lm_head_) {
    KLLM_LOG_INFO << "ArcHunyuanVideo: no separate lm_head weight, using tied embed_tokens (V2 style).";
  }

#define REGISTER_COMMON(key, weight_name)                                                        \
  weight_method_->GetRegistry()[#key][#weight_name].Add(weight_method_->GetCommonMethod().get(), \
                                                        &CommonMethod::key##_##weight_name)

#define REGISTER_CUSTOM(key, weight_name) \
  weight_method_->GetRegistry()[#key][#weight_name].Add(this, &ArcHunyuanVideoWeightLoader::key##_##weight_name)

#define REGISTER_FP8E4M3(key, weight_name)                                                        \
  weight_method_->GetRegistry()[#key][#weight_name].Add(weight_method_->GetFp8E4m3Method().get(), \
                                                        &Fp8E4m3Method::key##_##weight_name)

#define REGISTER_CUSTOM_FP8E4M3(key, weight_name) \
  weight_method_->GetRegistry()[#key][#weight_name].Add(this, &ArcHunyuanVideoWeightLoader::key##_fp8e4m3_##weight_name)

  if (arc_hunyuan_video_config_->quant_config.method == QUANT_FP8_E4M3) {
    // FP8: custom load for qkv/gate_up (TP split + reorder), FP8 load for o_proj/down_proj
    REGISTER_CUSTOM(load, attn_qkv_proj);
    REGISTER_CUSTOM(load, mlp_gate_up_proj);
    REGISTER_FP8E4M3(load, attn_o_proj);
    REGISTER_FP8E4M3(load, mlp_down_proj);

    // FP8: custom FP8 process for qkv/gate_up, FP8 method for o_proj/down_proj
    REGISTER_CUSTOM_FP8E4M3(process, attn_qkv_proj);
    REGISTER_CUSTOM_FP8E4M3(process, mlp_gate_up_proj);
    REGISTER_FP8E4M3(process, attn_o_proj);
    REGISTER_FP8E4M3(process, mlp_down_proj);
  } else {
    // non-quant
    REGISTER_CUSTOM(load, attn_qkv_proj);
    REGISTER_CUSTOM(load, mlp_gate_up_proj);
    REGISTER_COMMON(load, attn_o_proj);
    REGISTER_COMMON(load, mlp_down_proj);

    REGISTER_CUSTOM(process, attn_qkv_proj);
    REGISTER_CUSTOM(process, mlp_gate_up_proj);
    REGISTER_COMMON(process, attn_o_proj);
    REGISTER_COMMON(process, mlp_down_proj);
  }

  // non-linear weights
  REGISTER_COMMON(load, attn_norm);
  REGISTER_COMMON(load, norm);
  REGISTER_COMMON(load, embed_tokens);
  REGISTER_COMMON(load, lm_head);
  REGISTER_COMMON(process, lm_head);

#undef REGISTER_COMMON
#undef REGISTER_CUSTOM
#undef REGISTER_FP8E4M3
#undef REGISTER_CUSTOM_FP8E4M3

  return Status();
}

Status ArcHunyuanVideoWeightLoader::ProcessModelWeights(
    const std::unordered_map<std::string, Tensor>& host_model_weights, int dev_rank,
    std::unordered_map<std::string, Tensor>& device_model_weights,
    std::unordered_map<std::string, Tensor>& left_host_weights) {
  auto registry = weight_method_->GetRegistry()["load"];

  for (auto& [host_weight_name, host_weight_tensor] : host_model_weights) {
    KLLM_LOG_DEBUG << fmt::format("Dev_rank: {}, processing weight: {}, shape: {}", dev_rank, host_weight_name,
                                  Vector2Str(std::vector<size_t>(host_weight_tensor.shape)));

    const std::string new_host_weight_name = RemoveStringPrefix(host_weight_name, "language_model.");

    // 1. attn weights
    if (CheckWeightNameMatched(new_host_weight_name, {".self_attn.qkv_proj"}, MatchMode::PartialMatch)) {
      registry["attn_qkv_proj"].Run<Status>(device_model_weights, new_host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }
    if (CheckWeightNameMatched(new_host_weight_name, {".self_attn.o_proj"}, MatchMode::PartialMatch)) {
      registry["attn_o_proj"].Run<Status>(device_model_weights, new_host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }
    if (CheckWeightNameMatched(new_host_weight_name, {".self_attn.query_layernorm", ".self_attn.key_layernorm"},
                               MatchMode::PartialMatch)) {
      registry["attn_norm"].Run<Status>(device_model_weights, new_host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }

    // 2. mlp weights
    if (CheckWeightNameMatched(new_host_weight_name, {".mlp.gate_and_up_proj"}, MatchMode::PartialMatch)) {
      registry["mlp_gate_up_proj"].Run<Status>(device_model_weights, new_host_weight_name, host_weight_tensor,
                                               dev_rank);
      continue;
    }
    if (CheckWeightNameMatched(new_host_weight_name, {".mlp.down_proj"}, MatchMode::PartialMatch)) {
      registry["mlp_down_proj"].Run<Status>(device_model_weights, new_host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }

    // 3. norm weights
    if (CheckWeightNameMatched(new_host_weight_name, {"model.norm", ".input_layernorm", ".post_attention_layernorm"},
                               MatchMode::PartialMatch)) {
      registry["norm"].Run<Status>(device_model_weights, new_host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }

    // 4. embedding weights && lm_head weights
    if (CheckWeightNameMatched(new_host_weight_name, {"model.embed_tokens"}, MatchMode::PartialMatch)) {
      registry["embed_tokens"].Run<Status>(device_model_weights, new_host_weight_name, host_weight_tensor, dev_rank);
      // V2 tied weights: also load as lm_head
      if (!has_separate_lm_head_) {
        const std::string lm_head_weight_name = "lm_head.weight";
        registry["lm_head"].Run<Status>(device_model_weights, lm_head_weight_name, host_weight_tensor, dev_rank);
      }
      continue;
    }
    // V1.1: separate lm_head
    if (CheckWeightNameMatched(new_host_weight_name, {"lm_head"}, MatchMode::PartialMatch)) {
      registry["lm_head"].Run<Status>(device_model_weights, new_host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }
  }
  return Status();
}

Status ArcHunyuanVideoWeightLoader::PostProcessModelWeights(std::unordered_map<std::string, Tensor>& dev_weights_map,
                                                            int dev_rank) {
  auto registry = weight_method_->GetRegistry()["process"];

  for (int layer_idx = pipeline_config_.lower_layer_idx; layer_idx <= pipeline_config_.upper_layer_idx; layer_idx++) {
    // 1. attn weights
    const std::string qkv_proj_name = fmt::format("model.layers.{}.self_attn.query_key_value.", layer_idx);
    registry["attn_qkv_proj"].Run<Status>(dev_weights_map, qkv_proj_name, dev_rank);

    const std::string o_proj_name = fmt::format("model.layers.{}.self_attn.o_proj.", layer_idx);
    registry["attn_o_proj"].Run<Status>(dev_weights_map, o_proj_name, dev_rank);

    // 2. mlp weights
    const std::string gate_up_proj_name = fmt::format("model.layers.{}.mlp.gate_up_proj.", layer_idx);
    registry["mlp_gate_up_proj"].Run<Status>(dev_weights_map, gate_up_proj_name, dev_rank);

    const std::string down_proj_name = fmt::format("model.layers.{}.mlp.down_proj.", layer_idx);
    registry["mlp_down_proj"].Run<Status>(dev_weights_map, down_proj_name, dev_rank);
  }

  // 3. lm_head
  const std::string lm_head_name = "lm_head.";
  registry["lm_head"].Run<Status>(dev_weights_map, lm_head_name, dev_rank);

  return Status();
}

// load_attn_qkv_proj: TP split + head_groups reorder [Q||K||V]
Status ArcHunyuanVideoWeightLoader::load_attn_qkv_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                                       const std::string& weight_name, const Tensor& weight_tensor,
                                                       int dev_rank) {
  if (!CheckWeightNameEndMatched(weight_name, {".weight"})) {
    return Status();
  }

  // TP split (row-parallel), then reorder: [H, G+2, D, S] -> [Q||K||V, S]
  Tensor dev_tensor =
      common_weight_loader_->TensorParallelSplit(weight_tensor, dev_rank, tp_, TensorParallelMode::RowPara);

  torch::Tensor qkv = common_weight_loader_->GetTorchTensorFromTensor(dev_tensor);
  const size_t heads = arc_hunyuan_video_config_->num_key_value_heads / tp_;
  const size_t head_groups = arc_hunyuan_video_config_->head_groups;
  qkv = qkv.reshape({static_cast<int64_t>(heads), static_cast<int64_t>(head_groups + 2), -1, qkv.size(1)});
  auto tt = qkv.split({static_cast<int64_t>(head_groups), 1, 1}, 1);
  tt[0] = tt[0].reshape({-1, tt[0].size(3)});
  tt[1] = tt[1].reshape({-1, tt[1].size(3)});
  tt[2] = tt[2].reshape({-1, tt[2].size(3)});
  qkv = torch::concat(tt, 0).contiguous();

  const std::string qkv_proj_name = GetReplacedName(weight_name, "qkv_proj", "query_key_value");
  device_model_weights[qkv_proj_name] = common_weight_loader_->GetTensorFromTorchTensor(qkv);
  return Status();
}

// load_mlp_gate_up_proj: swap gate/up order, then TP split
Status ArcHunyuanVideoWeightLoader::load_mlp_gate_up_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                                          const std::string& weight_name, const Tensor& weight_tensor,
                                                          int dev_rank) {
  if (!CheckWeightNameEndMatched(weight_name, {".weight"})) {
    return Status();
  }

  // View gate and up halves from contiguous weight
  Tensor gate_host(weight_tensor.location, weight_tensor.dtype, {weight_tensor.shape[0] / 2, weight_tensor.shape[1]},
                   weight_tensor.GetPtr<void>());
  Tensor up_host(gate_host.location, gate_host.dtype, gate_host.shape,
                 weight_tensor.GetPtr<uint8_t>() + gate_host.GetTotalBytes());

  // TP split each half, concat as [up | gate]
  Tensor up_dev = common_weight_loader_->TensorParallelSplit(up_host, dev_rank, tp_, TensorParallelMode::RowPara);
  Tensor gate_dev = common_weight_loader_->TensorParallelSplit(gate_host, dev_rank, tp_, TensorParallelMode::RowPara);
  Tensor swapped_dev;
  common_weight_loader_->Concat({up_dev, gate_dev}, swapped_dev, dev_rank);

  const std::string gate_up_proj_name = GetReplacedName(weight_name, "gate_and_up_proj", "gate_up_proj");
  device_model_weights[gate_up_proj_name] = swapped_dev;
  return Status();
}

// process_attn_qkv_proj: Permute2D (QKV pre-merged, no AutoMerge needed)
Status ArcHunyuanVideoWeightLoader::process_attn_qkv_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                                          const std::string& weight_prefix_name, int dev_rank) {
  const std::string weight_name = weight_prefix_name + "weight";
  common_weight_loader_->Permute2D(device_model_weights.at(weight_name), dev_rank);
  return Status();
}

// process_mlp_gate_up_proj: Permute2D (gate_up pre-merged, no AutoMerge needed)
Status ArcHunyuanVideoWeightLoader::process_mlp_gate_up_proj(
    std::unordered_map<std::string, Tensor>& device_model_weights, const std::string& weight_prefix_name,
    int dev_rank) {
  const std::string weight_name = weight_prefix_name + "weight";
  common_weight_loader_->Permute2D(device_model_weights.at(weight_name), dev_rank);
  return Status();
}

// process_fp8e4m3_attn_qkv_proj: FP8 quantization (no Permute2D, FP8 GEMM uses non-transposed layout)
Status ArcHunyuanVideoWeightLoader::process_fp8e4m3_attn_qkv_proj(
    std::unordered_map<std::string, Tensor>& device_model_weights, const std::string& weight_prefix_name,
    int dev_rank) {
  const std::string weight_name = weight_prefix_name + "weight";
  return weight_method_->GetFp8E4m3Method()->ConvertWeightToFp8(device_model_weights, weight_name, dev_rank);
}

// process_fp8e4m3_mlp_gate_up_proj: FP8 quantization (no Permute2D, FP8 GEMM uses non-transposed layout)
Status ArcHunyuanVideoWeightLoader::process_fp8e4m3_mlp_gate_up_proj(
    std::unordered_map<std::string, Tensor>& device_model_weights, const std::string& weight_prefix_name,
    int dev_rank) {
  const std::string weight_name = weight_prefix_name + "weight";
  return weight_method_->GetFp8E4m3Method()->ConvertWeightToFp8(device_model_weights, weight_name, dev_rank);
}

}  // namespace ksana_llm
