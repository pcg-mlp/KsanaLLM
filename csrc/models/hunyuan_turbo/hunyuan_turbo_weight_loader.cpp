/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/hunyuan_turbo/hunyuan_turbo_weight_loader.h"

#include "model_loader/model_loader_utils.h"
#include "models/hunyuan_turbo/hunyuan_turbo_config.h"

namespace ksana_llm {

HunyuanTurboWeightLoader::HunyuanTurboWeightLoader(std::shared_ptr<BaseModelConfig> model_config,
                                                   std::shared_ptr<Environment> env, std::shared_ptr<Context> context)
    : BaseModelWeightLoader(model_config, env, context),
      common_weight_loader_(std::make_unique<CommonModelWeightLoader>(model_config, env, context)) {
  // Initialize pipeline config, for distributed mode.
  env->GetPipelineConfig(pipeline_config_);
  env->GetRuntimeConfig(runtime_config_);
  tp_ = runtime_config_.parallel_basic_config.tensor_parallel_size;
  weight_method_ = std::make_shared<WeightMethod>(common_weight_loader_, tp_);
}

HunyuanTurboWeightLoader::~HunyuanTurboWeightLoader() {}

Status HunyuanTurboWeightLoader::FilterWeightNames(std::vector<std::string>& weight_names) {
  std::shared_ptr<HunyuanTurboConfig> hunyuan_turbo_config =
      std::dynamic_pointer_cast<HunyuanTurboConfig>(model_config_);

  std::vector<std::string> skip_list = {"self_attn.rotary_emb.inv_freq"};
  std::vector<std::string> master_only_list = {"model.embed_tokens.weight"};
  std::vector<std::string> tail_only_list = {"lm_head.weight"};
  std::vector<std::string> master_and_tail_only_list = {"model.embed_tokens.weight"};

  int lower_layer_idx = pipeline_config_.lower_layer_idx;
  int upper_layer_idx = pipeline_config_.upper_layer_idx;

  for (auto it = weight_names.begin(); it != weight_names.end();) {
    if (CheckWeightNameMatched(*it, skip_list, false)) {
      weight_names.erase(it);
      continue;
    }

    // Skip some layers in distributed mode.
    if (lower_layer_idx >= 0 && upper_layer_idx >= 0) {
      int layer_idx = GetLayerIdxFromName(*it);
      if (layer_idx >= 0 && (layer_idx < lower_layer_idx || layer_idx > upper_layer_idx)) {
        weight_names.erase(it);
        continue;
      }
    }

    if (!context_->global->IsStandalone()) {
      if (!hunyuan_turbo_config->tie_word_embeddings) {
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
      } else {
        /* When the "tie-word-embeddings" is set to True in the model's config.json, the model's
         * "model.embed_tokens.weight" and "lm_head.weight" share the same data space. Therefore, it is necessary
         * to load "model.embed_tokens.weight" on tail node.
         */
        KLLM_LOG_DEBUG << "tie_word_embeddings = true, lm_head.weight = model.embed_tokens.weight";
        if (context_->global->IsDistributedWorker() && !context_->global->IsDistributedLastWorker()) {
          if (CheckWeightNameMatched(*it, master_and_tail_only_list, false)) {
            weight_names.erase(it);
            continue;
          }
        }
      }
    }
    ++it;
  }

  return Status();
}

Status HunyuanTurboWeightLoader::PreProcessModelWeights(
    const std::unordered_map<std::string, Tensor>& host_model_weights) {
#define REGISTER_COMMON(key, weight_name)                                                        \
  weight_method_->GetRegistry()[#key][#weight_name].Add(weight_method_->GetCommonMethod().get(), \
                                                        &CommonMethod::key##_##weight_name)
#define REGISTER_CUSTOM(key, weight_name) \
  weight_method_->GetRegistry()[#key][#weight_name].Add(this, &HunyuanTurboWeightLoader::key##_##weight_name)

  // Register common weight processing methods
  REGISTER_COMMON(load, attn_q_k_v_proj);
  REGISTER_COMMON(load, attn_norm);
  REGISTER_COMMON(load, attn_o_proj);
  REGISTER_COMMON(load, mlp_gate_up_proj);
  REGISTER_COMMON(load, mlp_down_proj);
  REGISTER_COMMON(load, norm);
  REGISTER_COMMON(load, embed_tokens);
  REGISTER_COMMON(load, lm_head);

  REGISTER_COMMON(process, attn_qkv_proj);
  REGISTER_COMMON(process, attn_o_proj);
  REGISTER_COMMON(process, mlp_gate_up_proj);
  REGISTER_COMMON(process, mlp_down_proj);
  REGISTER_COMMON(process, lm_head);

  REGISTER_CUSTOM(load, mlp_fused_gate_up_proj);
  REGISTER_CUSTOM(load, attn_fused_qkv_proj);
  REGISTER_CUSTOM(process, mlp_fused_gate_up_proj);
  REGISTER_CUSTOM(process, attn_fused_qkv_proj);

#undef REGISTER_COMMON
#undef REGISTER_CUSTOM
  return Status();
}

Status HunyuanTurboWeightLoader::ProcessModelWeights(const std::unordered_map<std::string, Tensor>& host_model_weights,
                                                     int dev_rank,
                                                     std::unordered_map<std::string, Tensor>& device_model_weights,
                                                     std::unordered_map<std::string, Tensor>& left_host_weights) {
  std::shared_ptr<HunyuanTurboConfig> hunyuan_turbo_config =
      std::dynamic_pointer_cast<HunyuanTurboConfig>(model_config_);
  auto registry = weight_method_->GetRegistry()["load"];

  for (auto& [host_weight_name, host_weight_tensor] : host_model_weights) {
    KLLM_LOG_DEBUG << fmt::format("Dev_rank: {}, processing weight: {}, shape: {}", dev_rank, host_weight_name,
                                  Vector2Str(std::vector<size_t>(host_weight_tensor.shape)));

    // 1. attn weights
    // For fused qkv weights
    if (CheckWeightNameMatched(host_weight_name, {".self_attn.W_pack"}, MatchMode::PartialMatch)) {
      CheckWeightStoreType(is_attn_qkv_store_type_initialized_, is_fused_attn_qkv_proj_, true);
      const std::string qkv_name = WeightNameReplace(host_weight_name, "W_pack", "query_key_value");
      registry["attn_fused_qkv_proj"].Run<Status>(device_model_weights, qkv_name, host_weight_tensor, dev_rank);
      continue;
    }
    // For separated qkv weights
    if (CheckWeightNameMatched(host_weight_name, {".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj"},
                               MatchMode::PartialMatch)) {
      CheckWeightStoreType(is_attn_qkv_store_type_initialized_, is_fused_attn_qkv_proj_, false);
      registry["attn_q_k_v_proj"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }

    if (CheckWeightNameMatched(host_weight_name, {".self_attn.o_proj"}, MatchMode::PartialMatch)) {
      registry["attn_o_proj"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }

    // 2. attn qk norm weights
    if (CheckWeightNameMatched(host_weight_name, {".self_attn.key_layernorm", ".self_attn.query_layernorm"},
                               MatchMode::PartialMatch)) {
      registry["attn_norm"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }

    // 3. mlp weights
    // For fused gate_up weights
    if (CheckWeightNameMatched(host_weight_name, {".mlp.gate_up_proj"}, MatchMode::PartialMatch)) {
      CheckWeightStoreType(is_mlp_gate_up_store_type_initialized_, is_fused_mlp_gate_up_proj_, true);
      registry["mlp_fused_gate_up_proj"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor,
                                                     dev_rank);
      continue;
    }
    // For separated gate_up weights
    if (CheckWeightNameMatched(host_weight_name, {".mlp.gate_proj", ".mlp.up_proj"}, MatchMode::PartialMatch)) {
      CheckWeightStoreType(is_mlp_gate_up_store_type_initialized_, is_fused_mlp_gate_up_proj_, false);
      registry["mlp_gate_up_proj"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }

    if (CheckWeightNameMatched(host_weight_name, {".mlp.down_proj"}, MatchMode::PartialMatch)) {
      registry["mlp_down_proj"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }

    // 4. norm weights
    if (CheckWeightNameMatched(host_weight_name, {"model.norm", ".input_layernorm", ".post_attention_layernorm"},
                               MatchMode::PartialMatch)) {
      registry["norm"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }

    // 5. embedding weights && lm_head weights
    if (CheckWeightNameMatched(host_weight_name, {"model.embed_tokens"}, MatchMode::PartialMatch)) {
      registry["embed_tokens"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      if (hunyuan_turbo_config->tie_word_embeddings) {
        KLLM_LOG_DEBUG << "tie_word_embeddings activated, lm_head.weight = model.embed_tokens.weight";
        const std::string lm_head_weight_name = "lm_head.weight";
        registry["lm_head"].Run<Status>(device_model_weights, lm_head_weight_name, host_weight_tensor, dev_rank);
      }
      continue;
    }
    if (CheckWeightNameMatched(host_weight_name, {"lm_head"}, MatchMode::PartialMatch)) {
      if (!hunyuan_turbo_config->tie_word_embeddings) {
        registry["lm_head"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      }
      continue;
    }
  }
  return Status();
}

Status HunyuanTurboWeightLoader::PostProcessModelWeights(std::unordered_map<std::string, Tensor>& dev_weights_map,
                                                         int dev_rank) {
  std::shared_ptr<HunyuanTurboConfig> hunyuan_turbo_config =
      std::dynamic_pointer_cast<HunyuanTurboConfig>(model_config_);
  auto registry = weight_method_->GetRegistry()["process"];

  KLLM_CHECK_WITH_INFO(is_attn_qkv_store_type_initialized_, "attn_qkv weight store type is not initialized");
  KLLM_CHECK_WITH_INFO(is_mlp_gate_up_store_type_initialized_, "mlp_gate_up weight store type is not initialized");

  for (int layer_idx = pipeline_config_.lower_layer_idx; layer_idx <= pipeline_config_.upper_layer_idx; layer_idx++) {
    // 1. attn weights
    if (is_fused_attn_qkv_proj_) {
      const std::string qkv_proj_name = fmt::format("model.layers.{}.self_attn.query_key_value.", layer_idx);
      registry["attn_fused_qkv_proj"].Run<Status>(dev_weights_map, qkv_proj_name, dev_rank);
    } else {
      const std::string q_proj_name = fmt::format("model.layers.{}.self_attn.q_proj.", layer_idx);
      registry["attn_qkv_proj"].Run<Status>(dev_weights_map, q_proj_name, dev_rank);
    }

    const std::string o_proj_name = fmt::format("model.layers.{}.self_attn.o_proj.", layer_idx);
    registry["attn_o_proj"].Run<Status>(dev_weights_map, o_proj_name, dev_rank);

    // 2. mlp weights
    if (is_fused_mlp_gate_up_proj_) {
      const std::string mlp_gate_up_proj_name = fmt::format("model.layers.{}.mlp.gate_up_proj.", layer_idx);
      registry["mlp_fused_gate_up_proj"].Run<Status>(dev_weights_map, mlp_gate_up_proj_name, dev_rank);
    } else {
      const std::string gate_proj_name = fmt::format("model.layers.{}.mlp.gate_proj.", layer_idx);
      registry["mlp_gate_up_proj"].Run<Status>(dev_weights_map, gate_proj_name, dev_rank);
    }

    const std::string mlp_down_proj_name = fmt::format("model.layers.{}.mlp.down_proj.", layer_idx);
    registry["mlp_down_proj"].Run<Status>(dev_weights_map, mlp_down_proj_name, dev_rank);
  }

  // 3. lm_head weights
  const std::string lm_head_name = "lm_head.";
  registry["lm_head"].Run<Status>(dev_weights_map, lm_head_name, dev_rank);

  for (auto& [weight_name, weight_tensor] : dev_weights_map) {
    if (weight_tensor.dtype != hunyuan_turbo_config->weight_data_type) {
      KLLM_LOG_DEBUG << fmt::format("Dev_rank-{}, cast weight {}, from {} to {}\n", dev_rank, weight_name,
                                    weight_tensor.dtype, hunyuan_turbo_config->weight_data_type);
      CastDeviceTensorType(weight_tensor, hunyuan_turbo_config->weight_data_type, dev_rank);
    }
  }

  return Status();
}

Status HunyuanTurboWeightLoader::load_mlp_fused_gate_up_proj(
    std::unordered_map<std::string, Tensor>& device_model_weights, const std::string& weight_name,
    const Tensor& weight_tensor, int dev_rank) {
  if (CheckWeightNameEndMatched(weight_name, {".weight"})) {
    // Hunyuan-Turbo stores gate_proj and up_proj in one tensor.
    // gate_up_proj layout: [up, gate]
    KLLM_CHECK_WITH_INFO(weight_tensor.shape.size() == 2, "gate_up_proj weight is not a 2d tensor");
    KLLM_CHECK_WITH_INFO(weight_tensor.shape[0] % 2 == 0, "gate_up_proj weight shape[0] is not even");

    std::vector<size_t> gate_up_proj_shape = {weight_tensor.shape[0] / tp_, weight_tensor.shape[1]};
    Tensor gate_up_proj_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, weight_tensor.dtype, gate_up_proj_shape);

    const size_t rank_offset = dev_rank % tp_;
    const size_t src_bytes = weight_tensor.GetTotalBytes() / 2;
    const size_t dst_bytes = src_bytes / tp_;
    const size_t src_offset = rank_offset * dst_bytes;

    MemcpyKind memcpy_kind = weight_tensor.location == MemoryLocation::LOCATION_HOST
                                 ? MemcpyKind::MEMCPY_HOST_TO_DEVICE
                                 : MemcpyKind::MEMCPY_DEVICE_TO_DEVICE;
    // gate_proj
    MemcpyAsync(gate_up_proj_tensor.GetPtr<void>(), weight_tensor.GetPtr<void>() + src_bytes + src_offset, dst_bytes,
                memcpy_kind, context_->device->GetMemoryManageStream());
    // up_proj
    MemcpyAsync(gate_up_proj_tensor.GetPtr<void>() + dst_bytes, weight_tensor.GetPtr<void>() + src_offset, dst_bytes,
                memcpy_kind, context_->device->GetMemoryManageStream());

    device_model_weights[weight_name] = gate_up_proj_tensor;
  }

  return Status();
}

Status HunyuanTurboWeightLoader::load_attn_fused_qkv_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                                          const std::string& weight_name, const Tensor& weight_tensor,
                                                          int dev_rank) {
  if (CheckWeightNameEndMatched(weight_name, {".weight"})) {
    std::shared_ptr<HunyuanTurboConfig> hunyuan_turbo_config =
        std::dynamic_pointer_cast<HunyuanTurboConfig>(model_config_);

    // Hunyuan-Turbo stores q_proj, k_proj and v_proj in one tensor.
    // self_attn.W_pack layout: [q * num_groups, k, v, q * num_groups, k, v, ...]
    const size_t head_num = hunyuan_turbo_config->head_num;
    const size_t num_key_value_heads = hunyuan_turbo_config->num_key_value_heads;
    const size_t head_size = weight_tensor.shape[0] / (head_num + 2 * num_key_value_heads);
    const size_t hidden_size = weight_tensor.shape[1];

    KLLM_CHECK_WITH_INFO(weight_tensor.shape.size() == 2, "W_pack weight is not a 2d tensor");
    KLLM_CHECK_WITH_INFO(weight_tensor.shape[0] % (head_num + 2 * num_key_value_heads) == 0,
                         "W_pack weight shape[0] is not divisible by total heads");

    const size_t q_heads_per_group = head_num / num_key_value_heads;
    std::vector<size_t> qkv_proj_shape = {weight_tensor.shape[0] / tp_, weight_tensor.shape[1]};
    Tensor qkv_proj_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, weight_tensor.dtype, qkv_proj_shape);

    const size_t type_size = weight_tensor.GetDTypeSize();
    const size_t rank_offset = dev_rank % tp_;

    const size_t q_heads_per_tp = head_num / tp_;
    const size_t kv_heads_per_tp = num_key_value_heads / tp_;

    const size_t start_group = rank_offset * kv_heads_per_tp;
    const size_t end_group = start_group + kv_heads_per_tp;

    const size_t head_bytes = head_size * hidden_size * type_size;
    const size_t group_size = (q_heads_per_group + 2) * head_bytes;

    const size_t k_dst_base = q_heads_per_tp * head_bytes;
    const size_t v_dst_base = k_dst_base + kv_heads_per_tp * head_bytes;

    MemcpyKind memcpy_kind = weight_tensor.location == MemoryLocation::LOCATION_HOST
                                 ? MemcpyKind::MEMCPY_HOST_TO_DEVICE
                                 : MemcpyKind::MEMCPY_DEVICE_TO_DEVICE;

    for (size_t group = start_group; group < end_group; group++) {
      const size_t tp_group_idx = group - start_group;

      const size_t q_bytes = q_heads_per_group * head_bytes;
      const size_t src_q_offset = group * group_size;
      const size_t dst_q_offset = tp_group_idx * q_bytes;
      MemcpyAsync(qkv_proj_tensor.GetPtr<void>() + dst_q_offset, weight_tensor.GetPtr<void>() + src_q_offset, q_bytes,
                  memcpy_kind, context_->device->GetMemoryManageStream());

      const size_t kv_bytes = head_bytes;

      const size_t src_k_offset = src_q_offset + q_bytes;
      const size_t dst_k_offset = k_dst_base + tp_group_idx * kv_bytes;
      MemcpyAsync(qkv_proj_tensor.GetPtr<void>() + dst_k_offset, weight_tensor.GetPtr<void>() + src_k_offset, kv_bytes,
                  memcpy_kind, context_->device->GetMemoryManageStream());

      const size_t src_v_offset = src_k_offset + kv_bytes;
      const size_t dst_v_offset = v_dst_base + tp_group_idx * kv_bytes;
      MemcpyAsync(qkv_proj_tensor.GetPtr<void>() + dst_v_offset, weight_tensor.GetPtr<void>() + src_v_offset, kv_bytes,
                  memcpy_kind, context_->device->GetMemoryManageStream());
    }

    device_model_weights[weight_name] = qkv_proj_tensor;
  }

  return Status();
}

Status HunyuanTurboWeightLoader::process_mlp_fused_gate_up_proj(
    std::unordered_map<std::string, Tensor>& device_model_weights, const std::string& weight_prefix_name,
    int dev_rank) {
  const std::string weight_name = weight_prefix_name + "weight";
  common_weight_loader_->Permute2D(device_model_weights.at(weight_name), dev_rank);
  return Status();
}

Status HunyuanTurboWeightLoader::process_attn_fused_qkv_proj(
    std::unordered_map<std::string, Tensor>& device_model_weights, const std::string& weight_prefix_name,
    int dev_rank) {
  const std::string weight_name = weight_prefix_name + "weight";
  common_weight_loader_->Permute2D(device_model_weights.at(weight_name), dev_rank);
  return Status();
}

void HunyuanTurboWeightLoader::CheckWeightStoreType(bool& is_initialized, bool& is_fused, bool is_fused_expected) {
  if (!is_initialized) {
    is_initialized = true;
    is_fused = is_fused_expected;
  } else {
    KLLM_CHECK_WITH_INFO(is_fused == is_fused_expected, "Weight store type is not consistent");
  }
}

}  // namespace ksana_llm
