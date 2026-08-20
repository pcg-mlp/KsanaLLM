/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/deepseek_v3/deepseek_v3_weight_loader.h"

#include <torch/torch.h>

#include <cstring>
#include <regex>

#include "configure/json_config_utils.h"
#include "kernels/trans_layout.h"
#include "model_loader/model_loader_utils.h"
#include "models/weight_method/w4a8_klx_method.h"
#include "utils/indexer_skip_config.h"
#include "utils/logger.h"
#include "utils/memory_utils.h"
#include "utils/string_utils.h"

#ifdef ENABLE_CUDA
#  include "kernels/nvidia/kernel_wrapper.h"
#endif

#ifdef ENABLE_KUNLUN
#  include "kernels/kunlun/quant_gemm_wrapper.h"
#endif

namespace ksana_llm {
DeepSeekV3WeightLoader::DeepSeekV3WeightLoader(std::shared_ptr<BaseModelConfig> model_config,
                                               std::shared_ptr<Environment> env, std::shared_ptr<Context> context)
    : BaseModelWeightLoader(model_config, env, context) {
  // Initialize pipeline config, for distributed mode.
  env->GetPipelineConfig(pipeline_config_);
  env->GetRuntimeConfig(runtime_config_);
  weights_to_permute_.resize(runtime_config_.parallel_basic_config.tensor_parallel_size);

#ifdef ENABLE_KUNLUN
  fc_use_transb_layout_ = true;
#endif

  std::shared_ptr<DeepSeekV3Config> deepseek_v3_config = std::dynamic_pointer_cast<DeepSeekV3Config>(model_config_);

  InitWeightLoaderImpl(deepseek_v3_config->weight_data_type);
}

DeepSeekV3WeightLoader::~DeepSeekV3WeightLoader() {}

Status DeepSeekV3WeightLoader::FilterWeightNames(std::vector<std::string>& weight_names) {
  const std::vector<std::string> skip_list = {"self_attn.rotary_emb.inv_freq", "vision_tower.", "mm_projector."};

  const int lower_layer_idx = pipeline_config_.lower_layer_idx;
  const int upper_layer_idx = pipeline_config_.upper_layer_idx;
  const int lower_nextn_layer_idx = pipeline_config_.lower_nextn_layer_idx;
  const int upper_nextn_layer_idx = pipeline_config_.upper_nextn_layer_idx;

  auto deepseek_v3_config = std::dynamic_pointer_cast<DeepSeekV3Config>(model_config_);
  const std::unordered_set<int> indexer_skip_layers = GetIndexerSkipLayers(
      deepseek_v3_config->num_layer, deepseek_v3_config->num_nextn_predict_layers, deepseek_v3_config->dsa_config);

  for (auto it = weight_names.begin(); it != weight_names.end();) {
    if (CheckWeightNameMatched(*it, skip_list, false)) {
      it = weight_names.erase(it);
      continue;
    }

    // Strip VLM prefix for layer index extraction (regex_match requires full-string match).
    const std::string& wname = *it;
    static const std::string vlm_prefix = "language_model.";
    const std::string& name_for_layer =
        (wname.size() > vlm_prefix.size() && wname.compare(0, vlm_prefix.size(), vlm_prefix) == 0)
            ? wname.substr(vlm_prefix.size())
            : wname;
    const int layer_idx = GetLayerIdxFromName(name_for_layer);

    // Skip some layers in distributed mode.
    if (lower_layer_idx >= 0 && upper_layer_idx >= 0) {
      if (layer_idx >= 0 && ((layer_idx < lower_layer_idx || layer_idx > upper_layer_idx) &&
                             (layer_idx < lower_nextn_layer_idx || layer_idx > upper_nextn_layer_idx))) {
        it = weight_names.erase(it);
        continue;
      }
      // Skip nextn layer in distributed non-sampling node
      if (layer_idx >= 0 && !context_->global->IsSamplingNode() &&
          (layer_idx >= lower_nextn_layer_idx && layer_idx <= upper_nextn_layer_idx)) {
        it = weight_names.erase(it);
        continue;
      }
    }

    // Skip the weights of unused indexers.
    if (indexer_skip_layers.count(layer_idx)) {
      if (CheckWeightNameMatched(*it, {fmt::format("model.layers.{}.self_attn.indexer.", layer_idx)},
                                 MatchMode::PartialMatch)) {
        it = weight_names.erase(it);
        continue;
      }
    }

    if (!context_->global->IsStandalone()) {
      // Skip embedding on if not master mode.
      if (context_->global->IsDistributedWorker() &&
          !(context_->global->IsSamplingNode() && runtime_config_.mtp_step_num > 0)) {
        if (CheckWeightNameMatched(*it, {"model.embed_tokens.weight"}, true)) {
          it = weight_names.erase(it);
          continue;
        }
      }

      // Skip lm_head/final_norm if not tail mode.
      if (!context_->global->IsDistributedLastWorker()) {
        if (CheckWeightNameMatched(*it, {"lm_head.weight"}, true) ||
            CheckWeightNameMatched(*it, {"model.norm.weight"}, true)) {
          it = weight_names.erase(it);
          continue;
        }
      }
    }
    ++it;
  }
  return Status();
}

#ifdef ENABLE_KUNLUN
std::unordered_set<std::string> DeepSeekV3WeightLoader::CreateW4A8KlxScales(
    std::unordered_map<std::string, Tensor>& dev_weights_map, int dev_rank,
    std::shared_ptr<DeepSeekV3Config>& deepseek_v3_config) {
  // Create weight_scale tensors for fused/renamed weights in W4A8_KLX quantization.
  // Returns the set of weight names already in [k, n] layout (permuted during ProcessModelWeights).
  std::unordered_set<std::string> already_permuted_weights;

  size_t num_layers = deepseek_v3_config->num_layer;
  size_t q_lora_rank = deepseek_v3_config->mla_config.q_lora_rank;
  size_t kv_lora_rank = deepseek_v3_config->mla_config.kv_lora_rank;
  size_t qk_rope_head_dim = deepseek_v3_config->mla_config.qk_rope_head_dim;
  size_t qk_nope_head_dim = deepseek_v3_config->mla_config.qk_nope_head_dim;
  size_t head_num = deepseek_v3_config->head_num;
  size_t attn_tp_size = context_->global->GetAttentionTensorParallelSize();
  size_t attn_dev_rank = dev_rank % attn_tp_size;

  for (size_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
    std::string layer_prefix = "model.layers." + std::to_string(layer_idx) + ".self_attn.";

    // 1. fused_lora_a_proj.weight_scale = concat(q_a_proj.weight_scale, kv_a_proj_with_mqa.weight_scale)
    std::string fused_weight_name = layer_prefix + "fused_lora_a_proj.weight";
    std::string fused_scale_name = fused_weight_name + "_scale";
    if (dev_weights_map.find(fused_weight_name) != dev_weights_map.end() &&
        dev_weights_map.find(fused_scale_name) == dev_weights_map.end()) {
      std::string q_a_scale_name = layer_prefix + "q_a_proj.weight_scale";
      std::string kv_a_scale_name = layer_prefix + "kv_a_proj_with_mqa.weight_scale";
      auto q_a_scale_it = dev_weights_map.find(q_a_scale_name);
      auto kv_a_scale_it = dev_weights_map.find(kv_a_scale_name);
      if (q_a_scale_it != dev_weights_map.end() && kv_a_scale_it != dev_weights_map.end()) {
        Tensor& q_a_scale = q_a_scale_it->second;
        Tensor& kv_a_scale = kv_a_scale_it->second;
        size_t fused_scale_elements = q_lora_rank + kv_lora_rank + qk_rope_head_dim;
        Tensor fused_scale_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, q_a_scale.dtype, {fused_scale_elements, 1},
                                           nullptr, &(context_->device->GetMemoryManageStream()));
        MemcpyAsync(fused_scale_tensor.GetPtr<void>(), q_a_scale.GetPtr<void>(), q_a_scale.GetTotalBytes(),
                    MEMCPY_DEVICE_TO_DEVICE, context_->device->GetMemoryManageStream());
        MemcpyAsync(fused_scale_tensor.GetPtr<uint8_t>() + q_a_scale.GetTotalBytes(), kv_a_scale.GetPtr<void>(),
                    kv_a_scale.GetTotalBytes(), MEMCPY_DEVICE_TO_DEVICE, context_->device->GetMemoryManageStream());
        dev_weights_map[fused_scale_name] = fused_scale_tensor;
        KLLM_LOG_DEBUG << fmt::format("W4A8_KLX: created fused scale {} shape=[{}, 1]", fused_scale_name,
                                      fused_scale_elements);
      }
    }

    // 2. q_b_nope_rope_proj.weight_scale = TP-split of q_b_proj.weight_scale
    //    q_b_nope_rope_proj.weight was already permuted during ProcessModelWeights.
    std::string q_b_nope_rope_weight_name = layer_prefix + "q_b_nope_rope_proj.weight";
    std::string q_b_nope_rope_scale_name = q_b_nope_rope_weight_name + "_scale";
    if (dev_weights_map.find(q_b_nope_rope_weight_name) != dev_weights_map.end() &&
        dev_weights_map.find(q_b_nope_rope_scale_name) == dev_weights_map.end()) {
      std::string q_b_scale_name = layer_prefix + "q_b_proj.weight_scale";
      auto q_b_scale_it = dev_weights_map.find(q_b_scale_name);
      if (q_b_scale_it != dev_weights_map.end()) {
        Tensor& q_b_scale = q_b_scale_it->second;
        size_t full_out_dim = head_num * (qk_nope_head_dim + qk_rope_head_dim);
        size_t split_out_dim = DivRoundUp(full_out_dim, attn_tp_size);
        size_t scale_element_size = q_b_scale.GetDTypeSize();
        size_t tp_offset = attn_dev_rank * split_out_dim * scale_element_size;
        size_t tp_bytes = split_out_dim * scale_element_size;
        Tensor split_scale_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, q_b_scale.dtype, {split_out_dim, 1},
                                           nullptr, &(context_->device->GetMemoryManageStream()));
        MemcpyAsync(split_scale_tensor.GetPtr<void>(), q_b_scale.GetPtr<uint8_t>() + tp_offset, tp_bytes,
                    MEMCPY_DEVICE_TO_DEVICE, context_->device->GetMemoryManageStream());
        dev_weights_map[q_b_nope_rope_scale_name] = split_scale_tensor;
        already_permuted_weights.insert(q_b_nope_rope_weight_name);
      }
    }

    // 3. kv_b_nope_proj and v_head_proj: their scales are created in ProcessModelWeights (host-side split).
    //    Just mark them as already permuted.
    if (dev_weights_map.count(layer_prefix + "kv_b_nope_proj.weight"))
      already_permuted_weights.insert(layer_prefix + "kv_b_nope_proj.weight");
    if (dev_weights_map.count(layer_prefix + "v_head_proj.weight"))
      already_permuted_weights.insert(layer_prefix + "v_head_proj.weight");

    // 3b. kv_b_combined_proj: created during ProcessModelWeights, already permuted.
    //     Create its weight_scale by concatenating kv_b_nope_proj and v_head_proj scales.
    std::string combined_weight_name = layer_prefix + "kv_b_combined_proj.weight";
    if (dev_weights_map.count(combined_weight_name)) {
      already_permuted_weights.insert(combined_weight_name);
      std::string combined_scale_name = combined_weight_name + "_scale";
      if (dev_weights_map.find(combined_scale_name) == dev_weights_map.end()) {
        std::string nope_scale_name = layer_prefix + "kv_b_nope_proj.weight_scale";
        std::string v_scale_name = layer_prefix + "v_head_proj.weight_scale";
        auto nope_scale_it = dev_weights_map.find(nope_scale_name);
        auto v_scale_it = dev_weights_map.find(v_scale_name);
        if (nope_scale_it != dev_weights_map.end() && v_scale_it != dev_weights_map.end()) {
          Tensor& nope_scale = nope_scale_it->second;
          Tensor& v_scale = v_scale_it->second;
          size_t combined_elements =
              nope_scale.GetTotalBytes() / nope_scale.GetDTypeSize() + v_scale.GetTotalBytes() / v_scale.GetDTypeSize();
          Tensor combined_scale_tensor =
              Tensor(MemoryLocation::LOCATION_DEVICE, nope_scale.dtype, {combined_elements, 1}, nullptr,
                     &(context_->device->GetMemoryManageStream()));
          MemcpyAsync(combined_scale_tensor.GetPtr<void>(), nope_scale.GetPtr<void>(), nope_scale.GetTotalBytes(),
                      MEMCPY_DEVICE_TO_DEVICE, context_->device->GetMemoryManageStream());
          MemcpyAsync(combined_scale_tensor.GetPtr<uint8_t>() + nope_scale.GetTotalBytes(), v_scale.GetPtr<void>(),
                      v_scale.GetTotalBytes(), MEMCPY_DEVICE_TO_DEVICE, context_->device->GetMemoryManageStream());
          dev_weights_map[combined_scale_name] = combined_scale_tensor;
          KLLM_LOG_DEBUG << fmt::format("W4A8_KLX: created combined scale {} elements={}", combined_scale_name,
                                        combined_elements);
        }
      }
    }
  }

  // 4. gate_up_proj.weight_scale = concat(TP-shard of gate_proj.weight_scale, TP-shard of up_proj.weight_scale)
  size_t mlp_tp_size = context_->global->GetTensorParallelSize();
  std::vector<std::string> gate_up_patterns = {".mlp.gate_up_proj.weight", ".mlp.shared_expert.gate_up_proj.weight"};
  for (size_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
    std::string layer_prefix_mlp = "model.layers." + std::to_string(layer_idx);
    for (const auto& pattern : gate_up_patterns) {
      std::string gate_up_weight_name = layer_prefix_mlp + pattern;
      std::string gate_up_scale_name = gate_up_weight_name + "_scale";
      if (dev_weights_map.find(gate_up_weight_name) == dev_weights_map.end()) continue;
      if (dev_weights_map.find(gate_up_scale_name) != dev_weights_map.end()) continue;

      std::string gate_scale_name =
          std::regex_replace(gate_up_weight_name, std::regex("gate_up_proj"), "gate_proj") + "_scale";
      std::string up_scale_name =
          std::regex_replace(gate_up_weight_name, std::regex("gate_up_proj"), "up_proj") + "_scale";
      auto gate_scale_it = dev_weights_map.find(gate_scale_name);
      auto up_scale_it = dev_weights_map.find(up_scale_name);
      if (gate_scale_it == dev_weights_map.end() || up_scale_it == dev_weights_map.end()) continue;

      Tensor& gate_scale = gate_scale_it->second;
      Tensor& up_scale = up_scale_it->second;
      size_t current_tp_size =
          (pattern.find("shared_expert") != std::string::npos && runtime_config_.enable_full_shared_expert)
              ? 1
              : mlp_tp_size;
      size_t current_dev_rank = dev_rank % current_tp_size;
      size_t full_dim = gate_scale.shape[0];
      size_t split_dim = DivRoundUp(full_dim, current_tp_size);
      size_t scale_elem_size = gate_scale.GetDTypeSize();
      size_t tp_off = current_dev_rank * split_dim * scale_elem_size;
      size_t tp_bytes = split_dim * scale_elem_size;
      size_t combined_dim = split_dim * 2;
      Tensor combined_scale = Tensor(MemoryLocation::LOCATION_DEVICE, gate_scale.dtype, {combined_dim, 1}, nullptr,
                                     &(context_->device->GetMemoryManageStream()));
      MemcpyAsync(combined_scale.GetPtr<void>(), gate_scale.GetPtr<uint8_t>() + tp_off, tp_bytes,
                  MEMCPY_DEVICE_TO_DEVICE, context_->device->GetMemoryManageStream());
      MemcpyAsync(combined_scale.GetPtr<uint8_t>() + tp_bytes, up_scale.GetPtr<uint8_t>() + tp_off, tp_bytes,
                  MEMCPY_DEVICE_TO_DEVICE, context_->device->GetMemoryManageStream());
      dev_weights_map[gate_up_scale_name] = combined_scale;
      KLLM_LOG_DEBUG << fmt::format("W4A8_KLX: created gate_up scale {} shape=[{}, 1]", gate_up_scale_name,
                                    combined_dim);
    }
  }

  // 5. BMM INT8 权重处理 (对齐 xsglang deepseek_v2.py:5162-5191 post_load_weights)
  //
  // NdScaleTensor B_scale 语义: scale = max_abs (quant2d 格式, 不除 127)
  // 模型文件 weight_scale 语义: scale = max_abs / 127
  //
  // w_uk_t: permute{0,2,1} 导致原 per-output-channel scale 变为 per-K-channel,
  //   需 dequant(per-K) → quant2d(per-N) 重量化
  // w_uv: 不做 permute, per-output-channel 方向天然为 per-N-channel,
  //   只需 scale *= 127 对齐 NdScaleTensor 格式
  //
  // NOTE(zhongzhicao): 当前 BMM layer 存在 bf16 fallback (weight.dtype != INT8 时走 XpuBatchedMatMul<T>)。
  //   此处仅在 W4A8_KLX 且 weight 为 INT8 时触发。
  StreamSynchronize(context_->device->GetMemoryManageStream());

  for (size_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
    std::string layer_prefix = "model.layers." + std::to_string(layer_idx) + ".self_attn.";

    // ─── w_uk_t: dequant(per-K) → quant2d(per-N) ───
    {
      std::string w_uk_t_name = layer_prefix + "w_uk_t.weight";
      std::string w_uk_t_scale_name = w_uk_t_name + "_scale";
      std::string nope_scale_name = layer_prefix + "kv_b_nope_proj.weight_scale";

      auto w_it = dev_weights_map.find(w_uk_t_name);
      auto s_it = dev_weights_map.find(nope_scale_name);
      if (w_it != dev_weights_map.end() && s_it != dev_weights_map.end()) {
        Tensor& w_tensor = w_it->second;   // [B, N, K] INT8 on device
        Tensor& old_scale = s_it->second;  // [B*K, 1] FP32 on device (per-K-channel)

        size_t B = w_tensor.shape[0];  // heads_tp
        size_t N = w_tensor.shape[1];  // kv_lora_rank
        size_t K = w_tensor.shape[2];  // qk_nope_head_dim

        // (1) D2H: int8 weight + old scale
        std::vector<int8_t> h_int8(B * N * K);
        std::vector<float> h_old_scale(B * K);
        Memcpy(h_int8.data(), w_tensor.GetPtr<void>(), B * N * K * sizeof(int8_t), MEMCPY_DEVICE_TO_HOST);
        Memcpy(h_old_scale.data(), old_scale.GetPtr<void>(), B * K * sizeof(float), MEMCPY_DEVICE_TO_HOST);

        // (2) Host dequant + cast to bf16
        // old_scale is per-K-channel: w_uk_t[b,:,k] shares scale[b*K+k]
        // dequant: real = int8 * scale (weight_scale = max_abs/127)
        std::vector<c10::BFloat16> h_bf16(B * N * K);
        for (size_t b = 0; b < B; ++b) {
          for (size_t n = 0; n < N; ++n) {
            for (size_t k = 0; k < K; ++k) {
              size_t idx = (b * N + n) * K + k;
              float real_val = static_cast<float>(h_int8[idx]) * h_old_scale[b * K + k];
              h_bf16[idx] = c10::BFloat16(real_val);
            }
          }
        }

        // (3) H2D: bf16 buffer → device temp tensor [B*N, K]
        Tensor dev_bf16 = Tensor(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_BF16, {B * N, K}, nullptr,
                                 &(context_->device->GetMemoryManageStream()));
        MemcpyAsync(dev_bf16.GetPtr<void>(), h_bf16.data(), B * N * K * sizeof(c10::BFloat16), MEMCPY_HOST_TO_DEVICE,
                    context_->device->GetMemoryManageStream());

        // (4) Device quant2d: [B*N, K] bf16 → [B*N, K] int8 + [B*N] fp32 scale (max_abs format)
        Tensor new_int8_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_INT8, {B * N, K}, nullptr,
                                        &(context_->device->GetMemoryManageStream()));
        Tensor new_scale_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_FP32, {B * N, 1}, nullptr,
                                         &(context_->device->GetMemoryManageStream()));

        StreamSynchronize(context_->device->GetMemoryManageStream());
        xdnn::Context* xctx = static_cast<xdnn::Context*>(
            context_->device->ext->GetXdnnContext(context_->device->GetMemoryManageStream().Get()));
        int ret = XpuPerTokenQuantInt8<bfloat16>(xctx, dev_bf16.GetPtr<bfloat16>(), new_int8_tensor.GetPtr<int8_t>(),
                                                 new_scale_tensor.GetPtr<float>(), static_cast<int64_t>(B * N),
                                                 static_cast<int64_t>(K));
        KLLM_CHECK_WITH_INFO(ret == 0,
                             fmt::format("W4A8_KLX: w_uk_t quant2d failed, layer={}, ret={}", layer_idx, ret));

        // (5) Replace w_uk_t weight and create new scale
        new_int8_tensor.shape = {B, N, K};
        dev_weights_map[w_uk_t_name] = new_int8_tensor;
        dev_weights_map[w_uk_t_scale_name] = new_scale_tensor;

        KLLM_LOG_DEBUG << fmt::format("W4A8_KLX: w_uk_t requantized per-N-channel, scale [{}, 1] max_abs format",
                                      B * N);
      }
    }

    // ─── w_uv: scale *= 127 (对齐 NdScaleTensor max_abs 格式) ───
    {
      std::string w_uv_scale_name = layer_prefix + "w_uv.weight_scale";
      std::string v_head_scale_name = layer_prefix + "v_head_proj.weight_scale";
      auto v_scale_it = dev_weights_map.find(v_head_scale_name);
      if (dev_weights_map.find(w_uv_scale_name) == dev_weights_map.end() && v_scale_it != dev_weights_map.end()) {
        Tensor& src_scale = v_scale_it->second;  // [B*N, 1] FP32
        size_t num_elements = src_scale.GetTotalBytes() / sizeof(float);

        std::vector<float> h_scale(num_elements);
        Memcpy(h_scale.data(), src_scale.GetPtr<void>(), num_elements * sizeof(float), MEMCPY_DEVICE_TO_HOST);
        for (size_t i = 0; i < num_elements; ++i) {
          h_scale[i] *= 127.0f;
        }
        Tensor new_scale = Tensor(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_FP32, src_scale.shape, nullptr,
                                  &(context_->device->GetMemoryManageStream()));
        MemcpyAsync(new_scale.GetPtr<void>(), h_scale.data(), num_elements * sizeof(float), MEMCPY_HOST_TO_DEVICE,
                    context_->device->GetMemoryManageStream());
        dev_weights_map[w_uv_scale_name] = new_scale;
      }
    }
  }

  return already_permuted_weights;
}
#endif  // ENABLE_KUNLUN

Status DeepSeekV3WeightLoader::PostProcessModelWeights(std::unordered_map<std::string, Tensor>& dev_weights_map,
                                                       int dev_rank) {
  KLLM_LOG_DEBUG << fmt::format("Dev_rank: {}, [PostProcess] start, total weights: {}", dev_rank,
                                dev_weights_map.size());
  std::shared_ptr<DeepSeekV3Config> deepseek_v3_config = std::dynamic_pointer_cast<DeepSeekV3Config>(model_config_);
  std::unordered_set<std::string> post_processed_weights;
  if (deepseek_v3_config->quant_config.is_fp8_blockwise) {
#if defined(ENABLE_FP8) && defined(ENABLE_FP8_TORCH)
    // Post process weight and scale for q_b_proj and kv_b_proj
    weight_impl_->PostProcessFp8E4m3BlockWiseQuantWeights(dev_weights_map, dev_rank, deepseek_v3_config);
    KLLM_LOG_DEBUG << fmt::format("Dev_rank: {}, [PostProcess] PostProcessFp8E4m3BlockWiseQuantWeights done", dev_rank);
#endif
    for (auto& [weight_name, weight_tensor] : dev_weights_map) {
      // for fp8 weight, bind weight scale
      if (weight_tensor.dtype == DataType::TYPE_FP8_E4M3 || weight_tensor.dtype == DataType::TYPE_BLOCK_FP8_E4M3) {
        std::string weight_scale_name = weight_name + "_scale_inv";
        auto scale_iter = dev_weights_map.find(weight_scale_name);
        if (scale_iter != dev_weights_map.end()) {
          weight_tensor.weight_scales = &(scale_iter->second);
          KLLM_LOG_DEBUG << fmt::format("Dev_rank: {}, [PostProcess] bind fp8 scale {} (shape: {}) -> {} (shape: {})",
                                        dev_rank, weight_scale_name,
                                        Vector2Str(std::vector<size_t>(weight_tensor.weight_scales->shape)),
                                        weight_name, Vector2Str(std::vector<size_t>(weight_tensor.shape)));
          post_processed_weights.insert(weight_scale_name);
          post_processed_weights.insert(weight_name);
        } else {
          KLLM_THROW(fmt::format("weight scale inv not found: {}", weight_scale_name));
        }
      }
    }
  }

#ifdef ENABLE_KUNLUN
  // W4A8 KLX post-processing: permute weights, cast scales to fp32, bind, apply *7 for W4, apply *127 for W8
  if (deepseek_v3_config->quant_config.method == QUANT_W4A8_KLX) {
    std::unordered_set<std::string> already_permuted_weights =
        CreateW4A8KlxScales(dev_weights_map, dev_rank, deepseek_v3_config);

    StreamSynchronize(context_->device->GetMemoryManageStream());

    for (auto& [weight_name, weight_tensor] : dev_weights_map) {
      // Skip non-INT8 weights and scale tensors
      if (weight_tensor.dtype != DataType::TYPE_INT8) {
        continue;
      }
      // Skip if it's a weight_scale tensor (handled alongside the weight)
      if (W4A8KlxMethod::IsWeightScale(weight_name)) {
        continue;
      }
      // Check if corresponding scale exists
      std::string scale_name = W4A8KlxMethod::GetScaleName(weight_name);
      if (dev_weights_map.find(scale_name) == dev_weights_map.end()) {
        continue;
      }

      // Process 3D tensors (including stacked expert tensors and mla absorb bmm tensors)
      if (weight_tensor.shape.size() != 2) {
        // For 3D stacked expert tensors: cast scale to FP32 and apply *7
        Tensor& scale_tensor = dev_weights_map[scale_name];
        if (scale_tensor.dtype != DataType::TYPE_FP32) {
          CastDeviceTensorType(scale_tensor, DataType::TYPE_FP32, dev_rank);
        }

        if (W4A8KlxMethod::IsW4ExpertWeight(weight_name)) {
          // Multiply scale by 7 for W4 weights (scale is FP32 on device, small tensor)
          size_t num_elements = scale_tensor.GetTotalBytes() / sizeof(float);
          std::vector<float> host_scale(num_elements);
          MemcpyAsync(host_scale.data(), scale_tensor.GetPtr<void>(), scale_tensor.GetTotalBytes(),
                      MEMCPY_DEVICE_TO_HOST, context_->device->GetMemoryManageStream());
          StreamSynchronize(context_->device->GetMemoryManageStream());
          for (size_t i = 0; i < num_elements; ++i) {
            host_scale[i] *= 7.0f;
          }
          MemcpyAsync(scale_tensor.GetPtr<void>(), host_scale.data(), scale_tensor.GetTotalBytes(),
                      MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());

          KLLM_LOG_DEBUG << fmt::format("W4A8_KLX W4 weight post-process: {} [{}x{}] scale*=7", weight_name,
                                        weight_tensor.shape[0], weight_tensor.shape[1]);
        }
        weight_tensor.weight_scales = &dev_weights_map[scale_name];
        post_processed_weights.insert(weight_name);
        post_processed_weights.insert(scale_name);

        continue;
      }

      // 2D weight: permute {1, 0} to get [k, n] layout (or [k/2, n] for W4)
      // Skip permute when using transb layout (weights stay in [n, k])
      // or for weights already in [k, n] layout (permuted during ProcessModelWeights)
      if (!fc_use_transb_layout_ && already_permuted_weights.find(weight_name) == already_permuted_weights.end()) {
        weight_impl_->PermuteWeight(weight_tensor, {1, 0}, dev_rank);
      }

      // Cast scale BF16 -> FP32
      Tensor& scale_tensor = dev_weights_map[scale_name];
      if (scale_tensor.dtype != DataType::TYPE_FP32) {
        CastDeviceTensorType(scale_tensor, DataType::TYPE_FP32, dev_rank);
      }

      // Scale denormalization: multiply by 127 (W8 INT8 range)
      // This converts scale from (max_abs / 127) format to max_abs format for Kunlun XPU kernel
      // Note: W4 expert weights are 3D (stacked) and handled in the 3D branch above.
      {
        size_t num_elements = scale_tensor.GetTotalBytes() / sizeof(float);
        std::vector<float> host_scale(num_elements);
        MemcpyAsync(host_scale.data(), scale_tensor.GetPtr<void>(), scale_tensor.GetTotalBytes(), MEMCPY_DEVICE_TO_HOST,
                    context_->device->GetMemoryManageStream());
        StreamSynchronize(context_->device->GetMemoryManageStream());
        for (size_t i = 0; i < num_elements; ++i) {
          host_scale[i] *= 127.0f;
        }
        MemcpyAsync(scale_tensor.GetPtr<void>(), host_scale.data(), scale_tensor.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
                    context_->device->GetMemoryManageStream());
      }

      // Bind scale to weight
      weight_tensor.weight_scales = &dev_weights_map[scale_name];

      post_processed_weights.insert(weight_name);
      post_processed_weights.insert(scale_name);

      KLLM_LOG_DEBUG << fmt::format("W4A8_KLX W8 weight post-process: {} [{}x{}] scale*=127", weight_name,
                                    weight_tensor.shape[0], weight_tensor.shape[1]);
    }
  }
#endif  // ENABLE_KUNLUN

  for (auto& [weight_name, weight_tensor] : dev_weights_map) {
    if (weights_to_permute_[dev_rank].find(weight_name) != weights_to_permute_[dev_rank].end()) {
      // permute weight tensor
      KLLM_LOG_DEBUG << fmt::format("Dev_rank: {}, [PostProcess] permute: {}, shape before: {}, dtype: {}", dev_rank,
                                    weight_name, Vector2Str(std::vector<size_t>(weight_tensor.shape)),
                                    weight_tensor.dtype);
      weight_impl_->PermuteWeight(weight_tensor, {1, 0}, dev_rank);
      KLLM_LOG_DEBUG << fmt::format("Dev_rank: {}, [PostProcess] permute done: {}, shape after: {}", dev_rank,
                                    weight_name, Vector2Str(std::vector<size_t>(weight_tensor.shape)));
    }

    if (post_processed_weights.find(weight_name) != post_processed_weights.end()) {
      continue;
    }

    if (!(weight_tensor.dtype == DataType::TYPE_FP32 || weight_tensor.dtype == DataType::TYPE_FP16 ||
          weight_tensor.dtype == DataType::TYPE_BF16)) {
      continue;
    }

    // NOTE: `e_score_correction_bias` need to cast into fp32 for better performance
    if (weight_name.find("gate.e_score_correction_bias") != std::string::npos) {
      KLLM_LOG_DEBUG << fmt::format("Dev_rank: {}, [PostProcess] cast {} from {} to {}", dev_rank, weight_name,
                                    weight_tensor.dtype, TYPE_FP32);
      CastDeviceTensorType(weight_tensor, TYPE_FP32, dev_rank);
      continue;
    }
    if (weight_tensor.dtype != deepseek_v3_config->weight_data_type) {
      KLLM_LOG_DEBUG << fmt::format("Dev_rank: {}, [PostProcess] cast {} from {} to {}", dev_rank, weight_name,
                                    weight_tensor.dtype, deepseek_v3_config->weight_data_type);
      CastDeviceTensorType(weight_tensor, deepseek_v3_config->weight_data_type, dev_rank);
    }
  }

  if (deepseek_v3_config->ContainGptqWeights()) {
    KLLM_LOG_DEBUG << fmt::format("Dev_rank: {}, [PostProcess] start gptq post process", dev_rank);
    weight_impl_->PostProcessInt4QuantWeights(dev_weights_map, dev_rank, deepseek_v3_config);
    KLLM_LOG_DEBUG << fmt::format("Dev_rank: {}, [PostProcess] gptq post process done", dev_rank);
  }

  KLLM_LOG_DEBUG << fmt::format("Dev_rank: {}, [PostProcess] complete, total weights: {}", dev_rank,
                                dev_weights_map.size());
  return Status();
}

#ifdef ENABLE_KUNLUN
void DeepSeekV3WeightLoader::LoadW4A8KlxWeightScale(const std::string& file_weight_name,
                                                    const Tensor& host_weight_tensor,
                                                    std::unordered_map<std::string, Tensor>& device_model_weights,
                                                    std::shared_ptr<DeepSeekV3Config>& deepseek_v3_config,
                                                    size_t attn_dev_rank, size_t head_num_tp, size_t qk_nope_head_dim,
                                                    size_t v_head_dim) {
  // For kv_b_proj.weight_scale: do host-side head-interleaved split into nope and v_head scales,
  // then H2D each separately. Avoids inefficient D2D strided copy in PostProcess.
  if (file_weight_name.find(".kv_b_proj.weight_scale") != std::string::npos) {
    size_t elem_size = GetTypeSize(host_weight_tensor.dtype);
    size_t stride_per_head = qk_nope_head_dim + v_head_dim;  // elements per head
    size_t tp_head_offset = attn_dev_rank * head_num_tp;     // starting head for this rank
    size_t total_elements = host_weight_tensor.GetTotalBytes() / elem_size;

    // 1. Cast full kv_b_proj.weight_scale to fp32 on host, then H2D
    std::vector<float> full_fp32_buf(total_elements);
    const void* raw_src = host_weight_tensor.GetPtr<void>();
    if (host_weight_tensor.dtype == DataType::TYPE_BF16) {
      const c10::BFloat16* bf16_src = reinterpret_cast<const c10::BFloat16*>(raw_src);
      for (size_t i = 0; i < total_elements; ++i) full_fp32_buf[i] = static_cast<float>(bf16_src[i]);
    } else if (host_weight_tensor.dtype == DataType::TYPE_FP16) {
      const c10::Half* fp16_src = reinterpret_cast<const c10::Half*>(raw_src);
      for (size_t i = 0; i < total_elements; ++i) full_fp32_buf[i] = static_cast<float>(fp16_src[i]);
    } else {
      std::memcpy(full_fp32_buf.data(), raw_src, total_elements * sizeof(float));
    }
    Tensor dev_full = Tensor(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_FP32, host_weight_tensor.shape, nullptr,
                             &(context_->device->GetMemoryManageStream()));
    MemcpyAsync(dev_full.GetPtr<void>(), full_fp32_buf.data(), total_elements * sizeof(float), MEMCPY_HOST_TO_DEVICE,
                context_->device->GetMemoryManageStream());
    device_model_weights[file_weight_name] = dev_full;

    // 2. Host-side gather from fp32 buffer: extract nope and v_head scale for this TP rank
    size_t nope_out_dim = head_num_tp * qk_nope_head_dim;
    size_t v_out_dim = head_num_tp * v_head_dim;
    std::vector<float> nope_fp32_buf(nope_out_dim);
    std::vector<float> v_fp32_buf(v_out_dim);
    for (size_t h = 0; h < head_num_tp; ++h) {
      size_t src_head_offset = (tp_head_offset + h) * stride_per_head;
      std::memcpy(nope_fp32_buf.data() + h * qk_nope_head_dim, full_fp32_buf.data() + src_head_offset,
                  qk_nope_head_dim * sizeof(float));
      std::memcpy(v_fp32_buf.data() + h * v_head_dim, full_fp32_buf.data() + src_head_offset + qk_nope_head_dim,
                  v_head_dim * sizeof(float));
    }

    // 3. H2D for nope scale (fp32)
    std::string layer_prefix_kv = file_weight_name.substr(0, file_weight_name.find(".kv_b_proj.weight_scale"));
    std::string nope_scale_name = layer_prefix_kv + ".kv_b_nope_proj.weight_scale";
    Tensor nope_dev = Tensor(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_FP32, {nope_out_dim, 1}, nullptr,
                             &(context_->device->GetMemoryManageStream()));
    MemcpyAsync(nope_dev.GetPtr<void>(), nope_fp32_buf.data(), nope_out_dim * sizeof(float), MEMCPY_HOST_TO_DEVICE,
                context_->device->GetMemoryManageStream());
    device_model_weights[nope_scale_name] = nope_dev;

    // 4. H2D for v_head scale (fp32)
    std::string v_scale_name = layer_prefix_kv + ".v_head_proj.weight_scale";
    Tensor v_dev = Tensor(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_FP32, {v_out_dim, 1}, nullptr,
                          &(context_->device->GetMemoryManageStream()));
    MemcpyAsync(v_dev.GetPtr<void>(), v_fp32_buf.data(), v_out_dim * sizeof(float), MEMCPY_HOST_TO_DEVICE,
                context_->device->GetMemoryManageStream());
    device_model_weights[v_scale_name] = v_dev;

    // 5. Combined scale for kv_b_combined_proj (TP-sharded, not head-interleaved)
    {
      std::string combined_scale_name = layer_prefix_kv + ".kv_b_combined_proj.weight_scale";
      size_t combined_out_dim = head_num_tp * (qk_nope_head_dim + v_head_dim);
      std::vector<float> combined_fp32_buf(combined_out_dim);
      for (size_t h = 0; h < head_num_tp; ++h) {
        size_t src_head_offset = (tp_head_offset + h) * stride_per_head;
        std::memcpy(combined_fp32_buf.data() + h * stride_per_head, full_fp32_buf.data() + src_head_offset,
                    stride_per_head * sizeof(float));
      }
      Tensor combined_dev = Tensor(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_FP32, {combined_out_dim, 1}, nullptr,
                                   &(context_->device->GetMemoryManageStream()));
      MemcpyAsync(combined_dev.GetPtr<void>(), combined_fp32_buf.data(), combined_out_dim * sizeof(float),
                  MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());
      device_model_weights[combined_scale_name] = combined_dev;
    }

    KLLM_LOG_DEBUG << fmt::format("W4A8_KLX: host-split kv_b scale -> {} [{}] + {} [{}]", nope_scale_name, nope_out_dim,
                                  v_scale_name, v_out_dim);
    return;
  }

  // Host-side cast scale bf16/fp16 → fp32 before H2D (Kunlun has no device cast kernel)
  size_t scale_num_elements = host_weight_tensor.GetTotalBytes() / GetTypeSize(host_weight_tensor.dtype);
  if (host_weight_tensor.dtype != DataType::TYPE_FP32) {
    std::vector<float> fp32_buf(scale_num_elements);
    const void* src = host_weight_tensor.GetPtr<void>();
    if (host_weight_tensor.dtype == DataType::TYPE_BF16) {
      const c10::BFloat16* bf16_src = reinterpret_cast<const c10::BFloat16*>(src);
      for (size_t i = 0; i < scale_num_elements; ++i) fp32_buf[i] = static_cast<float>(bf16_src[i]);
    } else if (host_weight_tensor.dtype == DataType::TYPE_FP16) {
      const c10::Half* fp16_src = reinterpret_cast<const c10::Half*>(src);
      for (size_t i = 0; i < scale_num_elements; ++i) fp32_buf[i] = static_cast<float>(fp16_src[i]);
    }
    Tensor dev_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_FP32, host_weight_tensor.shape, nullptr,
                               &(context_->device->GetMemoryManageStream()));
    MemcpyAsync(dev_tensor.GetPtr<void>(), fp32_buf.data(), scale_num_elements * sizeof(float), MEMCPY_HOST_TO_DEVICE,
                context_->device->GetMemoryManageStream());
    device_model_weights[file_weight_name] = dev_tensor;
  } else {
    Tensor dev_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_FP32, host_weight_tensor.shape, nullptr,
                               &(context_->device->GetMemoryManageStream()));
    MemcpyAsync(dev_tensor.GetPtr<void>(), host_weight_tensor.GetPtr<void>(), host_weight_tensor.GetTotalBytes(),
                MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());
    device_model_weights[file_weight_name] = dev_tensor;
  }
}
#endif  // ENABLE_KUNLUN

void DeepSeekV3WeightLoader::LoadMlaKvBProj(const std::string& file_weight_name, const Tensor& host_weight_tensor,
                                            int dev_rank, std::shared_ptr<DeepSeekV3Config>& deepseek_v3_config,
                                            std::unordered_map<std::string, Tensor>& device_model_weights,
                                            size_t head_num, size_t qk_nope_head_dim, size_t qk_rope_head_dim,
                                            size_t v_head_dim, size_t kv_lora_rank, size_t attn_tp_size,
                                            size_t attn_dev_rank, size_t head_num_tp) {
  if (head_num * (qk_nope_head_dim + v_head_dim) != host_weight_tensor.shape[0]) {
    KLLM_THROW(
        fmt::format("The shape of the 0th dim of the weight named '{}' is not equal to the sum of qk_nope_head_dim {} "
                    "and v_head_dim {}.",
                    file_weight_name, kv_lora_rank, qk_rope_head_dim));
  }

  if (!deepseek_v3_config->quant_config.is_fp8_blockwise) {
    // For kv_b_nope_proj weight load
    std::string kv_b_nope_name =
        file_weight_name.substr(0, file_weight_name.find_first_of('_')) + "_attn.kv_b_nope_proj.weight";
    std::vector<size_t> kv_b_nope_shape = {static_cast<size_t>(DivRoundUp(head_num * qk_nope_head_dim, attn_tp_size)),
                                           host_weight_tensor.shape[1]};

    size_t para_pitch = DivRoundUp(head_num, attn_tp_size) * (qk_nope_head_dim + v_head_dim) *
                        host_weight_tensor.shape[1] * GetTypeSize(host_weight_tensor.dtype);
    size_t tensor_para_offset = attn_dev_rank * para_pitch;

    Tensor kv_b_nope_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, kv_b_nope_shape,
                                     nullptr, &(context_->device->GetMemoryManageStream()));
    size_t nope_dst_pitch = qk_nope_head_dim * host_weight_tensor.shape[1] * GetTypeSize(host_weight_tensor.dtype);
    size_t src_pitch =
        (qk_nope_head_dim + v_head_dim) * host_weight_tensor.shape[1] * GetTypeSize(host_weight_tensor.dtype);
    Memcpy2DAsync(kv_b_nope_tensor.GetPtr<void>(), nope_dst_pitch,
                  host_weight_tensor.GetPtr<void>() + tensor_para_offset, src_pitch, nope_dst_pitch,
                  DivRoundUp(head_num, attn_tp_size), MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());

    Tensor kv_b_nope_permute = Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, kv_b_nope_shape,
                                      nullptr, &(context_->device->GetMemoryManageStream()));
    if (!fc_use_transb_layout_) {
      PermuteDeviceTensor(kv_b_nope_tensor, {1, 0}, dev_rank, kv_b_nope_permute);
      device_model_weights[kv_b_nope_name] = kv_b_nope_permute;
    } else {
      device_model_weights[kv_b_nope_name] = kv_b_nope_tensor;
    }
    KLLM_LOG_DEBUG << fmt::format("Dev_rank: {}, [MLA kv_b_nope] stored: {}, shape: {}, dtype: {}", dev_rank,
                                  kv_b_nope_name,
                                  Vector2Str(std::vector<size_t>(device_model_weights[kv_b_nope_name].shape)),
                                  device_model_weights[kv_b_nope_name].dtype);

    // For v_head_proj weight load
    std::string v_head_name =
        file_weight_name.substr(0, file_weight_name.find_first_of('_')) + "_attn.v_head_proj.weight";
    std::vector<size_t> v_head_shape = {static_cast<size_t>(DivRoundUp(head_num * v_head_dim, attn_tp_size)),
                                        host_weight_tensor.shape[1]};

    Tensor v_head_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, v_head_shape, nullptr,
                                  &(context_->device->GetMemoryManageStream()));
    size_t v_head_dst_pitch = v_head_dim * host_weight_tensor.shape[1] * GetTypeSize(host_weight_tensor.dtype);
    Memcpy2DAsync(v_head_tensor.GetPtr<void>(), v_head_dst_pitch,
                  host_weight_tensor.GetPtr<void>() + nope_dst_pitch + tensor_para_offset, src_pitch, v_head_dst_pitch,
                  DivRoundUp(head_num, attn_tp_size), MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());

    Tensor v_head_permute = Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, v_head_shape, nullptr,
                                   &(context_->device->GetMemoryManageStream()));
    if (!fc_use_transb_layout_) {
      PermuteDeviceTensor(v_head_tensor, {1, 0}, dev_rank, v_head_permute);
      device_model_weights[v_head_name] = v_head_permute;
    } else {
      device_model_weights[v_head_name] = v_head_tensor;
    }

    KLLM_LOG_DEBUG << fmt::format("Dev_rank: {}, [MLA v_head] stored: {}, shape: {}, dtype: {}", dev_rank, v_head_name,
                                  Vector2Str(std::vector<size_t>(device_model_weights[v_head_name].shape)),
                                  device_model_weights[v_head_name].dtype);

    // Create w_uk_t from kv_b_nope_proj data
    std::string w_uk_t_name = kv_b_nope_name.substr(0, kv_b_nope_name.find_first_of('_')) + "_attn.w_uk_t.weight";
    Tensor w_uk_t_tensor;
    if (fc_use_transb_layout_) {
      // Deep copy: PermuteWeight's buffer reuse would corrupt kv_b_nope_proj on next layer.
      w_uk_t_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, kv_b_nope_tensor.dtype,
                             {head_num_tp, qk_nope_head_dim, kv_b_nope_tensor.shape[1]}, nullptr,
                             &(context_->device->GetMemoryManageStream()));
      MemcpyAsync(w_uk_t_tensor.GetPtr<void>(), kv_b_nope_tensor.GetPtr<void>(), kv_b_nope_tensor.GetTotalBytes(),
                  MEMCPY_DEVICE_TO_DEVICE, context_->device->GetMemoryManageStream());
      // Kunlun: xblas::fc_batch requires trans_b=true, weight in [B, N, K] layout.
      // Current: [B, qk_nope_head_dim, kv_lora_rank] → permute to [B, kv_lora_rank(N), qk_nope_head_dim(K)]
      weight_impl_->PermuteWeight(w_uk_t_tensor, {0, 2, 1}, dev_rank);
    } else {
      w_uk_t_tensor = kv_b_nope_tensor;
      w_uk_t_tensor.shape = {head_num_tp, qk_nope_head_dim, kv_b_nope_tensor.shape[1]};
    }
    device_model_weights[w_uk_t_name] = w_uk_t_tensor;
    KLLM_LOG_DEBUG << fmt::format("Dev_rank: {}, [MLA w_uk_t] stored: {}, shape: {}, dtype: {}", dev_rank, w_uk_t_name,
                                  Vector2Str(std::vector<size_t>(w_uk_t_tensor.shape)), w_uk_t_tensor.dtype);

    // Permute vhead_weight_name to w_uv
    v_head_tensor.shape = {head_num_tp, v_head_dim, v_head_shape[1]};
    if (!fc_use_transb_layout_) {
      // CUDA: weight needs to be [B, K, N] = [B, kv_lora_rank, v_head_dim]
      weight_impl_->PermuteWeight(v_head_tensor, {0, 2, 1}, dev_rank);
    }
    // fc_use_transb_layout_: keep [B, v_head_dim(N), kv_lora_rank(K)] = [B, N, K] for trans_b=true.
    std::string w_uv_name = v_head_name.substr(0, v_head_name.find_first_of('_')) + "_attn.w_uv.weight";
    device_model_weights[w_uv_name] = v_head_tensor;
    KLLM_LOG_DEBUG << fmt::format("Dev_rank: {}, [MLA w_uv] stored: {}, shape: {}, dtype: {}", dev_rank, w_uv_name,
                                  Vector2Str(std::vector<size_t>(v_head_tensor.shape)), v_head_tensor.dtype);

#ifdef ENABLE_KUNLUN
    // Kunlun MLA optimization: create kv_b_combined_proj weight
    // This is the unsplit kv_b_proj (TP-sharded but NOT split into nope/v_head).
    // Used by dpsk_prefill_rotary_embedding which expects kv = [tokens, heads, qk_nope + v_head_dim].
    {
      std::string kv_b_combined_name =
          file_weight_name.substr(0, file_weight_name.find_first_of('_')) + "_attn.kv_b_combined_proj.weight";
      std::vector<size_t> combined_shape = {
          static_cast<size_t>(DivRoundUp(head_num * (qk_nope_head_dim + v_head_dim), attn_tp_size)),
          host_weight_tensor.shape[1]};

      Tensor combined_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, combined_shape,
                                      nullptr, &(context_->device->GetMemoryManageStream()));
      MemcpyAsync(combined_tensor.GetPtr<void>(), host_weight_tensor.GetPtr<void>() + tensor_para_offset, para_pitch,
                  MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());

      if (fc_use_transb_layout_) {
        // fc_use_transb_layout: keep [N, K] = [heads_tp*(nope+v), kv_lora_rank]
        device_model_weights[kv_b_combined_name] = combined_tensor;
      } else {
        // Legacy: permute to [K, N] = [kv_lora_rank, heads_tp*(nope+v)]
        Tensor combined_permute = Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype,
                                         {host_weight_tensor.shape[1], combined_shape[0]}, nullptr,
                                         &(context_->device->GetMemoryManageStream()));
        PermuteDeviceTensor(combined_tensor, {1, 0}, dev_rank, combined_permute);
        device_model_weights[kv_b_combined_name] = combined_permute;
      }
    }
#endif  // ENABLE_KUNLUN
  } else {
    // For fp8 blockwise quant, do not split the weights initially, split them after dequantization later.
    std::vector<size_t> kv_b_proj_shape = {static_cast<size_t>(DivRoundUp(host_weight_tensor.shape[0], attn_tp_size)),
                                           host_weight_tensor.shape[1]};

    Tensor kv_b_proj_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, kv_b_proj_shape,
                                     nullptr, &(context_->device->GetMemoryManageStream()));
    size_t para_pitch = DivRoundUp(head_num, attn_tp_size) * (qk_nope_head_dim + v_head_dim) *
                        host_weight_tensor.shape[1] * GetTypeSize(host_weight_tensor.dtype);
    size_t tensor_para_offset = attn_dev_rank * para_pitch;
    MemcpyAsync(kv_b_proj_tensor.GetPtr<void>(), host_weight_tensor.GetPtr<void>() + tensor_para_offset, para_pitch,
                MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());

    device_model_weights[file_weight_name] = kv_b_proj_tensor;
    KLLM_LOG_DEBUG << fmt::format("Dev_rank: {}, [MLA kv_b_proj fp8] stored: {}, shape: {}, dtype: {}", dev_rank,
                                  file_weight_name, Vector2Str(std::vector<size_t>(kv_b_proj_tensor.shape)),
                                  kv_b_proj_tensor.dtype);
  }
}

Status DeepSeekV3WeightLoader::ProcessModelWeights(const std::unordered_map<std::string, Tensor>& host_model_weights,
                                                   int dev_rank,
                                                   std::unordered_map<std::string, Tensor>& device_model_weights,
                                                   std::unordered_map<std::string, Tensor>& left_host_weights) {
  std::shared_ptr<DeepSeekV3Config> deepseek_v3_config = std::dynamic_pointer_cast<DeepSeekV3Config>(model_config_);

  int32_t layer_idx = -1, expert_idx = -1;
  size_t num_experts = deepseek_v3_config->moe_config.num_experts;
  bool use_vllm_moe = deepseek_v3_config->moe_config.use_vllm_moe;
  KLLM_LOG_DEBUG << fmt::format("num_experts = {}, use_vllm_moe = {}", num_experts, use_vllm_moe);
  KLLM_LOG_DEBUG << "is_fp8_blockwise: " << deepseek_v3_config->quant_config.is_fp8_blockwise;
  // moe combines tensor parallel and moe parallel
  ExpertParallelConfig& expert_parallel_config = deepseek_v3_config->expert_parallel_config;
  size_t expert_node_rank = expert_parallel_config.expert_node_rank;
  size_t global_expert_para_size = expert_parallel_config.expert_world_size * expert_parallel_config.expert_para_size;
  size_t num_experts_per_rank = DivRoundUp(num_experts, global_expert_para_size);

  // init expert map
  size_t rank_expert_offset = expert_node_rank * expert_parallel_config.expert_para_size * num_experts_per_rank;
  size_t expert_offset =
      (global_expert_para_size > 1) ? ((dev_rank % deepseek_v3_config->expert_para_size) * num_experts_per_rank) : 0;
  size_t expert_start_id = rank_expert_offset + expert_offset;
  size_t expert_end_id = std::min(num_experts, expert_start_id + num_experts_per_rank);
  size_t num_layers = deepseek_v3_config->num_layer;
  if (pipeline_config_.lower_nextn_layer_idx >= num_layers) {
    num_layers += pipeline_config_.upper_nextn_layer_idx - pipeline_config_.lower_nextn_layer_idx + 1;
  }
  std::vector<std::vector<int>> expert_map(num_layers, std::vector<int>(num_experts, -1));
  InitExpertMap(num_experts, expert_start_id, expert_end_id, dev_rank, expert_map, device_model_weights);

  KLLM_LOG_DEBUG << fmt::format("expert_world_size = {}, expert_para_size = {}, global_expert_para_size = {}",
                                expert_parallel_config.expert_world_size, expert_parallel_config.expert_para_size,
                                global_expert_para_size);
  KLLM_LOG_DEBUG << fmt::format("In rank {} node_rank {}, valid experts is [{}, {})", dev_rank, expert_node_rank,
                                expert_start_id, expert_end_id);

  size_t moe_inter_size_per_rank =
      DivRoundUp(deepseek_v3_config->moe_config.moe_inter_size, deepseek_v3_config->moe_tensor_para_size);
  size_t hidden_units = deepseek_v3_config->hidden_units;
  std::vector<size_t> up_gate_experts_shape = {num_experts_per_rank,
                                               /* up & gate*/ moe_inter_size_per_rank * 2, hidden_units};
  std::vector<size_t> down_experts_shape = {num_experts_per_rank, hidden_units, moe_inter_size_per_rank};

  size_t attn_tp_size = context_->global->GetAttentionTensorParallelSize();
  size_t attn_dev_rank = dev_rank % attn_tp_size;
  size_t kv_lora_rank = deepseek_v3_config->mla_config.kv_lora_rank;
  size_t qk_rope_head_dim = deepseek_v3_config->mla_config.qk_rope_head_dim;
  size_t qk_nope_head_dim = deepseek_v3_config->mla_config.qk_nope_head_dim;
  size_t v_head_dim = deepseek_v3_config->mla_config.v_head_dim;
  size_t q_lora_rank = deepseek_v3_config->mla_config.q_lora_rank;
  size_t head_num = deepseek_v3_config->head_num;
  size_t head_num_tp = static_cast<size_t>(DivRoundUp(deepseek_v3_config->head_num, attn_tp_size));

  if (deepseek_v3_config->model_format == ModelFormat::GGUF) {
    return Status(RET_INVALID_ARGUMENT, "Not support GGUF format yet.");
  }

  // Dequant GPTQ weight
  std::unordered_map<std::string, Tensor> host_gptq_weights;
  for (auto& [origin_file_weight_name, host_weight_tensor] : host_model_weights) {
    std::string file_weight_name = origin_file_weight_name;
    // replace weight name with condition
    if (deepseek_v3_config->ContainGptqWeights() && deepseek_v3_config->IsWeightMatchGptq(file_weight_name)) {
      for (const auto& [pattern, format] : deepseek_v3_config->w4a8_patterns_) {
        if (std::regex_search(origin_file_weight_name, pattern)) {
          file_weight_name = std::regex_replace(origin_file_weight_name, pattern, format);
          break;
        }
      }
    }

    // for GPTQ weight or moe-int4(mixed with fp8) weight, save to host_gptq_weights and load later
    if (deepseek_v3_config->ContainGptqWeights() && deepseek_v3_config->IsWeightMatchGptq(file_weight_name) &&
        deepseek_v3_config->IsGptqContain(file_weight_name)) {
      host_gptq_weights.emplace(file_weight_name, host_weight_tensor);
      continue;
    }

    // W4A8 KLX expert weights (.weight and .weight_scale): collect for LoadInt4QuantWeight path.
    // This path uses dev_tensor.shape[1] for pitch calculation, correctly handling INT4 packing.
    if (deepseek_v3_config->quant_config.method == QUANT_W4A8_KLX &&
        file_weight_name.find(".experts.") != std::string::npos) {
      host_gptq_weights.emplace(file_weight_name, host_weight_tensor);
      continue;
    }

    KLLM_LOG_DEBUG << fmt::format("Dev_rank: {}, processing weight: {}, shape: {}, dtype: {}", dev_rank,
                                  file_weight_name, Vector2Str(std::vector<size_t>(host_weight_tensor.shape)),
                                  host_weight_tensor.dtype);

    // W4A8 KLX: non-expert weight_scale tensors are loaded directly to device here.
    // Skip further processing to avoid mismatching with weight logic (since "weight_scale"
    // contains "weight" substring and would be incorrectly matched by find(".xxx.weight")).
#ifdef ENABLE_KUNLUN
    if (deepseek_v3_config->quant_config.method == QUANT_W4A8_KLX &&
        file_weight_name.find("weight_scale") != std::string::npos) {
      LoadW4A8KlxWeightScale(file_weight_name, host_weight_tensor, device_model_weights, deepseek_v3_config,
                             attn_dev_rank, head_num_tp, qk_nope_head_dim, v_head_dim);
      continue;
    }
#endif  // ENABLE_KUNLUN

    // 1. MOE layer
    // Instructions for loading MoE model weights:
    // For each layer of the model, the experts at the same positions
    // of up and gate need to be concatenated and named as up_gate.weight.
    // All experts corresponding to up_gate and down in each layer need to be stacked into one expert weight.
    if (file_weight_name.find(".experts.") != std::string::npos) {
      STATUS_CHECK_RETURN(weight_impl_->GetExpertsIdx(file_weight_name, layer_idx, expert_idx));
      if (layer_idx < 0 || expert_idx < 0) {
        continue;
      }
      int expert_idx_ = expert_map[layer_idx][expert_idx];
      if (expert_idx_ < 0) {
        // Skip load weight when the expert_id will be not used in current rank.
        continue;
      }
#ifdef ENABLE_FP8
      if (deepseek_v3_config->quant_config.is_fp8_blockwise &&
          weight_impl_->LoadMoeFp8E4m3BlockWiseScale(file_weight_name, host_weight_tensor, dev_rank, deepseek_v3_config,
                                                     device_model_weights, expert_idx_)) {
        continue;
      }
#endif
      if (deepseek_v3_config->quant_config.method == QUANT_FP8_E4M3 &&
          (file_weight_name.find("input_scale") != std::string::npos ||
           file_weight_name.find("weight_scale") != std::string::npos)) {
        continue;
      }

      if (file_weight_name.find(".up_proj.") != std::string::npos ||
          file_weight_name.find(".gate_proj.") != std::string::npos) {
        std::string up_gate_experts_name =
            "model.layers." + std::to_string(layer_idx) + ".mlp.experts.up_gate_proj.weight";
        if (device_model_weights.find(up_gate_experts_name) == device_model_weights.end()) {
          device_model_weights[up_gate_experts_name] =
              Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, up_gate_experts_shape, nullptr,
                     &(context_->device->GetMemoryManageStream()));
        }

        size_t expert_pitch = moe_inter_size_per_rank * hidden_units * GetTypeSize(host_weight_tensor.dtype);
        size_t double_expert_pitch = expert_pitch * 2;
        size_t src_upgate_offset = deepseek_v3_config->moe_tensor_para_size > 1
                                       ? (dev_rank / expert_parallel_config.expert_para_size) * expert_pitch
                                       : 0;

        Tensor& up_gate_experts_tensor = device_model_weights.at(up_gate_experts_name);
        if (file_weight_name.find(".up_proj.") != std::string::npos) {
          MemcpyAsync(up_gate_experts_tensor.GetPtr<void>() + static_cast<size_t>(expert_idx_) * double_expert_pitch +
                          (use_vllm_moe ? expert_pitch : 0),
                      host_weight_tensor.GetPtr<void>() + src_upgate_offset, expert_pitch, MEMCPY_HOST_TO_DEVICE,
                      context_->device->GetMemoryManageStream());
        } else if (file_weight_name.find(".gate_proj.") != std::string::npos) {
          MemcpyAsync(up_gate_experts_tensor.GetPtr<void>() + static_cast<size_t>(expert_idx_) * double_expert_pitch +
                          (use_vllm_moe ? 0 : expert_pitch),
                      host_weight_tensor.GetPtr<void>() + src_upgate_offset, expert_pitch, MEMCPY_HOST_TO_DEVICE,
                      context_->device->GetMemoryManageStream());
        }
        KLLM_LOG_DEBUG << fmt::format(
            "Dev_rank: {}, [MoE up_gate] stored: {}, shape: {}, dtype: {}", dev_rank, up_gate_experts_name,
            Vector2Str(std::vector<size_t>(up_gate_experts_tensor.shape)), up_gate_experts_tensor.dtype);
        continue;
      }

      if (file_weight_name.find(".down_proj.") != std::string::npos) {
        std::string down_experts_name = "model.layers." + std::to_string(layer_idx) + ".mlp.experts.down_proj.weight";
        if (device_model_weights.find(down_experts_name) == device_model_weights.end()) {
          device_model_weights[down_experts_name] =
              Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, down_experts_shape, nullptr,
                     &(context_->device->GetMemoryManageStream()));
        }

        size_t dst_pitch = moe_inter_size_per_rank * GetTypeSize(host_weight_tensor.dtype);
        size_t src_pitch =
            moe_inter_size_per_rank * deepseek_v3_config->moe_tensor_para_size * GetTypeSize(host_weight_tensor.dtype);
        size_t expert_pitch = moe_inter_size_per_rank * hidden_units * GetTypeSize(host_weight_tensor.dtype);
        size_t src_down_offset = deepseek_v3_config->moe_tensor_para_size > 1
                                     ? (dev_rank / expert_parallel_config.expert_para_size) * dst_pitch
                                     : 0;
        Tensor& down_expert_tensor = device_model_weights.at(down_experts_name);
        Memcpy2DAsync(down_expert_tensor.GetPtr<void>() + static_cast<size_t>(expert_idx_) * expert_pitch, dst_pitch,
                      host_weight_tensor.GetPtr<void>() + src_down_offset, src_pitch, dst_pitch, hidden_units,
                      MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());
        KLLM_LOG_DEBUG << fmt::format("Dev_rank: {}, [MoE down] stored: {}, shape: {}, dtype: {}", dev_rank,
                                      down_experts_name, Vector2Str(std::vector<size_t>(down_expert_tensor.shape)),
                                      down_expert_tensor.dtype);
        continue;
      }
    }

    // 2. MLA layer
    const std::string fused_lora_a_weight_name = ".fused_lora_a_proj.weight";
    if (file_weight_name.find("self_attn") != std::string::npos &&
        file_weight_name.find("norm.") == std::string::npos) {
#ifdef ENABLE_FP8
      if (deepseek_v3_config->quant_config.is_fp8_blockwise &&
          weight_impl_->LoadMlaFp8E4m3BlockWiseScale(file_weight_name, host_weight_tensor, dev_rank, deepseek_v3_config,
                                                     device_model_weights)) {
        continue;
      }
#endif
      // q_proj is for deepseek v2, q_b_proj is for deepseek v3
      if (file_weight_name.find(".q_proj.weight") != std::string::npos ||
          file_weight_name.find(".q_b_proj.weight") != std::string::npos) {
        // 3072 is deepseek v2
        if (host_weight_tensor.shape[0] != 3072 &&
            (qk_nope_head_dim + qk_rope_head_dim) * head_num != host_weight_tensor.shape[0]) {
          KLLM_THROW(fmt::format(
              "The shape of the 0th dim of the weight named '{} ({})' is not equal to the sum of qk_nope_head_dim {} "
              "and qk_rope_head_dim {}.",
              file_weight_name, host_weight_tensor.shape[0], qk_nope_head_dim, qk_rope_head_dim));
        }

        if (!deepseek_v3_config->quant_config.is_fp8_blockwise) {
          std::vector<size_t> q_b_nope_rope_shape = {
              static_cast<size_t>(DivRoundUp(head_num * (qk_nope_head_dim + qk_rope_head_dim), attn_tp_size)),
              host_weight_tensor.shape[1]};
          Tensor q_b_nope_rope_tensor =
              Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, q_b_nope_rope_shape, nullptr,
                     &(context_->device->GetMemoryManageStream()));
          size_t para_pitch = DivRoundUp(head_num, attn_tp_size) * (qk_nope_head_dim + qk_rope_head_dim) *
                              host_weight_tensor.shape[1] * GetTypeSize(host_weight_tensor.dtype);
          size_t tensor_para_offset = attn_dev_rank * para_pitch;
          MemcpyAsync(q_b_nope_rope_tensor.GetPtr<void>(), host_weight_tensor.GetPtr<void>() + tensor_para_offset,
                      para_pitch, MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());

          if (!fc_use_transb_layout_) {
            weight_impl_->PermuteWeight(q_b_nope_rope_tensor, {1, 0}, dev_rank);
          }

          std::string q_b_nope_rope_name =
              file_weight_name.substr(0, file_weight_name.find_first_of('_')) + "_attn.q_b_nope_rope_proj.weight";
          device_model_weights[q_b_nope_rope_name] = q_b_nope_rope_tensor;
          KLLM_LOG_DEBUG << fmt::format("Dev_rank: {}, [MLA q_b_nope_rope] stored: {}, shape: {}, dtype: {}", dev_rank,
                                        q_b_nope_rope_name, Vector2Str(std::vector<size_t>(q_b_nope_rope_tensor.shape)),
                                        q_b_nope_rope_tensor.dtype);
        } else {
          // For fp8 blockwise quant, do not split the weights initially, split them after dequantization later.
          std::string q_b_proj_name =
              file_weight_name.substr(0, file_weight_name.find_first_of('_')) + "_attn.q_b_proj.weight";
          std::vector<size_t> q_b_proj_shape = {
              static_cast<size_t>(DivRoundUp(host_weight_tensor.shape[0], attn_tp_size)), host_weight_tensor.shape[1]};
          Tensor q_b_proj_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, q_b_proj_shape,
                                          nullptr, &(context_->device->GetMemoryManageStream()));

          size_t para_pitch = DivRoundUp(head_num, attn_tp_size) * (qk_nope_head_dim + qk_rope_head_dim) *
                              host_weight_tensor.shape[1] * GetTypeSize(host_weight_tensor.dtype);
          size_t tensor_para_offset = attn_dev_rank * para_pitch;
          MemcpyAsync(q_b_proj_tensor.GetPtr<void>(), host_weight_tensor.GetPtr<void>() + tensor_para_offset,
                      para_pitch, MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());

          device_model_weights[q_b_proj_name] = q_b_proj_tensor;
          KLLM_LOG_DEBUG << fmt::format("Dev_rank: {}, [MLA q_b_proj fp8] stored: {}, shape: {}, dtype: {}", dev_rank,
                                        q_b_proj_name, Vector2Str(std::vector<size_t>(q_b_proj_tensor.shape)),
                                        q_b_proj_tensor.dtype);
        }
        continue;
      }
      // q_a_proj is for deepseek v3
      if (file_weight_name.find(".q_a_proj.weight") != std::string::npos) {
        // Weights are not split and are copied to each GPU (fused with kv_a_proj_with_mqa.weight).
        // fp16/bf16: weights needs to transpose
        std::string fused_tensor_name = GetReplacedName(file_weight_name, ".q_a_proj.weight", fused_lora_a_weight_name);
        Tensor fused_tensor;
        if (device_model_weights.find(fused_tensor_name) != device_model_weights.end()) {
          fused_tensor = device_model_weights.at(fused_tensor_name);
        } else {
          fused_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype,
                                {q_lora_rank + kv_lora_rank + qk_rope_head_dim, host_weight_tensor.shape[1]}, nullptr,
                                &(context_->device->GetMemoryManageStream()));
        }
        MemcpyAsync(fused_tensor.GetPtr<void>(), host_weight_tensor.GetPtr<void>(), host_weight_tensor.GetTotalBytes(),
                    MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());
        if (device_model_weights.find(fused_tensor_name) == device_model_weights.end()) {
          device_model_weights[fused_tensor_name] = fused_tensor;
        }
        if (!fc_use_transb_layout_ &&
            (fused_tensor.dtype == DataType::TYPE_FP16 || fused_tensor.dtype == DataType::TYPE_BF16)) {
          weights_to_permute_[dev_rank].insert(fused_tensor_name);
        }
        KLLM_LOG_DEBUG << fmt::format("Dev_rank: {}, [MLA q_a_proj fused] stored: {}, shape: {}, dtype: {}", dev_rank,
                                      fused_tensor_name, Vector2Str(std::vector<size_t>(fused_tensor.shape)),
                                      fused_tensor.dtype);
        continue;
      }
      if (file_weight_name.find(".kv_a_proj_with_mqa.weight") != std::string::npos) {
        if ((kv_lora_rank + qk_rope_head_dim) != host_weight_tensor.shape[0]) {
          KLLM_THROW(fmt::format(
              "The shape of the 0th dim of the weight named `{}` is not equal to the sum of kv_lora_rank {} "
              "and qk_rope_head_dim {}.",
              file_weight_name, kv_lora_rank, qk_rope_head_dim));
        }
        std::string fused_tensor_name =
            GetReplacedName(file_weight_name, ".kv_a_proj_with_mqa.weight", fused_lora_a_weight_name);
        Tensor fused_tensor;
        if (device_model_weights.find(fused_tensor_name) != device_model_weights.end()) {
          fused_tensor = device_model_weights.at(fused_tensor_name);
        } else {
          fused_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype,
                                {q_lora_rank + kv_lora_rank + qk_rope_head_dim, host_weight_tensor.shape[1]}, nullptr,
                                &(context_->device->GetMemoryManageStream()));
        }
        const int offset = q_lora_rank * host_weight_tensor.shape[1] * host_weight_tensor.GetDTypeSize();
        MemcpyAsync(fused_tensor.GetPtr<void>() + offset, host_weight_tensor.GetPtr<void>(),
                    host_weight_tensor.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
                    context_->device->GetMemoryManageStream());
        if (device_model_weights.find(fused_tensor_name) == device_model_weights.end()) {
          device_model_weights[fused_tensor_name] = fused_tensor;
        }
        if (!fc_use_transb_layout_ &&
            (fused_tensor.dtype == DataType::TYPE_FP16 || fused_tensor.dtype == DataType::TYPE_BF16)) {
          weights_to_permute_[dev_rank].insert(fused_tensor_name);
        }
        KLLM_LOG_DEBUG << fmt::format("Dev_rank: {}, [MLA kv_a fused] stored: {}, shape: {}, dtype: {}", dev_rank,
                                      fused_tensor_name, Vector2Str(std::vector<size_t>(fused_tensor.shape)),
                                      fused_tensor.dtype);
        continue;
      }
      if (file_weight_name.find(".kv_b_proj.weight") != std::string::npos) {
        LoadMlaKvBProj(file_weight_name, host_weight_tensor, dev_rank, deepseek_v3_config, device_model_weights,
                       head_num, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, kv_lora_rank, attn_tp_size,
                       attn_dev_rank, head_num_tp);
        continue;
      }
      if (file_weight_name.find(".o_proj.weight") != std::string::npos) {
        // bf16/fp16: Transpose, then split along axis = 0
        // fp8: Transpose, then split along axis = 0, then transpose
        size_t split_size =
            runtime_config_.enable_o_proj_out_of_dp ? deepseek_v3_config->tensor_para_size : attn_tp_size;
        Tensor dev_tensor;
        bool is_this_weight_quant = deepseek_v3_config->is_quant && !deepseek_v3_config->is_compressed_tensors;
        weight_impl_->TransSplitOptTrans(host_weight_tensor, dev_tensor, dev_rank, split_size,
                                         is_this_weight_quant || fc_use_transb_layout_);

        device_model_weights[file_weight_name] = dev_tensor;
        KLLM_LOG_DEBUG << fmt::format("Dev_rank: {}, [MLA o_proj] stored: {}, shape: {}, dtype: {}", dev_rank,
                                      file_weight_name, Vector2Str(std::vector<size_t>(dev_tensor.shape)),
                                      dev_tensor.dtype);
        continue;
      }
    }

    // 2. dense MLP layer and shared expert layer
    // The parameters of both the mlp (dense) layer and the shared expert layer need to be transposed.
    if (CheckWeightNameMatched(
            file_weight_name,
            {".mlp.gate_proj.", ".mlp.up_proj.", ".mlp.shared_expert.gate_proj", ".mlp.shared_expert.up_proj"},
            false)) {
      // "up && gate proj(bf16/fp16): First split along axis = 0, then transpose."
      // "up && gate proj(fp8 weight/fp32 weight_scale_inv): split along axis = 0."
      Tensor dev_tensor;
      size_t mlp_tensor_para_size =
          runtime_config_.enable_full_shared_expert ? 1 : context_->global->GetTensorParallelSize();
      // For compressed-tensors (moe-int4), dense/shared_expert weights are NOT quantized (bf16/fp16).
      // Use actual weight dtype to determine if transpose is needed, not global is_quant.
      bool is_this_weight_quant = deepseek_v3_config->is_quant && !deepseek_v3_config->is_compressed_tensors;
      weight_impl_->SplitOptTrans(host_weight_tensor, dev_tensor, dev_rank, mlp_tensor_para_size,
                                  !is_this_weight_quant && !fc_use_transb_layout_);
      if (deepseek_v3_config->use_mla && deepseek_v3_config->type != "deepseek_v2") {
        // deepseek v3 need to combine gate & up proj
        weight_impl_->ProcessGateUpProjWeight(file_weight_name, dev_tensor, device_model_weights, dev_rank,
                                              is_this_weight_quant);
        KLLM_LOG_DEBUG << fmt::format("Dev_rank: {}, [MLP gate_up fused] processed: {}, shape: {}, dtype: {}", dev_rank,
                                      file_weight_name, Vector2Str(std::vector<size_t>(dev_tensor.shape)),
                                      dev_tensor.dtype);
      } else {
        device_model_weights[file_weight_name] = dev_tensor;
        KLLM_LOG_DEBUG << fmt::format("Dev_rank: {}, [MLP gate/up] stored: {}, shape: {}, dtype: {}", dev_rank,
                                      file_weight_name, Vector2Str(std::vector<size_t>(dev_tensor.shape)),
                                      dev_tensor.dtype);
      }
      continue;
    }

    if (CheckWeightNameMatched(file_weight_name,
                               {
                                   ".mlp.down_proj.",
                                   ".mlp.shared_expert.down_proj",
                               },
                               false)) {
      // down proj(bf16/fp16): transpose first, then split along axis = 0
      // down proj(fp8/fp32): transpose first, then split along axis = 0, and then transpose back.
      Tensor dev_tensor;
      size_t mlp_tensor_para_size =
          runtime_config_.enable_full_shared_expert ? 1 : context_->global->GetTensorParallelSize();
      bool is_this_weight_quant = deepseek_v3_config->is_quant && !deepseek_v3_config->is_compressed_tensors;
      weight_impl_->TransSplitOptTrans(host_weight_tensor, dev_tensor, dev_rank, mlp_tensor_para_size,
                                       is_this_weight_quant || fc_use_transb_layout_);
      device_model_weights[file_weight_name] = dev_tensor;
      KLLM_LOG_DEBUG << fmt::format("Dev_rank: {}, [MLP down] stored: {}, shape: {}, dtype: {}", dev_rank,
                                    file_weight_name, Vector2Str(std::vector<size_t>(dev_tensor.shape)),
                                    dev_tensor.dtype);
      continue;
    }

    // 5. norm layer/`gate.e_score_correction_bias`/indexer fp8 weight
    // Directly load to each device.
    if (file_weight_name.find("norm.") != std::string::npos ||
        file_weight_name.find("gate.e_score_correction_bias") != std::string::npos ||
        file_weight_name.find("self_attn.indexer.wk") != std::string::npos ||
        file_weight_name.find("self_attn.indexer.wq_b") != std::string::npos) {
#ifdef ENABLE_KUNLUN
      // Kunlun moe_sigmoid_topk_norm_fusion requires e_score_correction_bias to be FP32.
      if (file_weight_name.find("gate.e_score_correction_bias") != std::string::npos &&
          host_weight_tensor.dtype != DataType::TYPE_FP32) {
        size_t num_elements = host_weight_tensor.GetTotalBytes() / GetTypeSize(host_weight_tensor.dtype);
        Tensor fp32_tensor(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP32, {num_elements});
        const void* src = host_weight_tensor.GetPtr<void>();
        float* dst = fp32_tensor.GetPtr<float>();
        if (host_weight_tensor.dtype == DataType::TYPE_BF16) {
          const c10::BFloat16* bf16_src = reinterpret_cast<const c10::BFloat16*>(src);
          for (size_t i = 0; i < num_elements; ++i) {
            dst[i] = static_cast<float>(bf16_src[i]);
          }
        } else if (host_weight_tensor.dtype == DataType::TYPE_FP16) {
          const c10::Half* fp16_src = reinterpret_cast<const c10::Half*>(src);
          for (size_t i = 0; i < num_elements; ++i) {
            dst[i] = static_cast<float>(fp16_src[i]);
          }
        }
        Tensor dev_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_FP32, host_weight_tensor.shape,
                                   nullptr, &(context_->device->GetMemoryManageStream()));
        MemcpyAsync(dev_tensor.GetPtr<void>(), fp32_tensor.GetPtr<void>(), num_elements * sizeof(float),
                    MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());
        device_model_weights[file_weight_name] = dev_tensor;
        continue;
      }
#endif
      Tensor dev_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, host_weight_tensor.shape,
                                 nullptr, &(context_->device->GetMemoryManageStream()));
      MemcpyAsync(dev_tensor.GetPtr<void>(), host_weight_tensor.GetPtr<void>(), host_weight_tensor.GetTotalBytes(),
                  MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());

      device_model_weights[file_weight_name] = dev_tensor;
      KLLM_LOG_DEBUG << fmt::format("Dev_rank: {}, [norm/bias/indexer] stored: {}, shape: {}, dtype: {}", dev_rank,
                                    file_weight_name, Vector2Str(std::vector<size_t>(dev_tensor.shape)),
                                    dev_tensor.dtype);
      continue;
    }

    // 6. gate weight/indexer bf16 weight
    // Copy to each device and transpose
    if (file_weight_name.find(".mlp.gate.weight") != std::string::npos ||
        file_weight_name.find("self_attn.indexer.weights_proj") != std::string::npos) {
      Tensor dev_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, host_weight_tensor.shape,
                                 nullptr, &(context_->device->GetMemoryManageStream()));

      MemcpyAsync(dev_tensor.GetPtr<void>(), host_weight_tensor.GetPtr<void>(), host_weight_tensor.GetTotalBytes(),
                  MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());
      if (!fc_use_transb_layout_) {
        weight_impl_->PermuteWeight(dev_tensor, {1, 0}, dev_rank);
      }
      device_model_weights[file_weight_name] = dev_tensor;
      KLLM_LOG_DEBUG << fmt::format("Dev_rank: {}, [gate/indexer permute] stored: {}, shape: {}, dtype: {}", dev_rank,
                                    file_weight_name, Vector2Str(std::vector<size_t>(dev_tensor.shape)),
                                    dev_tensor.dtype);
      continue;
    }

    // 7. eh_proj.weight (for mtp layer)
    // Split along axis = 0
    if (file_weight_name.find(".eh_proj.weight") != std::string::npos) {
      Tensor dev_tensor;
      weight_impl_->SplitOptTrans(host_weight_tensor, dev_tensor, dev_rank, context_->global->GetTensorParallelSize(),
                                  !fc_use_transb_layout_);
      device_model_weights[file_weight_name] = dev_tensor;
      KLLM_LOG_DEBUG << fmt::format("Dev_rank: {}, [eh_proj] stored: {}, shape: {}, dtype: {}", dev_rank,
                                    file_weight_name, Vector2Str(std::vector<size_t>(dev_tensor.shape)),
                                    dev_tensor.dtype);
      continue;
    }

    // 3. model.embed_tokens.weight;
    // ::FilterWeightNames() filtered this for non master nodes
    // Embedding TP needs to be transposed first, then split, and then transposed back.
    if (CheckWeightNameMatched(file_weight_name, {"model.embed_tokens.weight"}, true)) {
      Tensor dev_tensor;
      weight_impl_->TransSplitOptTrans(host_weight_tensor, dev_tensor, dev_rank,
                                       context_->global->GetTensorParallelSize(), true);

      device_model_weights[file_weight_name] = dev_tensor;
      KLLM_LOG_DEBUG << fmt::format("Dev_rank: {}, [embed_tokens] stored: {}, shape: {}, dtype: {}", dev_rank,
                                    file_weight_name, Vector2Str(std::vector<size_t>(dev_tensor.shape)),
                                    dev_tensor.dtype);
      continue;
    }

    // lm_head.weight
    // lm_head need to split along axis = 0, and then transpose
    if (CheckWeightNameMatched(file_weight_name, {"lm_head.weight"}, true)) {
      Tensor permute_dev_tensor;
      weight_impl_->SplitOptTrans(host_weight_tensor, permute_dev_tensor, dev_rank,
                                  context_->global->GetTensorParallelSize(), !fc_use_transb_layout_);
      device_model_weights[file_weight_name] = permute_dev_tensor;
      KLLM_LOG_DEBUG << fmt::format("Dev_rank: {}, [lm_head] stored: {}, shape: {}, dtype: {}", dev_rank,
                                    file_weight_name, Vector2Str(std::vector<size_t>(permute_dev_tensor.shape)),
                                    permute_dev_tensor.dtype);
      continue;
    }
  }

  // Load gptq weights or W4A8_KLX expert weights here
  if (deepseek_v3_config->ContainGptqWeights() || deepseek_v3_config->quant_config.method == QUANT_W4A8_KLX) {
    weight_impl_->LoadInt4QuantWeight(host_gptq_weights, dev_rank, device_model_weights, deepseek_v3_config,
                                      expert_map);
  }

  return Status();
}

Status DeepSeekV3WeightLoader::InitWeightLoaderImpl(DataType weight_data_type) {
  switch (weight_data_type) {
    case DataType::TYPE_FP32:
      weight_impl_ = std::make_unique<DeepSeekV3WeightImpl<float>>(context_, runtime_config_);
      break;
    case DataType::TYPE_BF16:
      weight_impl_ = std::make_unique<DeepSeekV3WeightImpl<bfloat16>>(context_, runtime_config_);
      break;
    case DataType::TYPE_FP16:
      weight_impl_ = std::make_unique<DeepSeekV3WeightImpl<float16>>(context_, runtime_config_);
      break;
    default:
      // Handle unexpected data type if needed
      KLLM_THROW(fmt::format("Unexpected data type: {}", weight_data_type));
      break;
  }
  return Status();
}

Status DeepSeekV3WeightLoader::InitExpertMap(const size_t& num_experts, const size_t& expert_start_id,
                                             const size_t& expert_end_id, int dev_rank,
                                             std::vector<std::vector<int>>& expert_map,
                                             std::unordered_map<std::string, Tensor>& device_model_weights) {
  // Load EPLB mapping configuration from JSON file
  std::unordered_map<int, std::vector<int>> eplb_map;
  if (runtime_config_.enable_load_eplb_config) {
    nlohmann::json json_config = ReadJsonFromFile(runtime_config_.eplb_config_path);
    if (json_config.empty()) {
      KLLM_LOG_ERROR << fmt::format("Failed to load EPLB config file: {}. File not found or empty.",
                                    runtime_config_.eplb_config_path);
    } else {
      for (auto& [key, value] : json_config.items()) {
        if (key.substr(0, 6) == "layer_" && value.is_array()) {
          int layer_idx = std::stoi(key.substr(6));  // 提取layer_后面的数字
          std::vector<int> eplb_data = value.get<std::vector<int>>();
          if (eplb_data.size() != num_experts) {
            KLLM_LOG_ERROR << fmt::format(
                "EPLB config file {} load failed in {}: wrong expert nums {} vs {}(actual model expert nums)",
                runtime_config_.eplb_config_path, key, eplb_data.size(), num_experts);
          } else {
            eplb_map[layer_idx] = eplb_data;
          }
        }
      }
      KLLM_LOG_DEBUG << fmt::format("Successed to load EPLB map from {}.", runtime_config_.eplb_config_path);
    }
  }

  int layer_num = static_cast<int>(expert_map.size());
  if (!eplb_map.empty()) {
    // 摊平 expert_map 到一维，以拷贝到 Device 空间
    std::vector<int> full_expert_map(static_cast<size_t>(layer_num) * num_experts);
    for (int layer_idx = 0; layer_idx < layer_num; layer_idx++) {
      for (size_t i = 0; i < num_experts; ++i) {
        if (eplb_map.find(layer_idx) != eplb_map.end()) {
          if (i >= expert_start_id && i < expert_end_id) {
            expert_map[layer_idx][eplb_map[layer_idx][i]] = static_cast<int>(i - expert_start_id);
          }
          full_expert_map[layer_idx * num_experts + eplb_map[layer_idx][i]] = static_cast<int>(i);
        } else {
          if (i >= expert_start_id && i < expert_end_id) {
            expert_map[layer_idx][i] = static_cast<int>(i - expert_start_id);
          }
          full_expert_map[layer_idx * num_experts + i] = static_cast<int>(i);
        }
      }
    }
    std::string expert_map_name = "expert_map";
    Tensor dev_tensor =
        Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {static_cast<size_t>(layer_num), num_experts}, nullptr,
               &(context_->device->GetMemoryManageStream()));
    MemcpyAsync(dev_tensor.GetPtr<void>(), full_expert_map.data(), full_expert_map.size() * sizeof(int),
                MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());
    device_model_weights[expert_map_name] = dev_tensor;
  } else {
    // Without eplb map
    for (int layer_idx = 0; layer_idx < layer_num; layer_idx++) {
      for (size_t i = expert_start_id; i < expert_end_id; ++i) {
        expert_map[layer_idx][i] = static_cast<int>(i - expert_start_id);
      }
    }
    Tensor dev_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {0}, nullptr,
                               &(context_->device->GetMemoryManageStream()), "expert_map");
    device_model_weights["expert_map"] = dev_tensor;
  }
  return Status();
}

}  // namespace ksana_llm
