/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/kimi_k3/kimi_k3_weight_loader.h"

#include <regex>

#include "configure/json_config_utils.h"
#include "model_loader/model_loader_utils.h"
#include "utils/logger.h"
#include "utils/memory_utils.h"
#include "utils/string_utils.h"

namespace ksana_llm {

KimiK3WeightLoader::KimiK3WeightLoader(std::shared_ptr<BaseModelConfig> model_config, std::shared_ptr<Environment> env,
                                       std::shared_ptr<Context> context)
    : DeepSeekV3WeightLoader(model_config, env, context) {
  // KimiK3Config inherits DeepSeekV3Config, so the parent ctor already initialized
  // weight_impl_ / pipeline_config_ / runtime_config_ / weights_to_permute_ correctly.
  kimi_k3_config_ = std::dynamic_pointer_cast<KimiK3Config>(model_config_);
}

KimiK3WeightLoader::~KimiK3WeightLoader() {}

Status KimiK3WeightLoader::FilterWeightNames(std::vector<std::string>& weight_names) {
  // For now, skip the multimodal vision tower and rope inv_freq. The vision tower will be
  // handled in a later multimodal stage.
  const std::vector<std::string> skip_list = {"self_attn.rotary_emb.inv_freq", "vision_tower.", "mm_projector.",
                                              "rotary_emb.inv_freq"};

  const int lower_layer_idx = pipeline_config_.lower_layer_idx;
  const int upper_layer_idx = pipeline_config_.upper_layer_idx;

  for (auto it = weight_names.begin(); it != weight_names.end();) {
    if (CheckWeightNameMatched(*it, skip_list, false)) {
      it = weight_names.erase(it);
      continue;
    }

    const int layer_idx = GetLayerIdxFromName(*it);

    // Skip some layers in distributed (pipeline parallel) mode.
    if (lower_layer_idx >= 0 && upper_layer_idx >= 0) {
      if (layer_idx >= 0 && (layer_idx < lower_layer_idx || layer_idx > upper_layer_idx)) {
        it = weight_names.erase(it);
        continue;
      }
    }

    if (!context_->global->IsStandalone()) {
      // Skip embedding if not master mode.
      if (context_->global->IsDistributedWorker()) {
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

void KimiK3WeightLoader::CopyReplicated(const std::string& weight_name, const Tensor& host_weight_tensor, int dev_rank,
                                        std::unordered_map<std::string, Tensor>& device_model_weights, bool transpose) {
  Tensor dev_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, host_weight_tensor.shape,
                             nullptr, &(context_->device->GetMemoryManageStream()));
  MemcpyAsync(dev_tensor.GetPtr<void>(), host_weight_tensor.GetPtr<void>(), host_weight_tensor.GetTotalBytes(),
              MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());
  if (transpose) {
    weight_impl_->PermuteWeight(dev_tensor, {1, 0}, dev_rank);
  }
  device_model_weights[weight_name] = dev_tensor;
  KLLM_LOG_INFO << fmt::format("Dev_rank: {}, [replicated] stored: {}, shape: {}, dtype: {}", dev_rank, weight_name,
                               Vector2Str(std::vector<size_t>(dev_tensor.shape)), dev_tensor.dtype);
}

void KimiK3WeightLoader::ColumnSplitAttn(const std::string& weight_name, const Tensor& host_weight_tensor, int dev_rank,
                                         std::unordered_map<std::string, Tensor>& device_model_weights) {
  // Column parallel: split dim0 following attn_tp, then transpose (bf16/fp16 layout) like deepseek_v3.
  size_t attn_tp_size = context_->global->GetAttentionTensorParallelSize();
  Tensor dev_tensor;
  weight_impl_->SplitOptTrans(host_weight_tensor, dev_tensor, dev_rank, attn_tp_size, /*transpose=*/true);
  device_model_weights[weight_name] = dev_tensor;
  KLLM_LOG_INFO << fmt::format("Dev_rank: {}, [KDA col-split] stored: {}, shape: {}, dtype: {}", dev_rank, weight_name,
                               Vector2Str(std::vector<size_t>(dev_tensor.shape)), dev_tensor.dtype);
}

bool KimiK3WeightLoader::ProcessMlaWeight(const std::string& file_weight_name, const Tensor& host_weight_tensor,
                                          int dev_rank, std::unordered_map<std::string, Tensor>& device_model_weights) {
  size_t attn_tp_size = context_->global->GetAttentionTensorParallelSize();
  size_t attn_dev_rank = dev_rank % attn_tp_size;
  size_t kv_lora_rank = kimi_k3_config_->mla_config.kv_lora_rank;
  size_t qk_rope_head_dim = kimi_k3_config_->mla_config.qk_rope_head_dim;
  size_t qk_nope_head_dim = kimi_k3_config_->mla_config.qk_nope_head_dim;
  size_t v_head_dim = kimi_k3_config_->mla_config.v_head_dim;
  size_t q_lora_rank = kimi_k3_config_->mla_config.q_lora_rank;
  size_t head_num = kimi_k3_config_->head_num;
  size_t head_num_tp = static_cast<size_t>(DivRoundUp(head_num, attn_tp_size));

  const std::string fused_lora_a_weight_name = ".fused_lora_a_proj.weight";

  // q_a_proj: replicated, fused into fused_lora_a_proj (front). (Reuse deepseek_v3 semantics.)
  if (file_weight_name.find(".q_a_proj.weight") != std::string::npos) {
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
    if (fused_tensor.dtype == DataType::TYPE_FP16 || fused_tensor.dtype == DataType::TYPE_BF16) {
      weights_to_permute_[dev_rank].insert(fused_tensor_name);
    }
    KLLM_LOG_INFO << fmt::format("Dev_rank: {}, [MLA q_a_proj fused] stored: {}, shape: {}, dtype: {}", dev_rank,
                                 fused_tensor_name, Vector2Str(std::vector<size_t>(fused_tensor.shape)),
                                 fused_tensor.dtype);
    return true;
  }

  // kv_a_proj_with_mqa: replicated, fused into fused_lora_a_proj (back, after q_lora_rank).
  if (file_weight_name.find(".kv_a_proj_with_mqa.weight") != std::string::npos) {
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
                host_weight_tensor.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());
    if (device_model_weights.find(fused_tensor_name) == device_model_weights.end()) {
      device_model_weights[fused_tensor_name] = fused_tensor;
    }
    if (fused_tensor.dtype == DataType::TYPE_FP16 || fused_tensor.dtype == DataType::TYPE_BF16) {
      weights_to_permute_[dev_rank].insert(fused_tensor_name);
    }
    KLLM_LOG_INFO << fmt::format("Dev_rank: {}, [MLA kv_a fused] stored: {}, shape: {}, dtype: {}", dev_rank,
                                 fused_tensor_name, Vector2Str(std::vector<size_t>(fused_tensor.shape)),
                                 fused_tensor.dtype);
    return true;
  }

  // q_b_proj -> q_b_nope_rope_proj: column split (dim0) + permute.
  if (file_weight_name.find(".q_b_proj.weight") != std::string::npos) {
    std::vector<size_t> q_b_nope_rope_shape = {
        static_cast<size_t>(DivRoundUp(head_num * (qk_nope_head_dim + qk_rope_head_dim), attn_tp_size)),
        host_weight_tensor.shape[1]};
    Tensor q_b_nope_rope_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, q_b_nope_rope_shape,
                                         nullptr, &(context_->device->GetMemoryManageStream()));
    size_t para_pitch = DivRoundUp(head_num, attn_tp_size) * (qk_nope_head_dim + qk_rope_head_dim) *
                        host_weight_tensor.shape[1] * GetTypeSize(host_weight_tensor.dtype);
    size_t tensor_para_offset = attn_dev_rank * para_pitch;
    MemcpyAsync(q_b_nope_rope_tensor.GetPtr<void>(), host_weight_tensor.GetPtr<void>() + tensor_para_offset, para_pitch,
                MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());
    weight_impl_->PermuteWeight(q_b_nope_rope_tensor, {1, 0}, dev_rank);
    std::string q_b_nope_rope_name =
        file_weight_name.substr(0, file_weight_name.find_first_of('_')) + "_attn.q_b_nope_rope_proj.weight";
    device_model_weights[q_b_nope_rope_name] = q_b_nope_rope_tensor;
    KLLM_LOG_INFO << fmt::format("Dev_rank: {}, [MLA q_b_nope_rope] stored: {}, shape: {}, dtype: {}", dev_rank,
                                 q_b_nope_rope_name, Vector2Str(std::vector<size_t>(q_b_nope_rope_tensor.shape)),
                                 q_b_nope_rope_tensor.dtype);
    return true;
  }

  // kv_b_proj -> kv_b_nope / v_head / w_uk_t / w_uv (reuse deepseek_v3 layout).
  if (file_weight_name.find(".kv_b_proj.weight") != std::string::npos) {
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
    PermuteDeviceTensor(kv_b_nope_tensor, {1, 0}, dev_rank, kv_b_nope_permute);
    device_model_weights[kv_b_nope_name] = kv_b_nope_permute;
    KLLM_LOG_INFO << fmt::format("Dev_rank: {}, [MLA kv_b_nope] stored: {}, shape: {}, dtype: {}", dev_rank,
                                 kv_b_nope_name, Vector2Str(std::vector<size_t>(kv_b_nope_permute.shape)),
                                 kv_b_nope_permute.dtype);

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
    PermuteDeviceTensor(v_head_tensor, {1, 0}, dev_rank, v_head_permute);
    device_model_weights[v_head_name] = v_head_permute;
    KLLM_LOG_INFO << fmt::format("Dev_rank: {}, [MLA v_head] stored: {}, shape: {}, dtype: {}", dev_rank, v_head_name,
                                 Vector2Str(std::vector<size_t>(v_head_permute.shape)), v_head_permute.dtype);

    // w_uk_t (share buffer with kv_b_nope) and w_uv (permuted v_head).
    Tensor w_uk_t_tensor = kv_b_nope_tensor;
    w_uk_t_tensor.shape = {head_num_tp, qk_nope_head_dim, kv_b_nope_tensor.shape[1]};
    std::string w_uk_t_name = kv_b_nope_name.substr(0, kv_b_nope_name.find_first_of('_')) + "_attn.w_uk_t.weight";
    device_model_weights[w_uk_t_name] = w_uk_t_tensor;
    KLLM_LOG_INFO << fmt::format("Dev_rank: {}, [MLA w_uk_t] stored: {}, shape: {}, dtype: {}", dev_rank, w_uk_t_name,
                                 Vector2Str(std::vector<size_t>(w_uk_t_tensor.shape)), w_uk_t_tensor.dtype);

    v_head_tensor.shape = {head_num_tp, v_head_dim, v_head_shape[1]};
    weight_impl_->PermuteWeight(v_head_tensor, {0, 2, 1}, dev_rank);
    std::string w_uv_name = v_head_name.substr(0, v_head_name.find_first_of('_')) + "_attn.w_uv.weight";
    device_model_weights[w_uv_name] = v_head_tensor;
    KLLM_LOG_INFO << fmt::format("Dev_rank: {}, [MLA w_uv] stored: {}, shape: {}, dtype: {}", dev_rank, w_uv_name,
                                 Vector2Str(std::vector<size_t>(v_head_tensor.shape)), v_head_tensor.dtype);
    return true;
  }

  // o_proj: row split (transpose, split dim0, transpose back).
  if (file_weight_name.find(".o_proj.weight") != std::string::npos) {
    Tensor dev_tensor;
    weight_impl_->TransSplitOptTrans(host_weight_tensor, dev_tensor, dev_rank, attn_tp_size, /*transpose=*/false);
    device_model_weights[file_weight_name] = dev_tensor;
    KLLM_LOG_INFO << fmt::format("Dev_rank: {}, [MLA o_proj] stored: {}, shape: {}, dtype: {}", dev_rank,
                                 file_weight_name, Vector2Str(std::vector<size_t>(dev_tensor.shape)), dev_tensor.dtype);
    return true;
  }

  // g_proj (MLA output gate, mla_use_output_gate): column split (dim0).
  if (file_weight_name.find(".g_proj.weight") != std::string::npos) {
    ColumnSplitAttn(file_weight_name, host_weight_tensor, dev_rank, device_model_weights);
    return true;
  }

  return false;
}

bool KimiK3WeightLoader::ProcessKdaWeight(const std::string& file_weight_name, const Tensor& host_weight_tensor,
                                          int dev_rank, std::unordered_map<std::string, Tensor>& device_model_weights) {
  size_t attn_tp_size = context_->global->GetAttentionTensorParallelSize();
  size_t attn_dev_rank = dev_rank % attn_tp_size;
  size_t kda_num_heads = kimi_k3_config_->kimi_linear_config.num_heads;

  // q/k/v/g_proj: fuse into fused_qkvg_proj.weight (4 segments, column split by attn_tp).
  // b_proj / f_b_proj: column split individually (not fused per design — too small to skew GEMM).
  if (CheckWeightNameMatched(file_weight_name, {".q_proj.weight", ".k_proj.weight", ".v_proj.weight", ".g_proj.weight"},
                             false)) {
    // Determine segment index: q=0, k=1, v=2, g=3.
    int seg_idx = -1;
    if (file_weight_name.find(".q_proj.weight") != std::string::npos)
      seg_idx = 0;
    else if (file_weight_name.find(".k_proj.weight") != std::string::npos)
      seg_idx = 1;
    else if (file_weight_name.find(".v_proj.weight") != std::string::npos)
      seg_idx = 2;
    else if (file_weight_name.find(".g_proj.weight") != std::string::npos)
      seg_idx = 3;

    // Extract layer prefix: model.layers.{N}.self_attn
    std::string fused_name = file_weight_name.substr(0, file_weight_name.find(".q_proj.")) + ".fused_qkvg_proj.weight";
    if (seg_idx == 1)
      fused_name = file_weight_name.substr(0, file_weight_name.find(".k_proj.")) + ".fused_qkvg_proj.weight";
    else if (seg_idx == 2)
      fused_name = file_weight_name.substr(0, file_weight_name.find(".v_proj.")) + ".fused_qkvg_proj.weight";
    else if (seg_idx == 3)
      fused_name = file_weight_name.substr(0, file_weight_name.find(".g_proj.")) + ".fused_qkvg_proj.weight";

    size_t dim0 = host_weight_tensor.shape[0];
    size_t dim1 = host_weight_tensor.shape[1];
    size_t dim0_tp = DivRoundUp(dim0, attn_tp_size);
    std::vector<size_t> fused_shape = {4 * dim0_tp, dim1};
    if (device_model_weights.find(fused_name) == device_model_weights.end()) {
      device_model_weights[fused_name] = Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, fused_shape,
                                                nullptr, &(context_->device->GetMemoryManageStream()));
    }
    Tensor& fused = device_model_weights.at(fused_name);
    size_t seg_pitch = dim0_tp * dim1 * GetTypeSize(host_weight_tensor.dtype);
    size_t src_offset = attn_dev_rank * dim0_tp * dim1 * GetTypeSize(host_weight_tensor.dtype);
    size_t copy_bytes = dim0_tp * dim1 * GetTypeSize(host_weight_tensor.dtype);
    MemcpyAsync(fused.GetPtr<void>() + seg_idx * seg_pitch, host_weight_tensor.GetPtr<void>() + src_offset, copy_bytes,
                MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());
    if (fused.dtype == DataType::TYPE_FP16 || fused.dtype == DataType::TYPE_BF16) {
      weights_to_permute_[dev_rank].insert(fused_name);
    }
    KLLM_LOG_INFO << fmt::format("Dev_rank: {}, [KDA qkvg fused seg {}] stored: {}, shape: {}, dtype: {}", dev_rank,
                                 seg_idx, fused_name, Vector2Str(std::vector<size_t>(fused.shape)), fused.dtype);
    return true;
  }

  // b_proj / f_b_proj: column split individually (not fused).
  if (CheckWeightNameMatched(file_weight_name, {".b_proj.weight", ".f_b_proj.weight"}, false)) {
    ColumnSplitAttn(file_weight_name, host_weight_tensor, dev_rank, device_model_weights);
    return true;
  }

  // f_a_proj: replicated (LoRA-style A; output dim too small to split).
  if (file_weight_name.find(".f_a_proj.weight") != std::string::npos) {
    CopyReplicated(file_weight_name, host_weight_tensor, dev_rank, device_model_weights, /*transpose=*/true);
    return true;
  }

  // o_norm: per-head RMSNorm, replicated.
  if (file_weight_name.find(".o_norm.weight") != std::string::npos) {
    CopyReplicated(file_weight_name, host_weight_tensor, dev_rank, device_model_weights, /*transpose=*/false);
    return true;
  }

  // q/k/v_conv1d: fuse into qkv_conv1d.weight (3 segments, column split by attn_tp), dtype forced fp32.
  // checkpoint shape [12288, 1, 4]; dim0 split by attn_tp.
  if (CheckWeightNameMatched(file_weight_name, {".q_conv1d.weight", ".k_conv1d.weight", ".v_conv1d.weight"}, false)) {
    int seg_idx = -1;
    if (file_weight_name.find(".q_conv1d.weight") != std::string::npos)
      seg_idx = 0;
    else if (file_weight_name.find(".k_conv1d.weight") != std::string::npos)
      seg_idx = 1;
    else if (file_weight_name.find(".v_conv1d.weight") != std::string::npos)
      seg_idx = 2;

    std::string prefix = file_weight_name.substr(0, file_weight_name.find(".q_conv1d."));
    if (seg_idx == 1)
      prefix = file_weight_name.substr(0, file_weight_name.find(".k_conv1d."));
    else if (seg_idx == 2)
      prefix = file_weight_name.substr(0, file_weight_name.find(".v_conv1d."));
    std::string fused_name = prefix + ".qkv_conv1d.weight";

    size_t dim0 = host_weight_tensor.shape[0];
    size_t inner = host_weight_tensor.GetElementNumber() / dim0;  // 1 * kernel_size
    size_t dim0_tp = DivRoundUp(dim0, attn_tp_size);
    std::vector<size_t> fused_shape = {3 * dim0_tp, 1, host_weight_tensor.shape[2]};
    if (device_model_weights.find(fused_name) == device_model_weights.end()) {
      device_model_weights[fused_name] = Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, fused_shape,
                                                nullptr, &(context_->device->GetMemoryManageStream()));
    }
    Tensor& fused = device_model_weights.at(fused_name);
    size_t seg_pitch = dim0_tp * inner * GetTypeSize(host_weight_tensor.dtype);
    size_t src_offset = attn_dev_rank * dim0_tp * inner * GetTypeSize(host_weight_tensor.dtype);
    size_t copy_bytes = dim0_tp * inner * GetTypeSize(host_weight_tensor.dtype);
    MemcpyAsync(fused.GetPtr<void>() + seg_idx * seg_pitch, host_weight_tensor.GetPtr<void>() + src_offset, copy_bytes,
                MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());
    // Force fp32 for conv1d weight (mamba-style short conv).
    if (fused.dtype != DataType::TYPE_FP32) {
      CastDeviceTensorType(fused, DataType::TYPE_FP32, dev_rank);
    }
    KLLM_LOG_INFO << fmt::format("Dev_rank: {}, [KDA qkv_conv1d fused seg {}] stored: {}, shape: {}, dtype: {}",
                                 dev_rank, seg_idx, fused_name, Vector2Str(std::vector<size_t>(fused.shape)),
                                 fused.dtype);
    return true;
  }

  // A_log: [head_dim] or [num_heads]; narrow to local heads following attn_tp, forced fp32.
  if (file_weight_name.find(".A_log") != std::string::npos) {
    size_t local_heads = DivRoundUp(kda_num_heads, attn_tp_size);
    std::vector<size_t> a_log_shape = {local_heads};
    Tensor dev_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, a_log_shape, nullptr,
                               &(context_->device->GetMemoryManageStream()));
    size_t offset = attn_dev_rank * local_heads * GetTypeSize(host_weight_tensor.dtype);
    MemcpyAsync(dev_tensor.GetPtr<void>(), host_weight_tensor.GetPtr<void>() + offset,
                local_heads * GetTypeSize(host_weight_tensor.dtype), MEMCPY_HOST_TO_DEVICE,
                context_->device->GetMemoryManageStream());
    if (dev_tensor.dtype != DataType::TYPE_FP32) {
      CastDeviceTensorType(dev_tensor, DataType::TYPE_FP32, dev_rank);
    }
    device_model_weights[file_weight_name] = dev_tensor;
    KLLM_LOG_INFO << fmt::format("Dev_rank: {}, [KDA A_log] stored: {}, shape: {}, dtype: {}", dev_rank,
                                 file_weight_name, Vector2Str(std::vector<size_t>(dev_tensor.shape)), dev_tensor.dtype);
    return true;
  }

  // dt_bias: [12288]; split dim0 following attn_tp, forced fp32.
  if (file_weight_name.find(".dt_bias") != std::string::npos) {
    size_t dim0 = host_weight_tensor.shape[0];
    size_t dim0_tp = DivRoundUp(dim0, attn_tp_size);
    Tensor dev_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, {dim0_tp}, nullptr,
                               &(context_->device->GetMemoryManageStream()));
    size_t offset = attn_dev_rank * dim0_tp * GetTypeSize(host_weight_tensor.dtype);
    MemcpyAsync(dev_tensor.GetPtr<void>(), host_weight_tensor.GetPtr<void>() + offset,
                dim0_tp * GetTypeSize(host_weight_tensor.dtype), MEMCPY_HOST_TO_DEVICE,
                context_->device->GetMemoryManageStream());
    if (dev_tensor.dtype != DataType::TYPE_FP32) {
      CastDeviceTensorType(dev_tensor, DataType::TYPE_FP32, dev_rank);
    }
    device_model_weights[file_weight_name] = dev_tensor;
    KLLM_LOG_INFO << fmt::format("Dev_rank: {}, [KDA dt_bias] stored: {}, shape: {}, dtype: {}", dev_rank,
                                 file_weight_name, Vector2Str(std::vector<size_t>(dev_tensor.shape)), dev_tensor.dtype);
    return true;
  }

  // o_proj: row split (dim1).
  if (file_weight_name.find(".o_proj.weight") != std::string::npos) {
    Tensor dev_tensor;
    weight_impl_->TransSplitOptTrans(host_weight_tensor, dev_tensor, dev_rank, attn_tp_size, /*transpose=*/false);
    device_model_weights[file_weight_name] = dev_tensor;
    KLLM_LOG_INFO << fmt::format("Dev_rank: {}, [KDA o_proj] stored: {}, shape: {}, dtype: {}", dev_rank,
                                 file_weight_name, Vector2Str(std::vector<size_t>(dev_tensor.shape)), dev_tensor.dtype);
    return true;
  }

  return false;
}

bool KimiK3WeightLoader::ProcessLatentMoeAndRouter(const std::string& file_weight_name,
                                                   const Tensor& host_weight_tensor, int dev_rank,
                                                   std::unordered_map<std::string, Tensor>& device_model_weights) {
  // Latent MoE projections and router gate are fully replicated (not quantized).
  // routed_expert_norm: replicated RMSNorm.
  if (file_weight_name.find(".routed_expert_norm.weight") != std::string::npos) {
    CopyReplicated(file_weight_name, host_weight_tensor, dev_rank, device_model_weights, /*transpose=*/false);
    return true;
  }
  // routed_expert_up_proj / routed_expert_down_proj: replicated (must reduce in latent space before norm).
  if (file_weight_name.find(".routed_expert_up_proj.weight") != std::string::npos ||
      file_weight_name.find(".routed_expert_down_proj.weight") != std::string::npos) {
    CopyReplicated(file_weight_name, host_weight_tensor, dev_rank, device_model_weights, /*transpose=*/true);
    return true;
  }
  // Router gate: replicated (each rank needs full routing logits), transposed.
  if (file_weight_name.find(".mlp.gate.weight") != std::string::npos) {
    CopyReplicated(file_weight_name, host_weight_tensor, dev_rank, device_model_weights, /*transpose=*/true);
    return true;
  }
  // gate.e_score_correction_bias: replicated (cast to fp32 in PostProcess).
  if (file_weight_name.find("gate.e_score_correction_bias") != std::string::npos) {
    CopyReplicated(file_weight_name, host_weight_tensor, dev_rank, device_model_weights, /*transpose=*/false);
    return true;
  }
  return false;
}

Status KimiK3WeightLoader::ProcessModelWeights(const std::unordered_map<std::string, Tensor>& host_model_weights,
                                               int dev_rank,
                                               std::unordered_map<std::string, Tensor>& device_model_weights,
                                               std::unordered_map<std::string, Tensor>& left_host_weights) {
  size_t num_experts = kimi_k3_config_->moe_config.num_experts;
  ExpertParallelConfig& expert_parallel_config = kimi_k3_config_->expert_parallel_config;
  size_t expert_node_rank = expert_parallel_config.expert_node_rank;
  size_t global_expert_para_size = expert_parallel_config.expert_world_size * expert_parallel_config.expert_para_size;
  size_t num_experts_per_rank = DivRoundUp(num_experts, global_expert_para_size);

  // Expert range for this rank.
  size_t rank_expert_offset = expert_node_rank * expert_parallel_config.expert_para_size * num_experts_per_rank;
  size_t expert_offset =
      (global_expert_para_size > 1) ? ((dev_rank % kimi_k3_config_->expert_para_size) * num_experts_per_rank) : 0;
  size_t expert_start_id = rank_expert_offset + expert_offset;
  size_t expert_end_id = std::min(num_experts, expert_start_id + num_experts_per_rank);
  size_t num_layers = kimi_k3_config_->num_layer;
  std::vector<std::vector<int>> expert_map(num_layers, std::vector<int>(num_experts, -1));
  InitExpertMap(num_experts, expert_start_id, expert_end_id, dev_rank, expert_map, device_model_weights);

  size_t moe_inter_size_per_rank =
      DivRoundUp(kimi_k3_config_->moe_config.moe_inter_size, kimi_k3_config_->moe_tensor_para_size);
  // Kimi-K3 routed experts run in latent space: expert "hidden" == routed_expert_hidden_size.
  size_t expert_hidden = kimi_k3_config_->moe_config.routed_expert_hidden_size;

  for (auto& [file_weight_name, host_weight_tensor] : host_model_weights) {
    KLLM_LOG_INFO << fmt::format("Dev_rank: {}, processing weight: {}, shape: {}, dtype: {}", dev_rank,
                                 file_weight_name, Vector2Str(std::vector<size_t>(host_weight_tensor.shape)),
                                 host_weight_tensor.dtype);

    int layer_idx = GetLayerIdxFromName(file_weight_name);

    // 1. MoE routed experts (MXFP4 packed weight + weight_scale). Split by EP, then intra-expert TP.
    // After weight_map renaming, expert weights are:
    // model.layers.N.mlp.experts.E.{gate|up|down}_proj.weight(_scale_inv)
    if (file_weight_name.find(".mlp.experts.") != std::string::npos) {
      int32_t elayer = -1, expert_idx = -1;
      weight_impl_->GetExpertsIdx(file_weight_name, elayer, expert_idx);
      if (elayer < 0 || expert_idx < 0) {
        continue;
      }
      int local_expert_idx = expert_map[elayer][expert_idx];
      if (local_expert_idx < 0) {
        // Expert not owned by this rank.
        continue;
      }
      bool is_scale = file_weight_name.find(".weight_scale_inv") != std::string::npos;
      bool is_up = file_weight_name.find(".up_proj.") != std::string::npos;
      bool is_gate = file_weight_name.find(".gate_proj.") != std::string::npos;
      bool is_down = file_weight_name.find(".down_proj.") != std::string::npos;

      // w13 (gate=w1 front, up=w3 back) share one stacked tensor; down is separate.
      // For MXFP4 the packed inner dim is halved (uint8 pack); use actual host tensor inner dim.
      if (is_up || is_gate) {
        size_t inner = host_weight_tensor.shape.size() > 1 ? host_weight_tensor.shape[1] : 1;
        std::string suffix = is_scale ? ".weight_scale_inv" : ".weight";
        std::string w13_name = "model.layers." + std::to_string(elayer) + ".mlp.experts.gate_up_proj" + suffix;
        std::vector<size_t> w13_shape = {num_experts_per_rank, moe_inter_size_per_rank * 2, inner};
        if (device_model_weights.find(w13_name) == device_model_weights.end()) {
          device_model_weights[w13_name] = Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, w13_shape,
                                                  nullptr, &(context_->device->GetMemoryManageStream()));
        }
        Tensor& w13 = device_model_weights.at(w13_name);
        size_t expert_pitch = moe_inter_size_per_rank * inner * GetTypeSize(host_weight_tensor.dtype);
        size_t double_pitch = expert_pitch * 2;
        // gate -> front half, up -> back half. moe_tp source offset (dim0 split within expert).
        size_t src_offset = kimi_k3_config_->moe_tensor_para_size > 1
                                ? (dev_rank / expert_parallel_config.expert_para_size) * expert_pitch
                                : 0;
        MemcpyAsync(
            w13.GetPtr<void>() + static_cast<size_t>(local_expert_idx) * double_pitch + (is_up ? expert_pitch : 0),
            host_weight_tensor.GetPtr<void>() + src_offset, expert_pitch, MEMCPY_HOST_TO_DEVICE,
            context_->device->GetMemoryManageStream());
        KLLM_LOG_INFO << fmt::format(
            "Dev_rank: {}, [MoE expert {}] stored: {} (local_expert {}, layer {}), shape: {}, dtype: {}", dev_rank,
            (is_gate ? "gate->w13" : "up->w13"), w13_name, local_expert_idx, elayer,
            Vector2Str(std::vector<size_t>(w13.shape)), w13.dtype);
        continue;
      }
      if (is_down) {
        size_t inner = host_weight_tensor.shape.size() > 1 ? host_weight_tensor.shape[1] : 1;
        std::string suffix = is_scale ? ".weight_scale_inv" : ".weight";
        std::string down_name = "model.layers." + std::to_string(elayer) + ".mlp.experts.down_proj" + suffix;
        // down: [expert_hidden, moe_inter]; TP splits dim1 (moe_inter). Store per-rank slice.
        size_t inner_tp = DivRoundUp(inner, static_cast<size_t>(kimi_k3_config_->moe_tensor_para_size));
        std::vector<size_t> down_shape = {num_experts_per_rank, host_weight_tensor.shape[0], inner_tp};
        if (device_model_weights.find(down_name) == device_model_weights.end()) {
          device_model_weights[down_name] = Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype,
                                                   down_shape, nullptr, &(context_->device->GetMemoryManageStream()));
        }
        Tensor& down = device_model_weights.at(down_name);
        size_t dst_pitch = inner_tp * GetTypeSize(host_weight_tensor.dtype);
        size_t src_pitch = inner * GetTypeSize(host_weight_tensor.dtype);
        size_t expert_pitch = host_weight_tensor.shape[0] * dst_pitch;
        size_t src_col_offset = kimi_k3_config_->moe_tensor_para_size > 1
                                    ? (dev_rank / expert_parallel_config.expert_para_size) * dst_pitch
                                    : 0;
        Memcpy2DAsync(down.GetPtr<void>() + static_cast<size_t>(local_expert_idx) * expert_pitch, dst_pitch,
                      host_weight_tensor.GetPtr<void>() + src_col_offset, src_pitch, dst_pitch,
                      host_weight_tensor.shape[0], MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());
        KLLM_LOG_INFO << fmt::format(
            "Dev_rank: {}, [MoE expert down] stored: {} (local_expert {}, layer {}), shape: {}, dtype: {}", dev_rank,
            down_name, local_expert_idx, elayer, Vector2Str(std::vector<size_t>(down.shape)), down.dtype);
        continue;
      }
      continue;
    }

    // 2. Attention: dispatch to MLA or KDA by layer type (both share o_proj/g_proj names).
    if (file_weight_name.find(".self_attn.") != std::string::npos) {
      bool handled = false;
      if (layer_idx >= 0 && kimi_k3_config_->kimi_linear_config.IsKdaLayer(layer_idx)) {
        handled = ProcessKdaWeight(file_weight_name, host_weight_tensor, dev_rank, device_model_weights);
      } else {
        handled = ProcessMlaWeight(file_weight_name, host_weight_tensor, dev_rank, device_model_weights);
      }
      if (handled) {
        continue;
      }
    }

    // 3. Latent MoE projections + router gate (fully replicated).
    if (ProcessLatentMoeAndRouter(file_weight_name, host_weight_tensor, dev_rank, device_model_weights)) {
      continue;
    }

    // 4. Dense MLP (layer 0) + shared expert gate/up: column split + fuse gate_up.
    if (CheckWeightNameMatched(
            file_weight_name,
            {".mlp.gate_proj.", ".mlp.up_proj.", ".mlp.shared_expert.gate_proj", ".mlp.shared_expert.up_proj"},
            false)) {
      Tensor dev_tensor;
      size_t mlp_tp = runtime_config_.enable_full_shared_expert ? 1 : context_->global->GetTensorParallelSize();
      weight_impl_->SplitOptTrans(host_weight_tensor, dev_tensor, dev_rank, mlp_tp, /*transpose=*/true);
      weight_impl_->ProcessGateUpProjWeight(file_weight_name, dev_tensor, device_model_weights, dev_rank,
                                            /*is_quant_weight=*/false);
      KLLM_LOG_INFO << fmt::format("Dev_rank: {}, [MLP gate_up fused] processed source: {}, split shape: {}, dtype: {}",
                                   dev_rank, file_weight_name, Vector2Str(std::vector<size_t>(dev_tensor.shape)),
                                   dev_tensor.dtype);
      continue;
    }

    // 5. Dense MLP down + shared expert down: row split.
    if (CheckWeightNameMatched(file_weight_name, {".mlp.down_proj.", ".mlp.shared_expert.down_proj"}, false)) {
      Tensor dev_tensor;
      size_t mlp_tp = runtime_config_.enable_full_shared_expert ? 1 : context_->global->GetTensorParallelSize();
      weight_impl_->TransSplitOptTrans(host_weight_tensor, dev_tensor, dev_rank, mlp_tp, /*transpose=*/false);
      device_model_weights[file_weight_name] = dev_tensor;
      KLLM_LOG_INFO << fmt::format("Dev_rank: {}, [MLP down] stored: {}, shape: {}, dtype: {}", dev_rank,
                                   file_weight_name, Vector2Str(std::vector<size_t>(dev_tensor.shape)),
                                   dev_tensor.dtype);
      continue;
    }

    // 6. Norm layers + attn-res projections (output dim 1) -> fully replicated.
    if (file_weight_name.find("norm.") != std::string::npos ||
        file_weight_name.find("_res_proj.weight") != std::string::npos) {
      CopyReplicated(file_weight_name, host_weight_tensor, dev_rank, device_model_weights, /*transpose=*/false);
      continue;
    }

    // 7. embed_tokens: Vocab split (dim0) with transpose-split-transpose.
    if (CheckWeightNameMatched(file_weight_name, {"model.embed_tokens.weight"}, true)) {
      Tensor dev_tensor;
      weight_impl_->TransSplitOptTrans(host_weight_tensor, dev_tensor, dev_rank,
                                       context_->global->GetTensorParallelSize(), /*transpose=*/true);
      device_model_weights[file_weight_name] = dev_tensor;
      KLLM_LOG_INFO << fmt::format("Dev_rank: {}, [embed_tokens] stored: {}, shape: {}, dtype: {}", dev_rank,
                                   file_weight_name, Vector2Str(std::vector<size_t>(dev_tensor.shape)),
                                   dev_tensor.dtype);
      continue;
    }

    // 8. lm_head: Vocab split (dim0), then transpose.
    if (CheckWeightNameMatched(file_weight_name, {"lm_head.weight"}, true)) {
      Tensor dev_tensor;
      weight_impl_->SplitOptTrans(host_weight_tensor, dev_tensor, dev_rank, context_->global->GetTensorParallelSize(),
                                  /*transpose=*/true);
      device_model_weights[file_weight_name] = dev_tensor;
      KLLM_LOG_INFO << fmt::format("Dev_rank: {}, [lm_head] stored: {}, shape: {}, dtype: {}", dev_rank,
                                   file_weight_name, Vector2Str(std::vector<size_t>(dev_tensor.shape)),
                                   dev_tensor.dtype);
      continue;
    }

    KLLM_LOG_INFO << fmt::format("Dev_rank: {}, [unhandled] left to next file: {}", dev_rank, file_weight_name);
    left_host_weights[file_weight_name] = host_weight_tensor;
  }

  return Status();
}

Status KimiK3WeightLoader::PostProcessModelWeights(std::unordered_map<std::string, Tensor>& dev_weights_map,
                                                   int dev_rank) {
  KLLM_LOG_INFO << fmt::format("Dev_rank: {}, [PostProcess] start, total weights: {}", dev_rank,
                               dev_weights_map.size());

  for (auto& [weight_name, weight_tensor] : dev_weights_map) {
    // Execute pending permutes recorded during ProcessModelWeights (fused_lora_a_proj etc.).
    if (weights_to_permute_[dev_rank].find(weight_name) != weights_to_permute_[dev_rank].end()) {
      weight_impl_->PermuteWeight(weight_tensor, {1, 0}, dev_rank);
    }

    if (!(weight_tensor.dtype == DataType::TYPE_FP32 || weight_tensor.dtype == DataType::TYPE_FP16 ||
          weight_tensor.dtype == DataType::TYPE_BF16)) {
      // Skip quantized (MXFP4/uint8) weights and their scales.
      continue;
    }

    // e_score_correction_bias needs fp32 for routing precision.
    if (weight_name.find("gate.e_score_correction_bias") != std::string::npos) {
      CastDeviceTensorType(weight_tensor, TYPE_FP32, dev_rank);
      continue;
    }

    if (weight_tensor.dtype != kimi_k3_config_->weight_data_type) {
      CastDeviceTensorType(weight_tensor, kimi_k3_config_->weight_data_type, dev_rank);
    }
  }

  // Final device weight inventory: name / shape / dtype after all post-processing.
  for (auto& [weight_name, weight_tensor] : dev_weights_map) {
    KLLM_LOG_INFO << fmt::format("Dev_rank: {}, [final weight] name: {}, shape: {}, dtype: {}", dev_rank, weight_name,
                                 Vector2Str(std::vector<size_t>(weight_tensor.shape)), weight_tensor.dtype);
  }

  KLLM_LOG_INFO << fmt::format("Dev_rank: {}, [PostProcess] complete, total weights: {}", dev_rank,
                               dev_weights_map.size());
  return Status();
}

}  // namespace ksana_llm
