/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "layers/moe_layer_factory.h"

#include "layers/batched_matmul_layer.h"
#include "layers/blockwise_matmul_layer.h"
#include "layers/cutlass_matmul_layer.h"
#include "layers/cutlass_moe_layer.h"
#include "layers/fp8_matmul_layer.h"
#include "layers/fp8_moe_layer.h"
#include "layers/humming_fused_fp4_a16_moe_layer.h"
#include "layers/humming_fused_fp4_a8_moe_layer.h"
#include "layers/machete_matmul_layer.h"
#include "layers/marlin_fused_fp4_a16_moe_layer.h"
#include "layers/marlin_matmul_layer.h"
#include "layers/marlin_moe_layer.h"
#include "layers/matmul_layer.h"
#include "layers/moe_layer.h"

namespace ksana_llm {

MoeLayerFactory::MoeLayerFactory(const ModelConfig& model_config, const RuntimeConfig& runtime_config, const int rank,
                                 std::shared_ptr<Context> context)
    : model_config_(model_config), runtime_config_(runtime_config), rank_(rank), context_(context) {
  expert_parallel_manager_ = std::make_shared<ExpertParallelManager>();

#ifdef ENABLE_CUDA
  // TODO(winminkong): Organize the quantization backend and quantization types of the MoE layer.
  // for common moe, qwen and llama moe
  builder_map_[{TYPE_FP32, TYPE_FP32, TYPE_FP32, MOE_QUANT_NONE, DEFAULT_MOE_BACKEND}] =
      &MoeLayerFactory::BuildLayer<MoeLayer>;
  builder_map_[{TYPE_FP16, TYPE_FP16, TYPE_FP16, MOE_QUANT_NONE, DEFAULT_MOE_BACKEND}] =
      &MoeLayerFactory::BuildLayer<MoeLayer>;
  builder_map_[{TYPE_BF16, TYPE_BF16, TYPE_BF16, MOE_QUANT_NONE, DEFAULT_MOE_BACKEND}] =
      &MoeLayerFactory::BuildLayer<MoeLayer>;

  // for hunyuan-large int4
  builder_map_[{TYPE_I4_GROUP, TYPE_FP16, TYPE_FP16, MOE_QUANT_GPTQ, MARLIN_MOE_BACKEND}] =
      &MoeLayerFactory::BuildLayer<MarlinMoeLayer>;

#  ifdef ENABLE_FP8
  // NOTE: Fp8MoeLayer only suport fp8e4m3 rightnow
  // for hunyuan-large fp8
  builder_map_[{TYPE_FP8_E4M3, TYPE_FP32, TYPE_FP32, MOE_QUANT_FP8_E4M3, DEFAULT_MOE_BACKEND}] =
      &MoeLayerFactory::BuildLayer<Fp8MoeLayer>;
  builder_map_[{TYPE_FP8_E4M3, TYPE_FP16, TYPE_FP16, MOE_QUANT_FP8_E4M3, DEFAULT_MOE_BACKEND}] =
      &MoeLayerFactory::BuildLayer<Fp8MoeLayer>;
  builder_map_[{TYPE_FP8_E4M3, TYPE_BF16, TYPE_BF16, MOE_QUANT_FP8_E4M3, DEFAULT_MOE_BACKEND}] =
      &MoeLayerFactory::BuildLayer<Fp8MoeLayer>;

  // for deepseek fp8 blockwise
  builder_map_[{TYPE_FP8_E4M3, TYPE_FP32, TYPE_FP32, MOE_QUANT_BLOCK_FP8_E4M3, TRITON_MOE_BACKEND}] =
      &MoeLayerFactory::BuildLayer<MoeLayer>;
  builder_map_[{TYPE_FP8_E4M3, TYPE_FP16, TYPE_FP16, MOE_QUANT_BLOCK_FP8_E4M3, TRITON_MOE_BACKEND}] =
      &MoeLayerFactory::BuildLayer<MoeLayer>;
  builder_map_[{TYPE_FP8_E4M3, TYPE_BF16, TYPE_BF16, MOE_QUANT_BLOCK_FP8_E4M3, TRITON_MOE_BACKEND}] =
      &MoeLayerFactory::BuildLayer<MoeLayer>;

  // for deepseek moe-int4
  builder_map_[{TYPE_UINT8, TYPE_FP16, TYPE_FP16, MOE_QUANT_GPTQ, TRITON_MOE_BACKEND}] =
      &MoeLayerFactory::BuildLayer<MoeLayer>;
  builder_map_[{TYPE_UINT8, TYPE_BF16, TYPE_BF16, MOE_QUANT_GPTQ, TRITON_MOE_BACKEND}] =
      &MoeLayerFactory::BuildLayer<MoeLayer>;

  // for deepseek w4afp8 (and compressed-tensors which uses TYPE_INT8 to route here)
  builder_map_[{TYPE_INT8, TYPE_FP16, TYPE_FP16, MOE_QUANT_GPTQ, CUTLASS_MOE_BACKEND}] =
      &MoeLayerFactory::BuildLayer<CutlassMoeLayer>;
  builder_map_[{TYPE_INT8, TYPE_BF16, TYPE_BF16, MOE_QUANT_GPTQ, CUTLASS_MOE_BACKEND}] =
      &MoeLayerFactory::BuildLayer<CutlassMoeLayer>;

  // for deepseek w4afp8 triton
  // NOTE(jinxcwu) fast than CUTLASS_MOE_BACKEND when prefill
  builder_map_[{TYPE_INT8, TYPE_FP16, TYPE_FP16, MOE_QUANT_GPTQ, TRITON_MOE_BACKEND}] =
      &MoeLayerFactory::BuildLayer<MoeLayer>;
  builder_map_[{TYPE_INT8, TYPE_BF16, TYPE_BF16, MOE_QUANT_GPTQ, TRITON_MOE_BACKEND}] =
      &MoeLayerFactory::BuildLayer<MoeLayer>;

  // for deepseek v4 FP4 MoE experts (FP4 weight → runtime dequant to FP8 → GroupTriton compute)
  builder_map_[{TYPE_FP4_E2M1FN_x2, TYPE_FP16, TYPE_FP16, MOE_QUANT_GPTQ, TRITON_MOE_BACKEND}] =
      &MoeLayerFactory::BuildLayer<MoeLayer>;
  builder_map_[{TYPE_FP4_E2M1FN_x2, TYPE_BF16, TYPE_BF16, MOE_QUANT_GPTQ, TRITON_MOE_BACKEND}] =
      &MoeLayerFactory::BuildLayer<MoeLayer>;

  // for deepseek v4 FP4 MoE experts — Marlin W4A16 path
  builder_map_[{TYPE_FP4_E2M1FN_x2, TYPE_FP16, TYPE_FP16, MOE_QUANT_GPTQ, MARLIN_MOE_BACKEND}] =
      &MoeLayerFactory::BuildLayer<MarlinFusedFp4A16MoeLayer>;
  builder_map_[{TYPE_FP4_E2M1FN_x2, TYPE_BF16, TYPE_BF16, MOE_QUANT_GPTQ, MARLIN_MOE_BACKEND}] =
      &MoeLayerFactory::BuildLayer<MarlinFusedFp4A16MoeLayer>;

  // for deepseek v4 FP4 MoE experts — Humming W4A16 path
  builder_map_[{TYPE_FP4_E2M1FN_x2, TYPE_FP16, TYPE_FP16, MOE_QUANT_GPTQ, HUMMING_W4A16_MOE_BACKEND}] =
      &MoeLayerFactory::BuildLayer<HummingFusedFp4A16MoeLayer>;
  builder_map_[{TYPE_FP4_E2M1FN_x2, TYPE_BF16, TYPE_BF16, MOE_QUANT_GPTQ, HUMMING_W4A16_MOE_BACKEND}] =
      &MoeLayerFactory::BuildLayer<HummingFusedFp4A16MoeLayer>;

  // for deepseek v4 FP4 MoE experts — Humming W4A8 path
  builder_map_[{TYPE_FP4_E2M1FN_x2, TYPE_FP16, TYPE_FP16, MOE_QUANT_GPTQ, HUMMING_W4A8_MOE_BACKEND}] =
      &MoeLayerFactory::BuildLayer<HummingFusedFp4A8MoeLayer>;
  builder_map_[{TYPE_FP4_E2M1FN_x2, TYPE_BF16, TYPE_BF16, MOE_QUANT_GPTQ, HUMMING_W4A8_MOE_BACKEND}] =
      &MoeLayerFactory::BuildLayer<HummingFusedFp4A8MoeLayer>;
#  endif
#endif
}

std::shared_ptr<BaseMoeLayer> MoeLayerFactory::AutoCreateMoeLayer(std::shared_ptr<BaseWeight> base_weight,
                                                                  std::vector<std::string> weight_names,
                                                                  DataType weight_type, DataType input_type,
                                                                  DataType output_type, int layer_idx,
                                                                  const std::vector<std::any>& init_params) {
  // moe layer   (weight_names[0]: up_gate_experts, weight_names[1]: down_experts)
  bool use_vllm_moe = model_config_.moe_config.use_vllm_moe;
  size_t hidden_size = static_cast<size_t>(model_config_.hidden_units);
  size_t moe_inter_size_per_rank = static_cast<size_t>(
      DivRoundUp(model_config_.moe_config.moe_inter_size, runtime_config_.parallel_basic_config.moe_tensor_para_size));
  size_t expert_num_per_node =
      model_config_.moe_config.num_experts / (runtime_config_.parallel_basic_config.expert_parallel_size *
                                              runtime_config_.parallel_basic_config.expert_world_size);
  // Expert count check
  size_t up_gate_experts_num = base_weight->GetModelWeights(weight_names[0]).shape[0];
  size_t down_experts_num = base_weight->GetModelWeights(weight_names[1]).shape[0];
  KLLM_CHECK_WITH_INFO(up_gate_experts_num == down_experts_num,
                       fmt::format("up_gate experts {} != down_experts {}", up_gate_experts_num, down_experts_num));
  // Parse enable_moe_int4
  bool enable_moe_int4 = false;
  if (model_config_.quant_config.method == QUANT_GPTQ && model_config_.quant_config.bits == 4) {
    enable_moe_int4 = true;
  } else if (model_config_.quant_config.method != QUANT_GPTQ && model_config_.sub_quant_configs.size() > 0) {
    // Check for mixed quantization with W4A8_AWQ or GPTQ for experts
    for (const auto& sub_config : model_config_.sub_quant_configs) {
      if (sub_config.method == QUANT_W4A8_AWQ || sub_config.method == QUANT_GPTQ) {
        for (const std::string& pattern_layer : sub_config.pattern_layers) {
          if (weight_names[0].find(pattern_layer) != std::string::npos) {
            enable_moe_int4 = true;
            break;
          }
        }
        if (enable_moe_int4) {
          for (const std::string& ignored_layer : sub_config.ignored_layers) {
            if (weight_names[0].find(ignored_layer) != std::string::npos) {
              enable_moe_int4 = false;
              break;
            }
          }
        }
        break;  // Found the relevant sub_config
      }
    }
  }
  // Detect FP4 expert weights
  bool enable_moe_fp4 = (model_config_.quant_config.expert_dtype == "fp4");
  // MoE layer size check
  size_t up_gate_hidden_size = base_weight->GetModelWeights(weight_names[0]).shape[2];
  size_t down_hidden_size = base_weight->GetModelWeights(weight_names[1]).shape[1];
  if (enable_moe_fp4 && runtime_config_.fp4_moe_backend == FP4_MOE_BACKEND_TYPE::FP4Marlin) {
    // Marlin FP4: weights are repacked to Marlin tile format [E, K/16, N*2],
    // so shape-based hidden_size derivation is not applicable.
    // Use model config values directly (already computed above).
    up_gate_hidden_size = hidden_size;
    down_hidden_size = hidden_size;
  } else if (enable_moe_fp4) {
    // FP4: 2 values per byte (same packing ratio as INT4)
    up_gate_hidden_size = base_weight->GetModelWeights(weight_names[0]).shape[2] * 2;
    down_hidden_size = base_weight->GetModelWeights(weight_names[1]).shape[1];
  } else if (enable_moe_int4) {
    if (weight_type == TYPE_INT8 || weight_type == TYPE_UINT8) {
      // INT8/UINT8 storage (W4A8/DeepSeek): 2 x 4-bit values per byte
      up_gate_hidden_size = base_weight->GetModelWeights(weight_names[0]).shape[2] * 2;
      down_hidden_size = base_weight->GetModelWeights(weight_names[1]).shape[1];
    } else if (use_vllm_moe) {
      // vLLM path: int32 packed
      up_gate_hidden_size = up_gate_hidden_size / 4 * (32 / model_config_.quant_config.bits);
    } else {
      // Marlin GPTQ path
      up_gate_hidden_size =
          base_weight->GetModelWeights(weight_names[0]).shape[1] / (sizeof(int) / model_config_.quant_config.bits) * 16;
      down_hidden_size = base_weight->GetModelWeights(weight_names[1]).shape[2] / model_config_.quant_config.bits * 2;
    }
  }
  KLLM_CHECK_WITH_INFO(up_gate_hidden_size == down_hidden_size,
                       fmt::format("up_gate_experts hidden_size {} != down_experts hidden_size {}", up_gate_hidden_size,
                                   down_hidden_size));
  // MoE layer weight type check
  weight_type = base_weight->GetModelWeights(weight_names[0]).dtype;
  DataType down_weight_type = base_weight->GetModelWeights(weight_names[1]).dtype;
  KLLM_CHECK_WITH_INFO(down_weight_type == weight_type,
                       fmt::format("down_experts dtype {} != up_gate_experts dtype {}", down_weight_type, weight_type));

  std::vector<std::any> moe_matmul_param = init_params;                                    // moe_scale_norm_mode
  moe_matmul_param.push_back(runtime_config_.max_step_token_num);                          // max_token_num
  moe_matmul_param.push_back(layer_idx);                                                   // layer_idx
  moe_matmul_param.push_back(expert_num_per_node);                                         // expert_num_per_node
  moe_matmul_param.push_back(hidden_size);                                                 // hidden_size
  moe_matmul_param.push_back(moe_inter_size_per_rank);                                     // Inter_size
  moe_matmul_param.push_back(model_config_.moe_config.experts_topk);                       // experts topk
  moe_matmul_param.push_back(runtime_config_.parallel_basic_config.tensor_parallel_size);  // TP_size
  moe_matmul_param.push_back(use_vllm_moe);                                                // use_vllm_moe
  moe_matmul_param.push_back(model_config_.moe_config.num_expert_group);                   // num_expert_group
  moe_matmul_param.push_back(model_config_.moe_config.expert_groups_topk);                 // expert_groups_topk
  moe_matmul_param.push_back(model_config_.moe_config.scoring_func);                       // scoring_func
  moe_matmul_param.push_back(model_config_.moe_config.topk_method);                        // topk_method
  moe_matmul_param.push_back(model_config_.moe_config.norm_topk_prob);                     // norm_topk_prob
  moe_matmul_param.push_back(model_config_.moe_config.routed_scaling_factor);              // routed_scaling_factor
  moe_matmul_param.push_back(model_config_.moe_config.use_e_score_correction_bias);  // use_e_score_correction_bias
  moe_matmul_param.push_back(runtime_config_.enable_full_shared_expert);             // dp = ep
  // Set fp8_weight_dtype: distinguish between blockwise FP8 and per-tensor FP8
  // When FP4 is enabled, experts use FP4 weights not FP8, so fp8_weight_dtype must be INVALID
  if (!enable_moe_int4 && !enable_moe_fp4 && model_config_.quant_config.is_fp8_blockwise) {
    moe_matmul_param.push_back(DataType::TYPE_BLOCK_FP8_E4M3);  // fp8_weight_dtype: blockwise FP8
  } else if (!enable_moe_int4 && !enable_moe_fp4 && weight_type == TYPE_FP8_E4M3) {
    moe_matmul_param.push_back(DataType::TYPE_FP8_E4M3);  // fp8_weight_dtype: per-tensor FP8
  } else {
    moe_matmul_param.push_back(DataType::TYPE_INVALID);  // fp8_weight_dtype: not FP8
  }
  if (enable_moe_fp4) {
    moe_matmul_param.push_back(DataType::TYPE_FP4_E2M1FN_x2);  // int_weight_dtype: FP4
    moe_matmul_param.push_back(static_cast<int>(128));         // group_size: 128 for merged FP4 scales
  } else if (enable_moe_int4) {
    if (weight_type == TYPE_UINT8) {
      moe_matmul_param.push_back(TYPE_UINT4x2);  // int_weight_dtype
    } else if (weight_type == TYPE_INT8) {
      moe_matmul_param.push_back(TYPE_INT4x2);  // int_weight_dtype
    } else {
      moe_matmul_param.push_back(TYPE_I4_GROUP);  // int_weight_dtype
    }
    // For mixed quantization (FP8 default + W4A8_AWQ/GPTQ for experts), get group_size from sub_quant_config
    size_t group_size = model_config_.quant_config.group_size;
    for (const auto& sub_config : model_config_.sub_quant_configs) {
      if (sub_config.method == QUANT_W4A8_AWQ || sub_config.method == QUANT_GPTQ) {
        group_size = sub_config.group_size;
        break;
      }
    }
    moe_matmul_param.push_back(static_cast<int>(group_size));  // group_size
  } else {
    moe_matmul_param.push_back(DataType::TYPE_INVALID);  // int_weight_dtype
    moe_matmul_param.push_back(0);                       // group_size
  }
  moe_matmul_param.push_back(model_config_.moe_config.apply_weight);
  moe_matmul_param.push_back(model_config_.swiglu_limit);  // swiglu_limit (>0 means enabled)
  moe_matmul_param.push_back(static_cast<uint32_t>(model_config_.num_hash_layers));
  moe_matmul_param.push_back(static_cast<uint32_t>(model_config_.vocab_size));

  // FP4 MoE layer creation
  if (enable_moe_fp4) {
    if (runtime_config_.fp4_moe_backend == FP4_MOE_BACKEND_TYPE::FP4Marlin) {
      return CreateLayer(weight_type, input_type, output_type, moe_matmul_param, MOE_QUANT_GPTQ, MARLIN_MOE_BACKEND);
    } else if (runtime_config_.fp4_moe_backend == FP4_MOE_BACKEND_TYPE::FP4HummingW4A16) {
      return CreateLayer(weight_type, input_type, output_type, moe_matmul_param, MOE_QUANT_GPTQ,
                         HUMMING_W4A16_MOE_BACKEND);
    } else if (runtime_config_.fp4_moe_backend == FP4_MOE_BACKEND_TYPE::FP4HummingW4A8) {
      return CreateLayer(weight_type, input_type, output_type, moe_matmul_param, MOE_QUANT_GPTQ,
                         HUMMING_W4A8_MOE_BACKEND);
    }
    return CreateLayer(weight_type, input_type, output_type, moe_matmul_param, MOE_QUANT_GPTQ, TRITON_MOE_BACKEND);
  }
  // FP8 MoE layer creation
  if (weight_type == TYPE_FP8_E4M3) {
    QuantMode quant_mode = model_config_.quant_config.is_fp8_blockwise ? MOE_QUANT_BLOCK_FP8_E4M3 : MOE_QUANT_FP8_E4M3;
    MoeComputeBackend backend = model_config_.quant_config.is_fp8_blockwise ? TRITON_MOE_BACKEND : DEFAULT_MOE_BACKEND;
    return CreateLayer(weight_type, input_type, output_type, moe_matmul_param, quant_mode, backend);
  }
  // INT4 MoE layer creation
  if (enable_moe_int4) {
    if (weight_type == TYPE_UINT8 || runtime_config_.w4afp8_moe_backend == W4AFP8_MOE_BACKEND::GroupTriton ||
        runtime_config_.w4afp8_moe_backend == W4AFP8_MOE_BACKEND::TensorTriton) {
      return CreateLayer(weight_type, input_type, output_type, moe_matmul_param, MOE_QUANT_GPTQ, TRITON_MOE_BACKEND);
    }
    // TYPE_INT8 routes to CUTLASS or Triton (via GroupTriton/TensorTriton env override).
    // compressed-tensors sets TYPE_INT8 because its weights are pre-transposed ([N,K] layout).
    // Note: w4afp8_moe_backend auto-detection may override this to GroupTriton for compressed-tensors,
    // in which case the above Triton branch is taken instead.
    if (weight_type == TYPE_INT8) {
      return CreateLayer(weight_type, input_type, output_type, moe_matmul_param, MOE_QUANT_GPTQ, CUTLASS_MOE_BACKEND);
    }
    if (!use_vllm_moe) {
      return CreateLayer(TYPE_I4_GROUP, input_type, output_type, moe_matmul_param, MOE_QUANT_GPTQ, MARLIN_MOE_BACKEND);
    }
  }
  // Default MoE layer creation
  return CreateLayer(weight_type, input_type, output_type, moe_matmul_param, MOE_QUANT_NONE, DEFAULT_MOE_BACKEND);
}

std::shared_ptr<BaseMoeLayer> MoeLayerFactory::CreateLayer(DataType weight_type, DataType input_type,
                                                           DataType output_type,
                                                           const std::vector<std::any>& init_params,
                                                           QuantMode quant_mode, MoeComputeBackend backend) {
  auto it = builder_map_.find({weight_type, input_type, output_type, quant_mode, backend});
  KLLM_CHECK_WITH_INFO(
      it != builder_map_.end(),
      fmt::format("Moe Not support weight_type {}, input_type {}, output_type {}, quant_mode {}, backend {}.",
                  GetTypeString(weight_type), GetTypeString(input_type), GetTypeString(output_type),
                  GetQuantModeString(quant_mode), GetMoeComputeBackendString(backend)));
  std::shared_ptr<BaseMoeLayer> layer = (this->*(it->second))();
  layer->Init(init_params, runtime_config_, context_, rank_);

  // All BaseMoeLayer subclasses expose SetExpertParallelManager via the virtual interface.
  // Subclasses without EP support (e.g., MarlinMoeLayer) inherit the default no-op from BaseMoeLayer.
  layer->SetExpertParallelManager(expert_parallel_manager_);

  return layer;
}

}  // namespace ksana_llm
