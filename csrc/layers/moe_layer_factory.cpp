/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "layers/moe_layer_factory.h"

#include "layers/batched_matmul_layer.h"
#include "layers/blockwise_matmul_layer.h"
#include "layers/cutlass_group_wint4afp8_moe_layer.h"
#include "layers/cutlass_matmul_layer.h"
#include "layers/fp8_matmul_layer.h"
#include "layers/fp8_moe_layer.h"
#include "layers/humming_fused_wfp4a16_moe_layer.h"
#include "layers/humming_fused_wfp4afp8_moe_layer.h"
#include "layers/machete_matmul_layer.h"
#include "layers/marlin_fused_wfp4a16_moe_layer.h"
#include "layers/marlin_matmul_layer.h"
#include "layers/marlin_moe_layer.h"
#include "layers/matmul_layer.h"
#include "layers/moe_layer.h"

#ifdef ENABLE_KUNLUN
#include "layers/kunlun_moe_layer.h"
#endif

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

  // for per-tensor fp8 experts on the hpc-ops fused MoE (bf16 activation only — the hpc kernel
  // emits bf16). MoeLayer supplies the gating + expert map; the per-tensor GEMMs are hpc's.
  builder_map_[{TYPE_FP8_E4M3, TYPE_BF16, TYPE_BF16, MOE_QUANT_FP8_E4M3, HPC_MOE_BACKEND}] =
      &MoeLayerFactory::BuildLayer<MoeLayer>;

  // for deepseek moe-int4
  builder_map_[{TYPE_UINT8, TYPE_FP16, TYPE_FP16, MOE_QUANT_GPTQ, TRITON_MOE_BACKEND}] =
      &MoeLayerFactory::BuildLayer<MoeLayer>;
  builder_map_[{TYPE_UINT8, TYPE_BF16, TYPE_BF16, MOE_QUANT_GPTQ, TRITON_MOE_BACKEND}] =
      &MoeLayerFactory::BuildLayer<MoeLayer>;

  // for deepseek w4afp8 (and compressed-tensors which uses TYPE_INT8 to route here)
  builder_map_[{TYPE_INT8, TYPE_FP16, TYPE_FP16, MOE_QUANT_GPTQ, CUTLASS_MOE_BACKEND}] =
      &MoeLayerFactory::BuildLayer<CutlassGroupWint4Afp8MoeLayer>;
  builder_map_[{TYPE_INT8, TYPE_BF16, TYPE_BF16, MOE_QUANT_GPTQ, CUTLASS_MOE_BACKEND}] =
      &MoeLayerFactory::BuildLayer<CutlassGroupWint4Afp8MoeLayer>;

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
      &MoeLayerFactory::BuildLayer<MarlinFusedWfp4A16MoeLayer>;
  builder_map_[{TYPE_FP4_E2M1FN_x2, TYPE_BF16, TYPE_BF16, MOE_QUANT_GPTQ, MARLIN_MOE_BACKEND}] =
      &MoeLayerFactory::BuildLayer<MarlinFusedWfp4A16MoeLayer>;

  // for deepseek v4 FP4 MoE experts — Humming W4A16 path
  builder_map_[{TYPE_FP4_E2M1FN_x2, TYPE_FP16, TYPE_FP16, MOE_QUANT_GPTQ, HUMMING_W4A16_MOE_BACKEND}] =
      &MoeLayerFactory::BuildLayer<HummingFusedWfp4A16MoeLayer>;
  builder_map_[{TYPE_FP4_E2M1FN_x2, TYPE_BF16, TYPE_BF16, MOE_QUANT_GPTQ, HUMMING_W4A16_MOE_BACKEND}] =
      &MoeLayerFactory::BuildLayer<HummingFusedWfp4A16MoeLayer>;

  // for deepseek v4 FP4 MoE experts — Humming W4A8 path
  builder_map_[{TYPE_FP4_E2M1FN_x2, TYPE_FP16, TYPE_FP16, MOE_QUANT_GPTQ, HUMMING_W4A8_MOE_BACKEND}] =
      &MoeLayerFactory::BuildLayer<HummingFusedWfp4Afp8MoeLayer>;
  builder_map_[{TYPE_FP4_E2M1FN_x2, TYPE_BF16, TYPE_BF16, MOE_QUANT_GPTQ, HUMMING_W4A8_MOE_BACKEND}] =
      &MoeLayerFactory::BuildLayer<HummingFusedWfp4Afp8MoeLayer>;
#  endif
#endif

#ifdef ENABLE_KUNLUN
  // Kunlun XPU MoE layer (supports fp16/bf16, no quantization)
  builder_map_[{TYPE_FP16, TYPE_FP16, TYPE_FP16, MOE_QUANT_NONE, DEFAULT_MOE_BACKEND}] =
      &MoeLayerFactory::BuildLayer<KunlunMoeLayer>;
  builder_map_[{TYPE_BF16, TYPE_BF16, TYPE_BF16, MOE_QUANT_NONE, DEFAULT_MOE_BACKEND}] =
      &MoeLayerFactory::BuildLayer<KunlunMoeLayer>;
  // W4A8_KLX quantized MoE (INT4 packed as INT8 storage, dispatched as TYPE_INT4x2 to align with CUDA)
  // TODO(zhongzhicao): Replace with KunlunQuantMoeLayer once inference kernels are implemented
  builder_map_[{TYPE_INT8, TYPE_BF16, TYPE_BF16, MOE_QUANT_NONE, DEFAULT_MOE_BACKEND}] =
      &MoeLayerFactory::BuildLayer<KunlunMoeLayer>;
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
  size_t moe_hidden_size = model_config_.moe_config.routed_expert_hidden_size > 0
                               ? model_config_.moe_config.routed_expert_hidden_size
                               : hidden_size;
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
#ifdef ENABLE_KUNLUN
  } else if (model_config_.quant_config.method == QUANT_W4A8_KLX) {
    // W4A8_KLX: expert weights are INT4 packed into INT8 (pack_factor=2)
    enable_moe_int4 = true;
#endif
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
    up_gate_hidden_size = moe_hidden_size;
    down_hidden_size = moe_hidden_size;
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

  MoeLayerParameters moe_params;
  moe_params.moe_scale_norm_mode = std::any_cast<MoeScaleNormMode>(init_params[0]);
  moe_params.max_token_num = runtime_config_.max_step_token_num;
  moe_params.layer_idx = layer_idx;
  moe_params.expert_num_per_node = expert_num_per_node;
  moe_params.expert_hidden_size = moe_hidden_size;
  moe_params.expert_inter_size = moe_inter_size_per_rank;
  moe_params.expert_topk = model_config_.moe_config.experts_topk;
  moe_params.tp_size = runtime_config_.parallel_basic_config.tensor_parallel_size;
  moe_params.use_vllm_moe = use_vllm_moe;
  moe_params.num_expert_group = model_config_.moe_config.num_expert_group;
  moe_params.expert_groups_topk = model_config_.moe_config.expert_groups_topk;
  moe_params.scoring_func = model_config_.moe_config.scoring_func;
  moe_params.topk_method = model_config_.moe_config.topk_method;
  moe_params.norm_topk_prob = model_config_.moe_config.norm_topk_prob;
  moe_params.routed_scaling_factor = model_config_.moe_config.routed_scaling_factor;
  moe_params.use_e_score_correction_bias = model_config_.moe_config.use_e_score_correction_bias;
  moe_params.enable_full_shared_expert = runtime_config_.enable_full_shared_expert;
  if (!enable_moe_int4 && !enable_moe_fp4 && model_config_.quant_config.is_fp8_blockwise) {
    moe_params.fp8_weight_dtype = DataType::TYPE_BLOCK_FP8_E4M3;
  } else if (!enable_moe_int4 && !enable_moe_fp4 && weight_type == TYPE_FP8_E4M3) {
    moe_params.fp8_weight_dtype = DataType::TYPE_FP8_E4M3;
  } else {
    moe_params.fp8_weight_dtype = DataType::TYPE_INVALID;
  }
  if (enable_moe_fp4) {
    moe_params.int_weight_dtype = DataType::TYPE_FP4_E2M1FN_x2;
    moe_params.group_size = 128;
  } else if (enable_moe_int4) {
    if (weight_type == TYPE_UINT8) {
      moe_params.int_weight_dtype = TYPE_UINT4x2;
    } else if (weight_type == TYPE_INT8) {
      moe_params.int_weight_dtype = TYPE_INT4x2;
    } else {
      moe_params.int_weight_dtype = TYPE_I4_GROUP;
    }
    size_t group_size = model_config_.quant_config.group_size;
    for (const auto& sub_config : model_config_.sub_quant_configs) {
      if (sub_config.method == QUANT_W4A8_AWQ || sub_config.method == QUANT_GPTQ) {
        group_size = sub_config.group_size;
        break;
      }
    }
    moe_params.group_size = static_cast<int>(group_size);
  } else {
    moe_params.int_weight_dtype = DataType::TYPE_INVALID;
    moe_params.group_size = 0;
  }
  moe_params.apply_weight = model_config_.moe_config.apply_weight;
  moe_params.swiglu_limit = model_config_.swiglu_limit;
  moe_params.num_hash_layers = static_cast<uint32_t>(model_config_.num_hash_layers);
  moe_params.vocab_size = static_cast<uint32_t>(model_config_.vocab_size);
  moe_params.situ_beta = model_config_.situ_beta;
  moe_params.situ_linear_beta = model_config_.situ_linear_beta;
  moe_params.activation_function = model_config_.activation_function;

  std::vector<std::any> moe_matmul_param = {moe_params};

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
    if (model_config_.quant_config.is_fp8_blockwise) {
      // Blockwise FP8 always lands on MoeLayer; it decides internally whether the fused MoE is
      // Triton's or hpc-ops'.
      return CreateLayer(weight_type, input_type, output_type, moe_matmul_param, MOE_QUANT_BLOCK_FP8_E4M3,
                         TRITON_MOE_BACKEND);
    }
    // Per-tensor FP8: hpc-ops when it is selected (MoeLayer), cutlass otherwise (Fp8MoeLayer).
    // The hpc kernel only emits bf16, so anything else keeps the cutlass path.
    if (runtime_config_.moe_backend_config.use_hpc_moe && input_type == TYPE_BF16 && output_type == TYPE_BF16) {
      return CreateLayer(weight_type, input_type, output_type, moe_matmul_param, MOE_QUANT_FP8_E4M3, HPC_MOE_BACKEND);
    }
    return CreateLayer(weight_type, input_type, output_type, moe_matmul_param, MOE_QUANT_FP8_E4M3,
                       DEFAULT_MOE_BACKEND);
  }
  // INT4 MoE layer creation (NVIDIA/Ascend backends only; Kunlun W4A8_KLX uses default MoE path)
#ifdef ENABLE_KUNLUN
  if (enable_moe_int4 && model_config_.quant_config.method != QUANT_W4A8_KLX) {
#else
  if (enable_moe_int4) {
#endif
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
