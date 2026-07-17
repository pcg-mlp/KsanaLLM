/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "device/device_types.h"
#include "utils/search_path.h"

namespace ksana_llm {

struct RoPEScalingFactor {
  std::string type{"default"};
  float factor{1.0f};
  float low_freq_factor{1.0f};
  float high_freq_factor{4.0f};
  int original_max_position_embeddings{8192};
  bool has_alpha{false};
  float scaling_alpha{1.0f};          // for dynamic alpha rope
  std::vector<int> mrope_section{};   // for multimodal rope
  bool mrope_interleaved{false};      // for qwen3-vl: interleaved rotary embedding
  std::vector<int> xdrope_section{};  // for arc-hunyuan-video

  // deepseek-yarn config params
  bool use_deepseek_rope{false};
  float beta_fast{32.0f};
  float beta_slow{1.0f};
  float mscale{1.0f};
  float mscale_all_dim{1.0f};
  bool use_deepseek_v4_rope{false};
};

enum QuantMode {
  QUANT_NONE,
  QUANT_GPTQ,
  QUANT_AWQ,
  QUANT_FP8_E4M3,
  QUANT_BLOCK_FP8_E4M3,
  QUANT_W4A8_AWQ,
  MOE_QUANT_NONE,
  MOE_QUANT_BLOCK_FP8_E4M3,
  MOE_QUANT_FP8_E4M3,
  MOE_QUANT_GPTQ
};

__attribute__((unused)) static std::string GetQuantModeString(QuantMode quant_mode) {
  static const std::unordered_map<QuantMode, std::string> quant_mode_map{
      {QUANT_NONE, "None"},
      {QUANT_GPTQ, "GPTQ"},
      {QUANT_AWQ, "AWQ"},
      {QUANT_FP8_E4M3, "FP8_E4M3"},
      {QUANT_BLOCK_FP8_E4M3, "BLOCK_FP8_E4M3"},
      {QUANT_W4A8_AWQ, "W4A8_AWQ"},
      {MOE_QUANT_NONE, "MOE_QUANT None"},
      {MOE_QUANT_BLOCK_FP8_E4M3, "MOE_QUANT BLOCK_FP8_E4M3"},
      {MOE_QUANT_FP8_E4M3, "MOE_QUANT FP8_E4M3"},
      {MOE_QUANT_GPTQ, "MOE_QUANT GPTQ"}};
  return quant_mode_map.count(quant_mode) ? quant_mode_map.at(quant_mode) : "Unknown";
}

/*
 * LinearComputeBackend:
 * - DEFAULT_LINEAR_BACKEND:
 *     The default linear backend. Suitable for FP8, FP16, BF16 and FP32 models.
 *     Internally may use cuBLAS, CUTLASS or other implementations.
 * - CUTLASS_LINEAR_BACKEND:
 *     An int4-only backend. It is the default choice for int4 inference and
 *     is suitable for GPTQ/AWQ models.
 * - MARLIN_LINEAR_BACKEND:
 *     An int4-only backend suitable for GPTQ/AWQ models. Additionally supports the
 *     `act_order` feature.
 * - MACHETE_LINEAR_BACKEND:
 *     An int4-only backend restricted to SM90 (NVIDIA Hopper class). Currently
 *     only used by DeepSeek GPTQ int4 model.
 */
enum LinearComputeBackend {
  DEFAULT_LINEAR_BACKEND,
  CUTLASS_LINEAR_BACKEND,
  MARLIN_LINEAR_BACKEND,
  MACHETE_LINEAR_BACKEND
};

__attribute__((unused)) static std::string GetLinearComputeBackendString(LinearComputeBackend linear_compute_backend) {
  static const std::unordered_map<LinearComputeBackend, std::string> linear_compute_backend_map{
      {DEFAULT_LINEAR_BACKEND, "default_linear"},
      {CUTLASS_LINEAR_BACKEND, "cutlass_linear"},
      {MARLIN_LINEAR_BACKEND, "marlin_linear"},
      {MACHETE_LINEAR_BACKEND, "machete_linear"}};
  return linear_compute_backend_map.count(linear_compute_backend)
             ? linear_compute_backend_map.at(linear_compute_backend)
             : "Unknown";
}

/*
 * MoeComputeBackend:
 * - DEFAULT_MOE_BACKEND:
 *     The default MoE backend based on cutlass. General-purpose implementation that supports
 *     common models such as Qwen, LLaMA, etc.
 * - MARLIN_MOE_BACKEND:
 *     An int4-only MoE backend. Intended for int4 versions of models like Qwen
 *     and LLaMA.
 * - TRITON_MOE_BACKEND:
 *     DeepSeek-only backend that implements a fused-MoE strategy (using Triton).
 *     Supports FP16, BF16, FP8-blockwise and int4.
 * - CUTLASS_MOE_BACKEND:
 *     DeepSeek-only backend that implements a group-GEMM strategy (using CUTLASS).
 *     Only supports the w4afp8 weight format.
 */
enum MoeComputeBackend {
  DEFAULT_MOE_BACKEND,
  CUTLASS_MOE_BACKEND,
  MARLIN_MOE_BACKEND,
  TRITON_MOE_BACKEND,
  HUMMING_MOE_BACKEND
};

__attribute__((unused)) static std::string GetMoeComputeBackendString(MoeComputeBackend moe_compute_backend) {
  static const std::unordered_map<MoeComputeBackend, std::string> moe_compute_backend_map{
      {DEFAULT_MOE_BACKEND, "default_moe"},
      {CUTLASS_MOE_BACKEND, "cutlass_moe"},
      {MARLIN_MOE_BACKEND, "marlin_moe"},
      {TRITON_MOE_BACKEND, "triton_moe"},
      {HUMMING_MOE_BACKEND, "humming_moe"}};
  return moe_compute_backend_map.count(moe_compute_backend) ? moe_compute_backend_map.at(moe_compute_backend)
                                                            : "Unknown";
}

enum class DistributedCommunicationType {
  DEFAULT = 0,   // send and recevie
  SCATTER = 1,   // scatter
  ALLTOALL = 2,  // all to all
};

// Pooling type for embedding models.
enum class PoolingType {
  LAST = 0,  // Extract last token hidden state (Qwen3-Embedding, GTE-Qwen, etc.)
  CLS = 1,   // Extract first token (CLS) hidden state (BERT, BGE, etc.)
  MEAN = 2,  // Average all token hidden states (E5, Jina, etc.)
};

inline std::string GetPoolingTypeString(PoolingType pooling_type) {
  static const std::unordered_map<PoolingType, std::string> pooling_type_map{
      {PoolingType::LAST, "last"}, {PoolingType::CLS, "cls"}, {PoolingType::MEAN, "mean"}};
  return pooling_type_map.count(pooling_type) ? pooling_type_map.at(pooling_type) : "Unknown";
}

// The Quant informations.
struct QuantConfig {
  // The quant method
  QuantMode method = QUANT_NONE;

  // (gptq/awq) The quant bits
  size_t bits = 4;

  // (gptq/awq) The quant group size
  size_t group_size = 128;

  // (fp8-blockwise) The quant block size
  std::vector<size_t> weight_block_size;

  // (gptq) The desc act mode
  bool desc_act = false;

  bool input_scale = false;

  LinearComputeBackend backend = DEFAULT_LINEAR_BACKEND;

  // (fp8) Whether weight_scale shape is empty.
  bool is_fp8_blockwise = false;

  // (fp8) Whether weight is quantized in checkpoint.
  bool is_checkpoint_fp8_serialized = false;

  // (fp8) Whether input_scale is in checkpoint.
  bool is_activation_scheme_static = false;

  // (fp8) Whether enable int4 moe in fp8 model.
  bool enable_moe_int4 = false;

  // (fp8) format for fp8 scales
  //       "none" (default, standard float32 scale, e.g., GLM-5)
  //       "ue8m0" (uint8 exponential, powers-of-2 scales, e.g., DeepSeek-V3.2)
  std::string scale_fmt{"none"};

  // Whether this is a compressed-tensors quantized model.
  // compressed-tensors INT4 weights are stored pre-transposed ([N, K] layout, i.e. output-major),
  // unlike GPTQ which stores weights in [K, N] layout (input-major).
  // When true, MoE layer must use CUTLASS backend which supports pre-transposed weights,
  // rather than Triton backend which expects standard [K, N] layout.
  bool is_compressed_tensors = false;

  // Adaptation layers
  std::vector<std::string> pattern_layers;

  // Ignored layers
  std::vector<std::string> ignored_layers;

  // for deepseek v4 fp4 moe
  std::string expert_dtype{"none"};
};

// The Moe informations.
struct MoeConfig {
  size_t num_experts{1};
  size_t num_shared_experts{0};

  size_t experts_topk;

  size_t moe_inter_size;
  size_t first_k_dense_replace = 0;
  size_t shared_expert_inter_size = 0;

  // For group topk
  uint32_t num_expert_group = 1;
  uint32_t expert_groups_topk = 1;
  std::string scoring_func = "softmax";
  std::string topk_method = "greedy";
  bool norm_topk_prob = false;
  bool use_e_score_correction_bias = false;
  float routed_scaling_factor = 1.0f;

  // TODO(winminkong): 增加临时算子选择项，后期改为配置项
  bool use_vllm_moe = false;

  // for llama4
  size_t interleave_moe_layer_step = 1;
  // when is_moe == true,
  // if moe_layers.empty() means all layers' mlp use moe,
  // else only layer_idx in moe_layers use moe.
  std::vector<size_t> moe_layers;
  bool output_router_logits = true;
  float router_aux_loss_coef = 0;
  float router_jitter_noise = 0;
  bool apply_weight = false;
};

// The MLA informations.
struct MlaConfig {
  uint32_t q_lora_rank = 0;
  uint32_t kv_lora_rank = 0;

  uint32_t qk_nope_head_dim = 0;
  uint32_t qk_rope_head_dim = 0;
  uint32_t v_head_dim = 0;
};

// The DeepSeek Sparse MLA informations.
struct DsaConfig {
  uint32_t index_head_dim = 0;
  uint32_t index_n_heads = 0;
  uint32_t index_topk = 0;
  bool indexer_rope_interleave = false;
  bool index_share_for_mtp_iteration = false;
  uint32_t index_skip_topk_offset = 1;
  uint32_t index_topk_freq = 1;
};

// The DeepSeek v4 mHC informations.
struct MhcConfig {
  uint32_t hc_mult = 1;
  uint32_t hc_sinkhorn_iters = 0;
  float hc_eps = 0.0f;
};

// The DeepSeek v4 attn informations.
struct Dsv4AttnConfig {
  uint32_t o_groups = 0;
  uint32_t o_lora_rank = 0;
  uint32_t q_lora_rank = 0;
  uint32_t qk_rope_head_dim = 0;
  uint32_t sliding_window = 0;
  float compress_rope_theta{160000.0f};
  std::vector<uint32_t> compress_ratios;
};

// The DSpark (Draft Speculation with Parallel Ranking/Knowledge) config.
struct DsparkConfig {
  uint32_t dspark_block_size = 0;  // 0 means DSpark is disabled.
  uint32_t dspark_noise_token_id = 0;
  std::unordered_set<uint32_t> dspark_target_layer_ids;
  uint32_t dspark_markov_rank = 256;
};

// The model informations.
struct ModelConfig {
  // The model name.
  std::string name = "";

  // The model type, such as llama.
  std::string type;

  // The dir path.
  std::string path;

  std::string tokenizer_path;

  // Type of weight
  DataType weight_data_type;

  // Max sequence length during model training (unit: token num)
  size_t max_training_seq_len;

  size_t head_num;
  uint32_t size_per_head;
  uint32_t inter_size;
  uint32_t hidden_units;
  uint32_t num_layer;
  uint32_t num_nextn_predict_layers = 0;
  uint32_t rotary_embedding;
  float rope_theta;
  float layernorm_eps;
  uint32_t vocab_size;
  uint32_t start_id;
  std::vector<uint32_t> end_ids;
  uint32_t pad_id;
  size_t num_key_value_heads;  // nobody use it, to be removed
  int max_position_embeddings;

  // Parallel Token Prediction (PTP)
  size_t ptp_step_num;
  uint32_t ptp_token_id;

  std::vector<float> k_scales;  // to be removed
  std::vector<float> v_scales;  // to be removed

  RoPEScalingFactor rope_scaling_factor_config;

  bool tie_word_embeddings;
  bool exist_tie_embeddings_param = true;

  // SwiGLU activation clamp (DeepSeek-V4 起引入).
  // swiglu_limit > 0 时,在 SiLU*Mul 前 clamp:
  //   gate (前半): clamp(max = swiglu_limit)
  //   up   (后半): clamp(min = -swiglu_limit, max = swiglu_limit)
  float swiglu_limit = 0.0f;

  // The activation function used.
  std::string activation_function{"swiglu"};

  // Determines if the model is a visual llm model.
  bool is_visual = false;

  // Determines if the model is an embedding model.
  bool is_embedding_model = false;

  // Pooling type for embedding models (LAST, CLS, MEAN).
  PoolingType pooling_type = PoolingType::LAST;

  // Whether to apply L2 normalization to embedding output.
  bool normalize_embedding = true;

  // Determines if the model is a quant model.
  bool is_quant;
  QuantConfig quant_config;
  std::vector<QuantConfig> sub_quant_configs;

  // Determines if the model is a moe model.
  bool is_moe = false;
  bool has_shared_experts = false;
  MoeConfig moe_config;

  // others attributes
  std::unordered_map<std::string, std::string> model_attributes;

  ModelFileFormat model_file_format;

  bool load_bias = true;     // Check if load all weights bias.
  int cla_share_factor = 0;  // Determines the number of layers that share k and v.
  bool use_cla = false;
  // NOTE(jinxcwu) 这个参数不是从配置解析的，是在创建Model/Model函数内等位置手动指定的
  bool use_qk_norm = false;  // Check if normlize the attention out q and k.
  bool mlp_bias = false;     // Check if use bias in mlp layer.

  // For mla model
  bool use_mla = false;
  MlaConfig mla_config;
  // For dsa model
  bool use_dsa = false;
  DsaConfig dsa_config;
  // For dsv4 model
  bool use_dsv4_attn = false;
  Dsv4AttnConfig dsv4_attn_config;
  // For mhc model
  bool use_mhc = false;
  MhcConfig mhc_config;
  // For dspark model
  DsparkConfig dspark_config;

  // For arc-hunyuan-video
  // TODO(jinxcwu) 后续要简化下，不是所有都需要的
  uint32_t eod_token_id = 0;
  uint32_t im_end_id = 0;
  uint32_t im_newline_id = 0;
  uint32_t im_start_id = 0;
  uint32_t image_token_id = 0;
  bool position_embedding_xdrope = false;

  // Number of DeepStack injection layers (from vision_config.deepstack_visual_indexes).
  // 0 means no DeepStack injection (non-VL or Qwen2-VL).
  int num_deepstack_layers = 0;

  // Whether to normalize q and k before rotary position embedding in attention.
  bool enable_qk_pre_norm_before_rotary_pos = false;  // to be removed

  bool enable_add_qkv_bias = false;  // TODO(robertyuan): this is a model config, not func switch. Change name

  // for llama4
  std::vector<size_t> no_rope_layers;
  size_t attn_temperature_tuning = 0;
  float attn_scale = 0;
  size_t floor_scale = 0;
  size_t attention_chunk_size = 0;

  // The word embedding scale factor.
  float emb_scale = 1.f;
  // Scaling the hidden states of residual connections.
  float scale_depth = 1.f;
  // For choose lm_head.linear_head weight (for bge-reranker-v2-minicpm-layerwise)
  int start_layer = 0;

  // for deepseek_v4
  uint32_t num_hash_layers = 0;
};

}  // namespace ksana_llm
