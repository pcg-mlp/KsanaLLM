/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/layers/add_layer.h"
#include "ksana_llm/layers/assemble_last_token_layer.h"
#include "ksana_llm/layers/cast_layer.h"
#include "ksana_llm/layers/emb_lookup_layer.h"
#include "ksana_llm/layers/flash_attention_layer.h"
#include "ksana_llm/layers/layernorm_layer.h"
#include "ksana_llm/layers/matmul_layer_factory.h"
#include "ksana_llm/layers/paged_attention_layer.h"
#include "ksana_llm/layers/silu_mul_layer.h"
#include "ksana_llm/layers/input_refit_layer.h"
#include "ksana_llm/utils/context.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/optional_file.h"

#include "ksana_llm/models/base/base_model.h"
#include "ksana_llm/models/base/model_communicator.h"
#include "ksana_llm/models/base/model_input.h"
#include "ksana_llm/models/base/model_output.h"
#include "ksana_llm/models/llama/llama_weight.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/tensor.h"
#include "ksana_llm/utils/utils.h"

namespace ksana_llm {

// The positional encoding.
enum PositionEncoding { ROPE = 0, ALIBI = 1 };

// Describe the model architecture.
struct ModelRunConfig {
  // The model position embedding
  PositionEncoding position_encoding = PositionEncoding::ROPE;

  // Whether add a bias on qkv output.
  bool qkv_add_bias = false;
};

// A common implement of transfer based model.
template <typename T>
class __attribute__((visibility("hidden"))) CommonModel : public BaseModel {
 public:
  CommonModel(const ModelConfig& model_config, const int rank, std::shared_ptr<Context> context);
  ~CommonModel();

  // Initialize the run config.
  void InitRunConfig(const ModelRunConfig& model_run_config, std::shared_ptr<BaseWeight> base_weight);

  float* GetLogitsPtr();

  // The prefill stage.
  Status ContextDecode(std::shared_ptr<ksana_llm::BaseWeight>& base_weight, std::vector<ForwardRequest>& forward_reqs);

  // The decode stage.
  Status Decode(std::shared_ptr<ksana_llm::BaseWeight>& base_weight, std::vector<ForwardRequest>& forward_reqs);

 public:
  // plugin name
  std::string plugin_name = "";

 private:
  using BaseModel::context_;
  using BaseModel::rank_;

  // The model config.
  ModelConfig model_config_;

  // The model input information.
  std::shared_ptr<ModelInput> model_input_;

  // The model output.
  std::shared_ptr<ModelOutput> model_output_;

  // The model communicator.
  std::shared_ptr<ModelCommunicator<T>> model_communicator_;

  std::shared_ptr<EmbLookupLayer<T>> emb_lookup_layer_;
  std::shared_ptr<CpuEmbLookupLayer<T>> cpu_emb_lookup_layer_;
  std::shared_ptr<LayernormLayer<T>> layernorm_layer_;
  std::vector<std::shared_ptr<FlashAttentionLayer<T>>> flash_attention_layers_;
  std::vector<std::shared_ptr<PagedAttentionLayer<T>>> paged_attention_layers_;
  std::shared_ptr<AddLayer<T>> add_layer_;
  std::shared_ptr<SiluMulLayer<T>> silu_mul_layer_;
  std::shared_ptr<BaseLayer> attn_qkv_proj_layer_;
  std::shared_ptr<BaseLayer> attn_o_proj_layer_;
  std::shared_ptr<BaseLayer> mlp_gate_proj_layer_;
  std::shared_ptr<BaseLayer> mlp_up_proj_layer_;
  std::shared_ptr<BaseLayer> mlp_down_proj_layer_;
  std::shared_ptr<BaseLayer> lm_head_proj_layer_;
  std::shared_ptr<AssembleLastTokenLayer<T>> assemble_last_token_layer_;
  std::shared_ptr<CastLayer<T>> cast_layer_;
  std::shared_ptr<InputRefitLayer<T>> input_refit_layer_;

  std::shared_ptr<py::object> plugin_;

  // The layer number of the model
  int num_layer_;

  // Vocab size aligned and padded with tensor_para_size
  size_t vocab_size_pad_;

  // Whether to add bias values during the QKV calculation.
  bool qkv_add_bias_;

  // Whether or not the model use GQA
  bool is_gqa_;

  Tensor tensor_buffer_0_;
  Tensor tensor_buffer_1_;
  Tensor tensor_buffer_2_;
  Tensor up_matmul_tensor_buffer_;
  Tensor forward_shape_;
  Tensor cos_sin_cache_tensor_;
  Tensor cpu_input_tokens_tensor_;
  Tensor cpu_tokens_emb_tensor_;

#ifdef ENABLE_ACL
  // Used for ascend attention.
  Tensor ascend_buffer_0_;
  Tensor ascend_buffer_1_;
  Tensor ascend_buffer_2_;
  Tensor ascend_buffer_3_;
  Tensor ascend_buffer_4_;

  std::vector<Tensor> ascend_key_caches_;
  std::vector<Tensor> ascend_val_caches_;
#endif

 private:
  Status AddPrefixCache(std::vector<Tensor>& mmha_origin_input, std::vector<Tensor>& mmha_prefix_input);

  Status RemovePrefixCache(std::vector<Tensor>& mmha_prefix_output, std::vector<Tensor>& mmha_output);

  bool IsPrefixCachingComputationReuse();

  Status FlashAttentionForward(std::vector<Tensor>& flash_attention_input, std::vector<Tensor>& flash_attention_output,
                               int layer_idx);

  // refer to
  // github huggingface/transformers main/src/transformers/models/llama/modeling_llama.py#L257
  Status CommonAttention(const int layer_idx, std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                         Tensor& hidden_states, std::vector<Tensor>& output_0, std::vector<Tensor>& output_1,
                         std::vector<Tensor>& output_2, const bool is_context_stage);

  // refer to
  // github huggingface/transformers main/src/transformers/models/llama/modeling_llama.py#L211
  Status CommonMlp(const int layer_idx, std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                   Tensor& post_layernorm_output, std::vector<Tensor>& output_0, std::vector<Tensor>& output_1,
                   std::vector<Tensor>& output_2);

  // refer to
  // github huggingface/transformers main/src/transformers/models/llama/modeling_llama.py#L694
  Status CommonDecoder(const int layer_idx, std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                       std::vector<Tensor>& temp_buffer_0, std::vector<Tensor>& temp_buffer_1,
                       std::vector<Tensor>& temp_buffer_2, const bool is_context_stage);

  // refer
  // github huggingface/transformers main/src/transformers/models/llama/modeling_llama.py#L942
  Status CommonForward(std::shared_ptr<ksana_llm::BaseWeight>& base_weight, std::vector<ForwardRequest>& forward_reqs,
                       const bool is_context_stage);

  Status EmbedTokensUseCpu(Tensor& embedding_weight, std::vector<ForwardRequest>& forward_reqs,
                           const bool is_context_stage, std::vector<Tensor>& temp_buffer_0);

  Status PythonPluginPreproces(std::vector<ForwardRequest>& forward_reqs);
};

}  // namespace ksana_llm
