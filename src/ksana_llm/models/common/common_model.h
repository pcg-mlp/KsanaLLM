/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/data_hub/hidden_unit_buffer.h"
#include "ksana_llm/layers/add_layer.h"
#include "ksana_llm/layers/assemble_last_token_layer.h"
#include "ksana_llm/layers/base_layer.h"
#include "ksana_llm/layers/cast_layer.h"
#include "ksana_llm/layers/emb_lookup_layer.h"
#include "ksana_llm/layers/flash_attention_layer.h"
#include "ksana_llm/layers/group_matmul_layer.h"
#include "ksana_llm/layers/input_refit_layer.h"
#include "ksana_llm/layers/layernorm_layer.h"
#include "ksana_llm/layers/matmul_layer_factory.h"
#include "ksana_llm/layers/paged_attention_layer.h"
#include "ksana_llm/layers/silu_mul_layer.h"
#include "ksana_llm/runtime/infer_stage.h"
#ifdef ENABLE_VLLM_FLASH_ATTN_2
#  include "ksana_llm/layers/set_torch_stream_layer.h"
#endif
#include "ksana_llm/models/base/base_model.h"
#include "ksana_llm/models/base/model_communicator.h"
#include "ksana_llm/models/base/model_input.h"
#include "ksana_llm/models/base/model_output.h"
#include "ksana_llm/models/common_moe/moe_config.h"
#include "ksana_llm/models/llama/llama_weight.h"
#include "ksana_llm/utils/context.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/optional_file.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/tensor.h"
#include "ksana_llm/utils/utils.h"
#ifdef ENABLE_CUDA
#  include "ksana_llm/runtime/cuda_graph_runner.h"
#endif
namespace ksana_llm {

// The layernorm position type.
enum class LayerNormPosition { PRE_NORM = 0, POST_NORM = 1 };

// Describe the model architecture.
struct ModelRunConfig {
  // The model position embedding.
  PositionEncoding position_encoding = PositionEncoding::ROPE;
  // The rope parameters.
  bool is_neox = true;
  // The word embedding scale factor.
  float emb_scale = 1.f;
  // Whether add a bias on qkv output.
  bool qkv_add_bias = false;
  // Use pre-norm or post-norm.
  LayerNormPosition layernorm_position = LayerNormPosition::PRE_NORM;
  // Is run inference with graph
  bool is_graph_mode = false;
  // Use no_norm or re_norm.
  MoeScaleNormMode moe_scale_norm_mode = MoeScaleNormMode::NO_NORM;
};

// A common implement of transformer based model.
template <typename T>
class __attribute__((visibility("hidden"))) CommonModel : public BaseModel {
 public:
  CommonModel(const ModelConfig& model_config, const int rank, std::shared_ptr<Context> context);
  ~CommonModel() override;

  // Initialize the run config.
  void InitRunConfig(const ModelRunConfig& model_run_config, std::shared_ptr<BaseWeight> base_weight);

  float* GetLogitsPtr() override;

  // refer
  // github huggingface/transformers main/src/transformers/models/llama/modeling_llama.py#L942
  Status Forward(std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                 std::vector<ForwardRequest>& forward_reqs, bool epilogue) override;

  // Update response. Stop inference when the return value is true.
  bool UpdateResponse(std::vector<ForwardRequest>& forward_reqs, Tensor& output, const std::string& stage);

 private:
  // Execute the embedding lookup.
  Status LookupEmbedding(std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                         std::vector<ForwardRequest>& forward_reqs);

  // Execute the forward of specific layers.
  Status LayerForward(std::shared_ptr<ksana_llm::BaseWeight>& base_weight);

  // Execute the lm head, and generate the logits.
  Status LmHead(std::shared_ptr<ksana_llm::BaseWeight>& base_weight, std::vector<ForwardRequest>& forward_reqs);

 protected:
  using BaseModel::context_;
  using BaseModel::rank_;

  // Whether auto prefix caching is enabled.
  bool prefix_caching_enabled_;

  // The model config.
  ModelConfig model_config_;

  // The pipeline_config for distributed mode.
  PipelineConfig pipeline_config_;

  // The model run config.
  ModelRunConfig model_run_config_;

  // The model input information.
  std::shared_ptr<ModelInput> model_input_;

  // The model output.
  std::shared_ptr<ModelOutput> model_output_;

  // The model communicator.
  std::shared_ptr<ModelCommunicator<T>> model_communicator_;

  std::shared_ptr<BaseLayer> emb_lookup_layer_;
  std::shared_ptr<BaseLayer> cpu_emb_lookup_layer_;
  std::shared_ptr<BaseLayer> layernorm_layer_;
  std::vector<std::shared_ptr<BaseLayer>> flash_attention_layers_;
  std::vector<std::shared_ptr<BaseLayer>> paged_attention_layers_;
  std::shared_ptr<BaseLayer> add_layer_;
  std::shared_ptr<BaseLayer> silu_mul_layer_;
  std::shared_ptr<BaseLayer> assemble_last_token_layer_;
  std::shared_ptr<BaseLayer> cast_layer_;
  std::shared_ptr<BaseLayer> input_refit_layer_;
#ifdef ENABLE_VLLM_FLASH_ATTN_2
  std::shared_ptr<BaseLayer> set_torch_stream_layer_;
#endif

  std::shared_ptr<MatMulLayerFactory<T>> matmul_layer_factory_;
  std::shared_ptr<BaseLayer> attn_qkv_proj_layer_;
  std::shared_ptr<BaseLayer> attn_o_proj_layer_;
  std::shared_ptr<BaseLayer> mlp_gate_proj_layer_;
  std::shared_ptr<BaseLayer> mlp_up_proj_layer_;
  std::shared_ptr<BaseLayer> mlp_down_proj_layer_;
  std::shared_ptr<BaseLayer> lm_head_proj_layer_;

#ifdef ENABLE_CUDA
  std::shared_ptr<CudaGraphRunner> cudagraph_runner;
#endif

  // The layer number of the model on current node.
  int layer_num_on_node_;

  // The original vocab size of the model
  size_t vocab_size_;

  // Vocab size aligned and padded with tensor_para_size
  size_t vocab_size_pad_;

  // Whether cudagraph is enabled
  bool is_cudagraph_enabled = false;

  /**
   * The following 4 buffers are used as temporary buffers during the whole model inference:
   * 1. `hidden_buffer_0_` and `hidden_buffer_1_` serve as the input and output for each layer.
   *    We assume that the input of each layer is taken from `hidden_buffer_0_`, the output is
   *    put into `hidden_buffer_1_`, and then swapped with `hidden_buffer_0_`. This convention
   *    makes each layer independent and pluggable.
   * 2. `residual_buffer_` is dedicated to storing the output used for residual connection.
   * 3. `shared_buffer_` is shared to store the output of the up layer for gated activation
   *    (`gated_buffer_`), as the fixed input buffer for custom reduce sum (`reduce_buffer_`),
   *    and as the extra workspace for paged attention (`paged_buffer_`).
   *
   * Below is an example of how these buffers are used:
   *
   * ```
   * // The input for layernorm is used for residual connection, so is stored in `residual_buffer_`.
   * layernorm_layer->Forward(residual_buffer_, hidden_buffer_0_);
   *
   * // Normally, a non-inplace layer takes input from `hidden_buffer_0_`, puts the output to `hidden_buffer_1_`.
   * attention_layer->Forward(hidden_buffer_0_, hidden_buffer_1_);
   * // Swap `hidden_buffer_1_` with `hidden_buffer_0` to ensure the next layer's input is in `hidden_buffer_0_`.
   * std::swap(hidden_buffer_1_, hidden_buffer_0_);
   *
   * // Residual layer takes the input from `hidden_buffer_0_` and `residual_buffer_`, and places the ouput
   * // to `residual_buffer_` for the residual connection after mlp.
   * residual_add_layer->Forward({hidden_buffer_0_[0], residual_buffer_[0]}, residual_buffer_);
   * ```
   *
   */
  std::vector<Tensor> hidden_buffer_0_{1};
  std::vector<Tensor> hidden_buffer_1_{1};
  std::vector<Tensor> residual_buffer_{1};
  std::vector<Tensor> shared_buffer_{1};
  std::vector<Tensor>& gated_buffer_{shared_buffer_};
  std::vector<Tensor>& reduce_buffer_{shared_buffer_};
  std::vector<Tensor>& paged_buffer_{shared_buffer_};
  std::vector<Tensor> cuda_graph_input_{1};
  std::vector<Tensor> cuda_graph_output_{1};

  Tensor forward_shape_;

  // This tensor is used to store boolean flag variables.
  Tensor flag_tensor_;

  Tensor cos_sin_cache_tensor_;
  Tensor cpu_input_tokens_tensor_;
  Tensor cpu_tokens_emb_tensor_;

  std::shared_ptr<Tensor> shared_matmul_workspace_buffer_ = nullptr;

 protected:
  Status CreateProjLayer(std::shared_ptr<BaseWeight>& base_weight);

  Status AddAttentionPrefixCache();

  Status RemoveAttentionPrefixCache();

  bool IsPrefixCachingComputationReuse();

  virtual Status FlashAttentionForward(const int layer_idx);

  virtual Status PagedAttentionForward(const int layer_idx);

  virtual Status LayerNormForward(const std::string& layer_name, std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                                  const std::vector<Tensor>& layernorm_input, std::vector<Tensor>& layernorm_output);

  // refer to
  // github huggingface/transformers main/src/transformers/models/llama/modeling_llama.py#L257
  virtual Status CommonAttention(const int layer_idx, std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                                 const std::vector<Tensor>& attention_input, const bool is_multi_token_forward);

  // refer to
  // github huggingface/transformers main/src/transformers/models/llama/modeling_llama.py#L211
  virtual Status CommonMlp(const int layer_idx, std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                           const std::vector<Tensor>& mlp_input, const bool is_multi_token_forward);

  // refer to
  // github huggingface/transformers main/src/transformers/models/llama/modeling_llama.py#L694
  Status CommonDecoderPreNorm(const int layer_idx, std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                              const bool is_multi_token_forward);

  // refer to
  // github huggingface/transformers main/src/transformers/models/openai/modeling_openai.py#L247
  Status CommonDecoderPostNorm(const int layer_idx, std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                               const bool is_multi_token_forward);

  Status EmbedTokensUseCpu(Tensor& embedding_weight, std::vector<ForwardRequest>& forward_reqs);

  virtual Status EmbedTokensUseGpu(Tensor& embedding_weight);

  // Load embeddings obtained from external input or computed by plugins.
  virtual Status LoadEmbeddings(std::vector<ForwardRequest>& forward_reqs);

  // Copy between hidden_unit_buffer and tensor.
  Status CopyFromHiddenUnitBuffer(Tensor& tensor, HiddenUnitDeviceBuffer* device_buffer);
  Status CopyToHiddenUnitBuffer(HiddenUnitDeviceBuffer* device_buffer, Tensor& tensor);
};

}  // namespace ksana_llm
