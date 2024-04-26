/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/layers/add_layer.h"
#include "ksana_llm/layers/assemble_last_token_layer.h"
#include "ksana_llm/layers/cast_layer.h"
#include "ksana_llm/layers/emb_lookup_layer.h"
#include "ksana_llm/layers/flash_attention_layer.h"
#include "ksana_llm/layers/layernorm_layer.h"
#include "ksana_llm/layers/matmul_layer.h"
#include "ksana_llm/layers/paged_attention_layer.h"
#include "ksana_llm/layers/silu_mul_layer.h"
#include "ksana_llm/utils/context.h"
#include "ksana_llm/utils/environment.h"

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
class CommonModel : public BaseModel {
  public:
    CommonModel(const ModelConfig& model_config, const int rank, std::shared_ptr<Context> context);
    ~CommonModel();

    // Initialize the run config.
    void InitRunConfig(const ModelRunConfig& model_run_config);

    float* GetLogitsPtr();

    // The prefill stage.
    Status ContextDecode(std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                         std::vector<ForwardRequest>& forward_reqs);

    // The decode stage.
    Status Decode(std::shared_ptr<ksana_llm::BaseWeight>& base_weight, std::vector<ForwardRequest>& forward_reqs);

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
    std::shared_ptr<LayernormLayer<T>> layernorm_layer_;
    std::vector<std::shared_ptr<FlashAttentionLayer<T>>> flash_attention_layers_;
    std::vector<std::shared_ptr<PagedAttentionLayer<T>>> paged_attention_layers_;
    std::shared_ptr<AddLayer<T>> add_layer_;
    std::shared_ptr<SiluMulLayer<T>> silu_mul_layer_;
    std::shared_ptr<MatMulLayer<T>> matmul_layer_;
    std::shared_ptr<AssembleLastTokenLayer<T>> assemble_last_token_layer_;
    std::shared_ptr<CastLayer<T>> cast_layer_;

    int num_layer_;
    bool qkv_add_bias_;

    Tensor tensor_buffer_0_;
    Tensor tensor_buffer_1_;
    Tensor tensor_buffer_2_;
    Tensor up_matmul_tensor_buffer_;
    Tensor forward_shape_;
    Tensor cos_sin_cache_tensor_;

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
  // refer to
  // https://github.com/huggingface/transformers/blob/ \
  // 00c1d87a7d5c8dfb4554370983b5a3f7c069edd7/src/transformers/models/llama/modeling_llama.py#L257
  Status LlamaAttention(const int layer_idx, std::shared_ptr<ksana_llm::BaseWeight>& base_weight, Tensor& hidden_states,
                        std::vector<Tensor>& output_0, std::vector<Tensor>& output_1, std::vector<Tensor>& output_2,
                        const bool is_context_stage);

    // refer to
    // github huggingface/transformers main/src/transformers/models/llama/modeling_llama.py#L211
    Status LlamaMlp(const int layer_idx, std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                    Tensor& post_layernorm_output, std::vector<Tensor>& output_0, std::vector<Tensor>& output_1,
                    std::vector<Tensor>& output_2);

    // refer to
    // github huggingface/transformers main/src/transformers/models/llama/modeling_llama.py#L694
    Status LlamaDecoder(const int layer_idx, std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                        std::vector<Tensor>& temp_buffer_0, std::vector<Tensor>& temp_buffer_1,
                        std::vector<Tensor>& temp_buffer_2, const bool is_context_stage);

    // refer
    // github huggingface/transformers main/src/transformers/models/llama/modeling_llama.py#L942
    Status LlamaForward(std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                              std::vector<ForwardRequest>& forward_reqs, const bool is_context_stage);
};

}  // namespace ksana_llm
