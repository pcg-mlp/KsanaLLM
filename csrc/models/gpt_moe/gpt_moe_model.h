/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "models/common/model_interface.h"

#include "models/communicator/tp_communicator.h"
#include "modules/attention/multihead_attention.h"
#include "modules/basic/add.h"
#include "modules/basic/layernorm.h"
#include "modules/basic/linear.h"
#include "modules/basic/moe.h"

namespace ksana_llm {

class GptMoeDecoderLayer {
 public:
  GptMoeDecoderLayer(int layer_idx, LayerCreationContext& creation_context, ModelCreationConfig& model_creation_config);
  ~GptMoeDecoderLayer() = default;

  Status Forward(std::vector<Tensor>& residual_buffer, const bool is_multi_token_forward,
                 ForwardingContext& forwarding_context);

 private:
  int layer_idx_;
  std::unique_ptr<Add> adds_;
  std::unique_ptr<Layernorm> input_layernorms_;
  std::unique_ptr<Layernorm> post_attention_layernorms_;
  std::shared_ptr<TpCommunicator> tp_comm_;

  std::unique_ptr<MultiHeadAttention> mha_;
  std::unique_ptr<MoE> moes_;
  std::unique_ptr<Linear> expert_gates_;
};

class GptMoe : public ModelInterface {
 public:
  GptMoe() {}
  ~GptMoe() = default;

  Status GetModelRunConfig(ModelRunConfig& model_run_config, const ModelConfig& model_config) override;
  Status CreateLayers(LayerCreationContext& creation_context, ModelCreationConfig& model_creation_config) override;
  Status Forward(std::vector<Tensor>& residual_buffer, ForwardingContext& forwarding_context) override;

 private:
  std::unordered_map<int, std::unique_ptr<GptMoeDecoderLayer>> decoder_layers_;
};

class GptMoeModel : public CommonModel {
 public:
  GptMoeModel(const ModelConfig& model_config, const RuntimeConfig& runtime_config, const int rank,
              std::shared_ptr<Context> context, std::shared_ptr<BaseWeight> base_weight,
              std::shared_ptr<BaseModelCacheLayout> block_cache_layout);
  ~GptMoeModel() = default;

 private:
  Status CreateLayers(LayerCreationContext& creation_context, ModelCreationConfig& model_creation_config);
  Status LayerForward(ForwardingContext& forwarding_context, const RunMode run_mode = RunMode::kMain) override;

 protected:
  using CommonModel::GetHiddenUnitBufferTensors;

 private:
  GptMoe gptmoe_;
};

}  // namespace ksana_llm
