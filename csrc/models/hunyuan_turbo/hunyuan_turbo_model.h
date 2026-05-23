/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "models/common/common_model.h"

#include "cache_manager/cache_layout/base_cache_layout.h"
#include "models/communicator/tp_communicator.h"
#include "modules/attention/cross_layer_attention.h"
#include "modules/ffn/two_layered_ffn.h"

#include "models/common/simple_decoder_layer.h"

#include "modules/basic/moe.h"

namespace ksana_llm {

class HunyuanTurboModel : public CommonModel {
 public:
  HunyuanTurboModel(const ModelConfig& model_config, const RuntimeConfig& runtime_config, const int rank,
                    std::shared_ptr<Context> context, std::shared_ptr<BaseWeight> base_weight,
                    std::shared_ptr<BaseModelCacheLayout> block_cache_layout);

 private:
  Status CreateLayers(LayerCreationContext& creation_context, ModelCreationConfig& model_creation_config) override;

  // Execute the forward of specific layers.
  Status LayerForward(ForwardingContext& forwarding_context, const RunMode run_mode = RunMode::kMain) override;

 protected:
  using CommonModel::GetHiddenUnitBufferTensors;

 private:
  // for cla (if the model not use cross of attention, default nullptr)
  int cla_share_factor_;
  ClaBuffers cla_buffers_;
  TensorBuffer* moe_buffer_;

  std::map<int, std::shared_ptr<SimpleDecoderLayer>> decoder_layers_;
};

}  // namespace ksana_llm
