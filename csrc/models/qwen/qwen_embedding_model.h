/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "cache_manager/cache_layout/base_cache_layout.h"
#include "models/common/common_model.h"
#include "models/qwen/qwen_model.h"

namespace ksana_llm {

// Embedding model built on top of Qwen/Qwen3 decoder layers.
// Overrides LmHead() to perform:
// FinalNorm → Pooling (LAST/CLS/MEAN) → optional L2Normalize → write embedding to PythonTensor response.
// Pooling type and normalize behavior are controlled by ModelConfig::pooling_type and
// ModelConfig::normalize_embedding, parsed from 1_Pooling/config.json or YAML config.
class QwenEmbeddingModel : public CommonModel {
 public:
  QwenEmbeddingModel(const ModelConfig& model_config, const RuntimeConfig& runtime_config, const int rank,
                     std::shared_ptr<Context> context, std::shared_ptr<BaseWeight> base_weight,
                     std::shared_ptr<BaseModelCacheLayout> block_cache_layout);
  ~QwenEmbeddingModel();

  // Non-copyable, non-movable (pinned_host_buffer_ is a raw CUDA allocation).
  QwenEmbeddingModel(const QwenEmbeddingModel&) = delete;
  QwenEmbeddingModel& operator=(const QwenEmbeddingModel&) = delete;

 protected:
  using CommonModel::context_;
  using CommonModel::GetHiddenUnitBufferTensors;
  using CommonModel::model_config_;
  using CommonModel::rank_;

 private:
  Status CreateLayers(LayerCreationContext& creation_context, ModelCreationConfig& model_creation_config) override;
  Status LayerForward(ForwardingContext& forwarding_context, const RunMode run_mode = RunMode::kMain) override;
  Status LmHead(ForwardingContext& forwarding_context, std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                std::vector<ForwardRequest>& forward_reqs, RunMode run_mode) override;

  // Copy embedding result into forward request response PythonTensor.
  void UpdateEmbeddingResponse(std::vector<ForwardRequest>& forward_reqs, Tensor& output);

  Qwen qwen_;

  // Pinned host memory buffer for async D2H copy of embedding results.
  // Pre-allocated once in constructor and reused across forward calls.
  void* pinned_host_buffer_ = nullptr;
  size_t pinned_host_buffer_size_ = 0;
};

}  // namespace ksana_llm
