/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <future>
#include <memory>
#include <string>

#include "cache_manager/cache_layout/base_cache_layout.h"
#include "configure/context.h"
#include "configure/environment.h"
#include "kernels/cast.h"
#include "models/base/base_model.h"
#include "runtime/forward_request.h"
#include "runtime/hidden_unit_buffer.h"
#include "runtime/infer_stage.h"
#include "runtime/profile_metrics.h"
#include "runtime/threadpool.h"
#include "runtime/weight_instance_inferface.h"
#include "utils/search_path.h"
#include "utils/singleton.h"
#include "utils/tensor.h"

namespace ksana_llm {

class ModelInstance {
 public:
  ModelInstance(const ModelConfig& model_config, const RuntimeConfig& runtime_config, std::shared_ptr<Context> context,
                std::shared_ptr<WeightInstanceInterface>& weight_instance)
      : model_config_(model_config),
        runtime_config_(runtime_config),
        context_(context),
        weight_instance_(weight_instance) {
    loader_models_threadpool_ = std::make_shared<ThreadPool>(context->global->GetTensorParallelSize());
    loader_models_threadpool_->Start();
  }

  ~ModelInstance() { loader_models_threadpool_->Stop(); }

  // Load model with specified model config.
  void Load();

  // Set the cache layout for specific model instance.
  void SetBlockCacheLayout(std::shared_ptr<BaseModelCacheLayout> block_cache_layout);

  // The instance name.
  std::string name;
  std::string type;

  // model_input_slot: 见 NewModelInput::slot_index 注释；slow path 默认 0。
  // prev_ring_dev: 见 NewModelInput::prev_ring_dev 注释；非 nullptr 即触发 ReplaceLastTokenKernel (slow path 默认
  // nullptr)。
  Status Forward(std::shared_ptr<BaseWeight> weight, InferStage stage, std::vector<ForwardRequest>& forward_reqs,
                 HiddenUnitDeviceBuffer* hidden_unit, RunMode run_mode, size_t model_input_slot = 0,
                 const int64_t* prev_ring_dev = nullptr, bool gpu_input_already_prepared = false);

  Status PrepareModelInputGpu(std::vector<ForwardRequest>& forward_reqs, RunMode run_mode, size_t model_input_slot);

  // 取本步 slot 对应 ModelInput 的 cur_batch_pairs_tensor 设备指针 + batch_size，
  // 供 ModelExecutor::Forward 完成后透传到 NewModelOutput（WriteRing / ReplaceKernel 使用）。
  void GetCurBatchPairs(size_t model_input_slot, const int64_t** out_pairs_dev, int* out_batch_size);

  // Set the inflight resource manager through to the base model.
  // If the model is already loaded, also triggers RegisterInflightResources.
  void SetInflightResourceManager(InflightResourceManager* mgr);

  // 将 ProfileMetrics 及当前子 metrics 指针透传到底层 BaseModel。
  void SetForwardProfileMetrics(ProfileMetrics* pm, ForwardPerfMetrics* current);

  // Get  the data type.
  DataType GetWeightDataType() { return model_config_.weight_data_type; }

  // Get the base ptr of model's logits buf.
  float* GetLogitsPtr();

  const ModelConfig& GetModelConfig() { return model_config_; }

  size_t GetMaxTokenNum() { return runtime_config_.max_seq_len; }

  uint32_t GetLayerNum() {
    if (pipeline_config_.upper_layer_idx < 0 || pipeline_config_.lower_layer_idx < 0) {
      Singleton<Environment>::GetInstance()->GetPipelineConfig(pipeline_config_);
    }
    uint32_t layer_num = pipeline_config_.upper_layer_idx - pipeline_config_.lower_layer_idx + 1;
    if (pipeline_config_.lower_nextn_layer_idx >= static_cast<int>(model_config_.num_layer)) {
      layer_num += pipeline_config_.upper_nextn_layer_idx - pipeline_config_.lower_nextn_layer_idx + 1;
    }
    return layer_num;
  }

  void Reset() {
    KLLM_LOG_INFO << "ModelInstance::Reset clear models and weights.";
    weight_instance_ = nullptr;
  }

  // Get the base model instance
  std::shared_ptr<BaseModel> GetBaseModel() { return model_instance_; }

 private:
  // The model config.
  ModelConfig model_config_;
  RuntimeConfig runtime_config_;

  PipelineConfig pipeline_config_;

  // The global context.
  std::shared_ptr<Context> context_ = nullptr;
  std::shared_ptr<WeightInstanceInterface> weight_instance_;

  // The base modes
  std::shared_ptr<BaseModel> model_instance_;
  std::shared_ptr<ThreadPool> loader_models_threadpool_ = nullptr;

  // Set once before or after model loading; used to auto-register inflight resources
  // as soon as the base model becomes available.
  InflightResourceManager* inflight_resource_mgr_ = nullptr;

  // The model-level cache layout (holds all block cache layouts keyed by type).
  std::shared_ptr<BaseModelCacheLayout> block_cache_layout_ = nullptr;
};

}  // namespace ksana_llm
