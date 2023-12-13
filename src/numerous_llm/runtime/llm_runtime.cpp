/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/runtime/llm_runtime.h"
#include "numerous_llm/runtime/worker.h"
#include "numerous_llm/utils/logger.h"

namespace numerous_llm {

template <typename Key, typename Value, typename T>
inline Value& GetMapValue(std::unordered_map<Key, Value>& m, const Key& key, T&& default_value) {
  return m.emplace(key, std::forward<T>(default_value)).first->second;
}

Status LlmRuntime::Step(std::vector<std::shared_ptr<InferRequest>>& reqs) {
  NLLM_LOG_INFO << "llm runtime step invoked.";

  // need group 3 things:
  // 1. model type(iden by model name)
  // 2. infer request stage
  std::unordered_map<std::string,
                     std::unordered_map<InferStage, std::vector<std::tuple<ModelInstance*, TensorMap*, TensorMap*>>>>
      grouped_reqs_map;

  for (std::shared_ptr<InferRequest> req_ptr : reqs) {
    std::unordered_map<InferStage, std::vector<std::tuple<ModelInstance*, TensorMap*, TensorMap*>>>&
        model_stage_tensor_map = GetMapValue(
            grouped_reqs_map, req_ptr->model_name,
            std::unordered_map<InferStage, std::vector<std::tuple<ModelInstance*, TensorMap*, TensorMap*>>>());
    std::vector<std::tuple<ModelInstance*, TensorMap*, TensorMap*>>& tensor_map_pair_vec =
        GetMapValue(model_stage_tensor_map, req_ptr->infer_stage,
                    std::vector<std::tuple<ModelInstance*, TensorMap*, TensorMap*>>());
    tensor_map_pair_vec.emplace_back(std::make_tuple<ModelInstance*, TensorMap*, TensorMap*>(
        req_ptr->model_instance.get(), &(req_ptr->input_tensor_map), &(req_ptr->output_tensor_map)));
  }

  // infer async
  for (auto& model_stage_tensor_map_it : grouped_reqs_map) {
    // model_stage_tensor_map_it instance of {std::string, std::unordered_map<InferStage,
    // std::vector<std::tuple<ModelInstance*, TensorMap*, TensorMap*>>>}
    NLLM_LOG_INFO << "llm runtime infer model: " << model_stage_tensor_map_it.first;
    std::shared_ptr<ModelInstance> model_instance_ptr{nullptr};
    for (auto& stage_tensor_map_it : model_stage_tensor_map_it.second) {
      // stage_tensor_map_it instance of {InferStage, std::vector<std::tuple<ModelInstance*, TensorMap*, TensorMap*>>}
      if (stage_tensor_map_it.second.empty()) {
        continue;
      }
      NLLM_LOG_INFO << "llm runtime infer model: " << model_stage_tensor_map_it.first
                    << " with stage: " << stage_tensor_map_it.first;
      if (model_instance_ptr == nullptr) {
        model_instance_ptr.reset(std::get<0>(stage_tensor_map_it.second[0]));
      }

      std::vector<TensorMap*> input_tensor_maps(stage_tensor_map_it.second.size(), nullptr);
      std::vector<TensorMap*> output_tensor_maps(stage_tensor_map_it.second.size(), nullptr);

      for (size_t req_tensor_map_idx = 0; req_tensor_map_idx < stage_tensor_map_it.second.size();
           ++req_tensor_map_idx) {
        input_tensor_maps[req_tensor_map_idx] = std::get<1>(stage_tensor_map_it.second[req_tensor_map_idx]);
        output_tensor_maps[req_tensor_map_idx] = std::get<2>(stage_tensor_map_it.second[req_tensor_map_idx]);
      }

      model_instance_ptr->Forward(stage_tensor_map_it.first, input_tensor_maps, output_tensor_maps);
    }
  }

  return Status();
}

}  // namespace numerous_llm
