/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/runtime/model_instance.h"
#include "numerous_llm/utils/logger.h"

namespace numerous_llm {

void ModelInstance::Load(const ModelConfig& model_config) {
  NLLM_LOG_INFO << "Start to load model " << model_config.name << std::endl;
}

void ModelInstance::Forward(const TensorMap& input_tensor_map, const SamplingConfig& sampling_config,
                            TensorMap& output_tensor_map) {
  // do infer ever layers

  // take result to output tensor map
}

}  // namespace numerous_llm
