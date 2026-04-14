/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "model_loader/weight_loader/model_weight_loader.h"
#include <future>

#include "model_loader/model_loader_utils.h"
#include "model_loader/weight_loader/model_weight_loader_factory.h"
#include "utils/memory_utils.h"
#include "utils/ret_code.h"

namespace ksana_llm {

ModelWeightLoader::ModelWeightLoader(std::shared_ptr<Environment> env, std::shared_ptr<Context> context) {
  env_ = env;
  context_ = context;

  weight_loader_threadpool_ = std::make_shared<ThreadPool>(context->global->GetTensorParallelSize());
  weight_loader_threadpool_->Start();
}

ModelWeightLoader::~ModelWeightLoader() { weight_loader_threadpool_->Stop(); }

Status ModelWeightLoader::LoadWeight(std::shared_ptr<BaseModelConfig>& model_config,
                                     std::shared_ptr<ModelWeight>& dev_weight) {
  const auto start_time = std::chrono::high_resolution_clock::now();
  std::shared_ptr<BaseModelWeightLoader> model_weight_loader;
  Status status = ModelWeightLoaderFactory::CreateModelWeightLoader(model_config->model_arch, model_config, env_,
                                                                    context_, model_weight_loader);
  STATUS_CHECK_RETURN(status);

  std::string model_type;
  status = GetModelTypeFromArchitecture(model_config->model_arch, model_type);
  STATUS_CHECK_RETURN(status);

  status = model_weight_loader->InitRegexPatterns(model_config->model_dir, model_type, model_config->model_format);
  STATUS_CHECK_RETURN(status);

  std::vector<std::string> model_file_list;
  status = GetModelFileList(model_config->model_dir, model_file_list);
  STATUS_CHECK_RETURN(status);

  status = FilterModelFormatFiles(model_config->model_format, model_file_list);
  STATUS_CHECK_RETURN(status);

  status = model_weight_loader->FilterModelFiles(model_file_list);
  STATUS_CHECK_RETURN(status);

  int device_id = context_->device->GetDeviceId();
  std::unordered_map<std::string, Tensor> left_host_weights;

  // file loader may use mmap, hold it until finish load weight
  std::vector<FileLoader> file_loaders;
  file_loaders.reserve(model_file_list.size());
  for (const std::string& model_file : model_file_list) {
    auto& file_loader = file_loaders.emplace_back(model_file);

    std::vector<std::string> weight_names;
    status = file_loader.LoadWeightNames(model_config->model_format, weight_names);
    STATUS_CHECK_RETURN(status);

    status = model_weight_loader->FilterWeightNames(weight_names);
    STATUS_CHECK_RETURN(status);

    auto host_model_weights = std::make_shared<std::unordered_map<std::string, Tensor>>();
    status = file_loader.LoadModelWeights(model_config->model_format, weight_names, *host_model_weights);
    STATUS_CHECK_RETURN(status);

    // Process common task for all tp devices.
    model_weight_loader->PreProcessModelWeights(*host_model_weights);

    // The current implementation can only process serially per card.
    std::unordered_map<std::string, Tensor> model_weights = *host_model_weights;

    // Strip VLM prefix "language_model." from weight names if detected.
    // Must happen before GetCustomWeightMap, whose regex_search would partially match prefixed names.
    {
      std::vector<std::string> keys_to_rename;
      const std::string vlm_prefix = "language_model.";
      for (const auto& [key, _] : model_weights) {
        if (key.substr(0, vlm_prefix.size()) == vlm_prefix) {
          keys_to_rename.push_back(key);
        }
      }
      if (!keys_to_rename.empty()) {
        KLLM_LOG_INFO << fmt::format("Stripping VLM prefix 'language_model.' from {} weights", keys_to_rename.size());
        for (const auto& old_key : keys_to_rename) {
          std::string new_key = old_key.substr(vlm_prefix.size());
          auto node = model_weights.extract(old_key);
          node.key() = std::move(new_key);
          model_weights.insert(std::move(node));
        }
      }
    }

    model_weight_loader->GetCustomWeightMap(model_config->model_dir, model_type, model_weights,
                                            model_config->model_format);

    model_weights.insert(left_host_weights.begin(), left_host_weights.end());
    left_host_weights.clear();
    model_weight_loader->ProcessModelWeights(model_weights, device_id, dev_weight->weights_map_, left_host_weights);
    StreamSynchronize(context_->device->GetMemoryManageStream());
  }

  // post process
  model_weight_loader->PostProcessModelWeights(dev_weight->weights_map_, device_id);
  StreamSynchronize(context_->device->GetMemoryManageStream());

  const std::chrono::duration<double> elapsed = std::chrono::high_resolution_clock::now() - start_time;
  KLLM_LOG_INFO << fmt::format("LoadWeights cost: {} seconds", elapsed.count());
  return Status();
}

}  // namespace ksana_llm
